import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import librosa
import torch
from tqdm import tqdm
from transformers import AutoProcessor, set_seed

from mechanistic.common import (find_final_mc_answer_letter_span, move_to_device,
                                parse_layer_list, tensor_cosine_distance, tensor_l2,
                                write_json)
from mechanistic.extract_qwen_logit_hidden_drift import (
    SYSTEM_PROMPT,
    get_option_token_ids,
    load_qwen_thinker,
    option_margin,
    processor_call,
    score_option_logits,
    stringify_response,
)
from utils.audio import decode_audio
from utils.datasets import get_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prefix-progressive mechanistic probe for Qwen2.5-Omni. It fixes "
            "a generated reasoning prefix at several progress points and "
            "compares clean vs audio-conditioned final-answer logits."
        )
    )
    parser.add_argument("--flip_path", required=True)
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--task_split", default="fsd")
    parser.add_argument("--condition", required=True, help="fsd or speech_random")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--layers", default="-1,-2,-4")
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument(
        "--response_source",
        default="condition",
        choices=["clean", "condition"],
        help="Which JSONL response supplies the progressive reasoning prefix.",
    )
    parser.add_argument(
        "--fractions",
        default="0,0.25,0.5,0.75,1.0",
        help="Comma-separated prefix fractions. 1.0 means before final answer letter.",
    )
    parser.add_argument(
        "--max_prefix_chars",
        type=int,
        default=6000,
        help="Maximum prefix characters kept before the final Answer: probe.",
    )
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_fast_processor", action="store_true")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    return parser.parse_args()


def parse_fractions(value: str) -> List[float]:
    fractions = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        fraction = float(part)
        if fraction < 0 or fraction > 1:
            raise ValueError(f"Prefix fraction must be in [0, 1], got {fraction}")
        fractions.append(fraction)
    if not fractions:
        raise ValueError("At least one prefix fraction is required")
    return fractions


def split_response_for_answer_probe(response: Any) -> Tuple[str, str, bool]:
    text = stringify_response(response).strip()
    if not text:
        return "", "Answer:", False

    final_answer = find_final_mc_answer_letter_span(text)
    if final_answer is None:
        return text, f"{text}\n\nAnswer:", False

    answer_start, _ = final_answer
    reasoning = text[:answer_start].strip()
    final_answer_prefix = text[:answer_start]
    return reasoning, final_answer_prefix, True


def truncate_prefix(prefix: str, max_prefix_chars: int) -> str:
    if max_prefix_chars > 0 and len(prefix) > max_prefix_chars:
        return prefix[-max_prefix_chars:]
    return prefix


def build_stage_prefix(
    response: Any,
    fraction: float,
    max_prefix_chars: int,
) -> Tuple[str, Dict[str, Any]]:
    reasoning, final_answer_prefix, found_answer = split_response_for_answer_probe(response)
    if fraction >= 1.0:
        prefix = final_answer_prefix
    else:
        cut = int(round(len(reasoning) * fraction))
        partial = reasoning[:cut].strip()
        prefix = "Answer:" if not partial else f"{partial}\n\nAnswer:"

    prefix = truncate_prefix(prefix, max_prefix_chars)
    return prefix, {
        "fraction": fraction,
        "found_answer_marker": found_answer,
        "reasoning_chars": len(reasoning),
        "prefix_chars": len(prefix),
    }


def build_prompt_and_audio(
    sample: Dict[str, Any],
    processor: AutoProcessor,
    task_type: str,
    sampling_rate: int,
    prefix: str,
) -> Tuple[str, Optional[List[Any]]]:
    if task_type == "text_bench":
        content = [{"type": "text", "text": sample["query"]}]
        audios = None
    else:
        content = [
            {"type": "audio", "audio_url": None},
            {"type": "text", "text": sample["query"]},
        ]
        audio_array, audio_sr = decode_audio(sample["audio"])
        audios = [
            librosa.resample(
                audio_array,
                orig_sr=audio_sr,
                target_sr=sampling_rate,
            )
        ]

    prompt = processor.apply_chat_template(
        [
            {
                "role": "system",
                "content": [{"type": "text", "text": SYSTEM_PROMPT}],
            },
            {"role": "user", "content": content},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    return prompt + prefix, audios


def summarize_forward(
    model: torch.nn.Module,
    processor: AutoProcessor,
    sample: Dict[str, Any],
    task_type: str,
    prefix: str,
    layers: List[int],
    option_token_ids: Dict[str, List[int]],
    sampling_rate: int,
) -> Dict[str, Any]:
    prompt, audios = build_prompt_and_audio(
        sample=sample,
        processor=processor,
        task_type=task_type,
        sampling_rate=sampling_rate,
        prefix=prefix,
    )
    batch = processor_call(processor, prompt, audios)
    device = next(model.parameters()).device
    batch = move_to_device(batch, device)

    with torch.inference_mode():
        outputs = model(
            **batch,
            output_hidden_states=True,
            return_dict=True,
        )

    logits = outputs.logits[0, -1]
    option_scores = score_option_logits(logits, option_token_ids)
    pred = max(option_scores.items(), key=lambda item: item[1])[0]

    hidden_states = outputs.hidden_states
    hidden_summary = {}
    for layer in layers:
        hidden = hidden_states[layer][0]
        hidden_summary[str(layer)] = {
            "last_token": hidden[-1].detach().float().cpu(),
            "mean": hidden.mean(dim=0).detach().float().cpu(),
        }

    return {
        "pred": pred,
        "option_logits": option_scores,
        "hidden": hidden_summary,
    }


def serialize_hidden_comparison(
    clean_hidden: Dict[str, Dict[str, torch.Tensor]],
    cond_hidden: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, Any]:
    comparison = {}
    for layer in clean_hidden:
        comparison[layer] = {}
        for key in clean_hidden[layer]:
            a = clean_hidden[layer][key]
            b = cond_hidden[layer][key]
            comparison[layer][key] = {
                "l2": tensor_l2(a, b),
                "cosine_distance": tensor_cosine_distance(a, b),
            }
    return comparison


def finite_margin_delta(clean_margin: float, condition_margin: float) -> float:
    if (
        isinstance(clean_margin, (int, float))
        and isinstance(condition_margin, (int, float))
        and math.isfinite(clean_margin)
        and math.isfinite(condition_margin)
    ):
        return condition_margin - clean_margin
    return math.nan


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    fractions = parse_fractions(args.fractions)

    with open(args.flip_path, "r", encoding="utf-8") as f:
        flip_data = json.load(f)
    samples = [
        sample for sample in flip_data["samples"]
        if sample["condition"] == args.condition
    ][:args.max_samples]
    if not samples:
        raise ValueError(f"No samples found for condition '{args.condition}'")

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=args.use_fast_processor,
    )
    model = load_qwen_thinker(args.model_id, dtype=dtype)
    model.eval()

    clean_dataset = get_dataset(args.task_id, args.task_split, mitigate_prompt=False)
    cond_dataset = get_dataset(args.task_id, args.condition, mitigate_prompt=False)

    labels = ["A", "B", "C", "D", "E"] if "arc" in args.task_id else ["A", "B", "C", "D"]
    option_token_ids = get_option_token_ids(processor.tokenizer, labels)
    num_layers = getattr(model.config, "num_hidden_layers", None)
    layers = parse_layer_list(args.layers, num_layers=num_layers)

    rows = []
    for item in tqdm(samples, desc="Prefix-progressive forward"):
        idx = item["idx"]
        clean_sample = clean_dataset[idx]
        cond_sample = cond_dataset[idx]
        response = item[f"{args.response_source}_response"]

        for fraction in fractions:
            prefix, prefix_info = build_stage_prefix(
                response=response,
                fraction=fraction,
                max_prefix_chars=args.max_prefix_chars,
            )
            clean_summary = summarize_forward(
                model=model,
                processor=processor,
                sample=clean_sample,
                task_type="text_bench",
                prefix=prefix,
                layers=layers,
                option_token_ids=option_token_ids,
                sampling_rate=args.sampling_rate,
            )
            cond_summary = summarize_forward(
                model=model,
                processor=processor,
                sample=cond_sample,
                task_type="text_bench_interference",
                prefix=prefix,
                layers=layers,
                option_token_ids=option_token_ids,
                sampling_rate=args.sampling_rate,
            )

            gold = item["gold"]
            clean_margin = option_margin(clean_summary["option_logits"], gold)
            cond_margin = option_margin(cond_summary["option_logits"], gold)
            rows.append({
                "idx": idx,
                "condition": args.condition,
                "flip_class": item.get("flip_class", ""),
                "response_source": args.response_source,
                "fraction": fraction,
                "gold": gold,
                "clean_jsonl_pred": item["clean_pred"],
                "condition_jsonl_pred": item["condition_pred"],
                "clean_forward_pred": clean_summary["pred"],
                "condition_forward_pred": cond_summary["pred"],
                "gold_margin_clean": clean_margin,
                "gold_margin_condition": cond_margin,
                "gold_margin_delta_condition_minus_clean": finite_margin_delta(
                    clean_margin,
                    cond_margin,
                ),
                "clean_option_logits": clean_summary["option_logits"],
                "condition_option_logits": cond_summary["option_logits"],
                "hidden_drift": serialize_hidden_comparison(
                    clean_summary["hidden"],
                    cond_summary["hidden"],
                ),
                **prefix_info,
            })

    output = {
        "model_id": args.model_id,
        "task_id": args.task_id,
        "task_split": args.task_split,
        "condition": args.condition,
        "response_source": args.response_source,
        "fractions": fractions,
        "layers": layers,
        "option_token_ids": option_token_ids,
        "max_prefix_chars": args.max_prefix_chars,
        "num_samples": len(samples),
        "num_rows": len(rows),
        "samples": rows,
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    write_json(args.output_path, output)
    print(f"Saved prefix-progressive output to {args.output_path}")


if __name__ == "__main__":
    main()
