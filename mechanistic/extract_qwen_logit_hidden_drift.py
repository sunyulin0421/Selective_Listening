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
from utils.audio import decode_audio
from utils.datasets import get_dataset


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal mechanistic pilot for Qwen2.5-Omni: compare clean vs "
            "audio-interfered option logits and selected-layer hidden states."
        )
    )
    parser.add_argument("--flip_path", required=True)
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--task_split", default="fsd")
    parser.add_argument("--condition", required=True, help="fsd or speech_random")
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--layers", default="-1,-2,-4,-8")
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument(
        "--probe_prompt",
        default="direct",
        choices=["direct", "cot", "tf_clean", "tf_condition", "tf_own"],
        help=(
            "direct asks for only one answer letter, making next-token option "
            "logits meaningful; cot reuses the original chain-of-thought query; "
            "tf_clean/tf_condition/tf_own append a teacher-forced assistant "
            "reasoning prefix before the final answer letter."
        ),
    )
    parser.add_argument(
        "--max_teacher_chars",
        type=int,
        default=6000,
        help=(
            "Maximum assistant-prefix characters kept for teacher-forced probes. "
            "Use 0 to disable truncation."
        ),
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


def load_qwen_thinker(model_id: str, dtype: torch.dtype):
    try:
        from transformers import Qwen2_5OmniThinkerForConditionalGeneration
        model_cls = Qwen2_5OmniThinkerForConditionalGeneration
    except ImportError as exc:
        raise ImportError(
            "Cannot import Qwen2_5OmniThinkerForConditionalGeneration. "
            "Upgrade transformers to a version that supports Qwen2.5-Omni."
        ) from exc

    from utils.hf_local import resolve_local_snapshot

    model_source = resolve_local_snapshot(model_id)
    common_kwargs = {
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "local_files_only": True,
    }
    try:
        import accelerate  # noqa: F401
    except ImportError:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(
            "accelerate is unavailable; loading the model directly on "
            f"{device} without device_map=auto."
        )
        model = model_cls.from_pretrained(model_source, **common_kwargs)
        return model.to(device)

    return model_cls.from_pretrained(
        model_source,
        device_map="auto",
        **common_kwargs,
    )


def build_direct_answer_query(sample: Dict[str, Any], task_id: str) -> str:
    if "arc" in task_id:
        question = sample["question"]
        choices = sample["choices"]
        labels = choices.get("label", ["A", "B", "C", "D", "E"])
        texts = choices["text"]
    elif "mmlu" in task_id:
        question = sample["question"]
        texts = sample["choices"]
        labels = ["A", "B", "C", "D"]
    else:
        raise ValueError("Direct-answer probe currently supports ARC and MMLU.")

    valid_labels = labels[:len(texts)]
    query = (
        "Answer the following multiple choice question. "
        f"Respond with only one letter from {', '.join(valid_labels)}.\n\n"
        f"{question.strip()}\n"
    )
    for label, text in zip(valid_labels, texts):
        query += f"\n{label}) {text}"
    query += "\n\nAnswer:"
    return query


def stringify_response(response: Any) -> str:
    if response is None:
        return ""
    if isinstance(response, list):
        return "\n".join(str(item) for item in response)
    return str(response)


def build_teacher_forced_prefix(response: Any, max_chars: int) -> str:
    text = stringify_response(response).strip()
    if not text:
        return "Answer:"

    # Keep the model's own reasoning, but stop immediately before the final
    # answer letter so the next-token logits score A/B/C/D/E directly.
    final_answer = find_final_mc_answer_letter_span(text)
    if final_answer is not None:
        prefix = text[:final_answer[0]]
    else:
        prefix = f"{text}\n\nAnswer:"

    if max_chars > 0 and len(prefix) > max_chars:
        prefix = prefix[-max_chars:]
    return prefix


def select_teacher_response(
    item: Dict[str, Any],
    task_type: str,
    probe_prompt: str,
) -> Tuple[Optional[Any], Optional[str]]:
    if probe_prompt == "tf_clean":
        return item.get("clean_response", ""), "clean_response"
    if probe_prompt == "tf_condition":
        return item.get("condition_response", ""), "condition_response"
    if probe_prompt == "tf_own":
        if task_type == "text_bench":
            return item.get("clean_response", ""), "clean_response"
        return item.get("condition_response", ""), "condition_response"
    return None, None


def build_prompt_and_audio(
    sample: Dict[str, Any],
    processor: AutoProcessor,
    task_type: str,
    sampling_rate: int,
    task_id: str,
    probe_prompt: str,
    teacher_response: Optional[Any],
    max_teacher_chars: int,
) -> Tuple[str, Optional[List[Any]]]:
    if probe_prompt == "direct":
        query = build_direct_answer_query(sample, task_id)
    else:
        query = sample["query"]
    if task_type == "text_bench":
        content = [{"type": "text", "text": query}]
        audios = None
    else:
        content = [
            {"type": "audio", "audio_url": None},
            {"type": "text", "text": query},
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
    if probe_prompt.startswith("tf_"):
        prompt += build_teacher_forced_prefix(teacher_response, max_teacher_chars)
    return prompt, audios


def processor_call(
    processor: AutoProcessor,
    prompt: str,
    audios: Optional[List[Any]],
) -> Dict[str, Any]:
    if audios is None:
        return processor(text=[prompt], return_tensors="pt", padding=True)

    try:
        return processor(
            text=[prompt],
            audio=audios,
            return_tensors="pt",
            padding=True,
        )
    except TypeError:
        return processor(
            text=[prompt],
            audios=audios,
            return_tensors="pt",
            padding=True,
        )


def get_option_token_ids(tokenizer: Any, labels: List[str]) -> Dict[str, List[int]]:
    option_ids = {}
    for label in labels:
        candidates = [label, f" {label}", f": {label}", f"\n{label}"]
        ids = []
        for candidate in candidates:
            tokenized = tokenizer(candidate, add_special_tokens=False)["input_ids"]
            if tokenized:
                ids.append(tokenized[-1])
        option_ids[label] = sorted(set(ids))
    return option_ids


def score_option_logits(
    logits: torch.Tensor,
    option_token_ids: Dict[str, List[int]],
) -> Dict[str, float]:
    scores = {}
    logits = logits.detach().float().cpu()
    for label, token_ids in option_token_ids.items():
        if not token_ids:
            scores[label] = float("-inf")
            continue
        scores[label] = float(torch.max(logits[token_ids]).item())
    return scores


def option_margin(scores: Dict[str, float], gold: str) -> float:
    if gold not in scores:
        return math.nan
    competitors = [v for k, v in scores.items() if k != gold]
    if not competitors:
        return math.nan
    return scores[gold] - max(competitors)


def summarize_forward(
    model: torch.nn.Module,
    processor: AutoProcessor,
    sample: Dict[str, Any],
    item: Dict[str, Any],
    task_type: str,
    layers: List[int],
    option_token_ids: Dict[str, List[int]],
    sampling_rate: int,
    task_id: str,
    probe_prompt: str,
    max_teacher_chars: int,
) -> Dict[str, Any]:
    teacher_response, teacher_response_source = select_teacher_response(
        item=item,
        task_type=task_type,
        probe_prompt=probe_prompt,
    )
    prompt, audios = build_prompt_and_audio(
        sample=sample,
        processor=processor,
        task_type=task_type,
        sampling_rate=sampling_rate,
        task_id=task_id,
        probe_prompt=probe_prompt,
        teacher_response=teacher_response,
        max_teacher_chars=max_teacher_chars,
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
        "teacher_response_source": teacher_response_source,
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]

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
    for item in tqdm(samples, desc="Mechanistic forward"):
        idx = item["idx"]
        clean_sample = clean_dataset[idx]
        cond_sample = cond_dataset[idx]

        clean_summary = summarize_forward(
            model=model,
            processor=processor,
            sample=clean_sample,
            item=item,
            task_type="text_bench",
            layers=layers,
            option_token_ids=option_token_ids,
            sampling_rate=args.sampling_rate,
            task_id=args.task_id,
            probe_prompt=args.probe_prompt,
            max_teacher_chars=args.max_teacher_chars,
        )
        cond_summary = summarize_forward(
            model=model,
            processor=processor,
            sample=cond_sample,
            item=item,
            task_type="text_bench_interference",
            layers=layers,
            option_token_ids=option_token_ids,
            sampling_rate=args.sampling_rate,
            task_id=args.task_id,
            probe_prompt=args.probe_prompt,
            max_teacher_chars=args.max_teacher_chars,
        )

        gold = item["gold"]
        rows.append({
            "idx": idx,
            "condition": args.condition,
            "flip_class": item.get("flip_class", ""),
            "probe_prompt": args.probe_prompt,
            "gold": gold,
            "clean_jsonl_pred": item["clean_pred"],
            "condition_jsonl_pred": item["condition_pred"],
            "clean_forward_pred": clean_summary["pred"],
            "condition_forward_pred": cond_summary["pred"],
            "clean_teacher_response_source": clean_summary["teacher_response_source"],
            "condition_teacher_response_source": cond_summary["teacher_response_source"],
            "clean_option_logits": clean_summary["option_logits"],
            "condition_option_logits": cond_summary["option_logits"],
            "gold_margin_clean": option_margin(clean_summary["option_logits"], gold),
            "gold_margin_condition": option_margin(cond_summary["option_logits"], gold),
            "hidden_drift": serialize_hidden_comparison(
                clean_summary["hidden"],
                cond_summary["hidden"],
            ),
        })

    output = {
        "model_id": args.model_id,
        "task_id": args.task_id,
        "task_split": args.task_split,
        "condition": args.condition,
        "probe_prompt": args.probe_prompt,
        "max_teacher_chars": args.max_teacher_chars,
        "layers": layers,
        "option_token_ids": option_token_ids,
        "num_samples": len(rows),
        "samples": rows,
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    write_json(args.output_path, output)
    print(f"Saved mechanistic pilot output to {args.output_path}")


if __name__ == "__main__":
    main()
