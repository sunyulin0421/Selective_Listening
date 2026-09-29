import argparse
import json
import math
import os
import re
from typing import Any, Dict, List, Tuple

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoProcessor, set_seed

from mechanistic.common import move_to_device, write_json
from mechanistic.extract_qwen_logit_hidden_drift import (
    get_option_token_ids,
    load_qwen_thinker,
    option_margin,
    processor_call,
    score_option_logits,
)
from mechanistic.prefix_progressive_probe import (
    build_prompt_and_audio,
    build_stage_prefix,
    parse_fractions,
)
from utils.datasets import get_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generic module-output ablation for Qwen2.5-Omni. This probes "
            "whether audio/fusion/projector modules are causal for answer drift."
        )
    )
    parser.add_argument("--flip_path", required=True)
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--task_split", default="fsd")
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model_id", default="Qwen/Qwen2.5-Omni-7B")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--max_samples", type=int, default=128)
    parser.add_argument("--flip_class", default="")
    parser.add_argument(
        "--response_source",
        default="condition",
        choices=["clean", "condition"],
    )
    parser.add_argument("--fractions", default="0.75")
    parser.add_argument(
        "--module_groups",
        required=True,
        help=(
            "Semicolon-separated ablation groups: label=regex;label2=regex2. "
            "Regexes match module names and class names case-insensitively."
        ),
    )
    parser.add_argument(
        "--leaf_only",
        action="store_true",
        help="Only hook matched modules without children.",
    )
    parser.add_argument(
        "--max_modules_per_group",
        type=int,
        default=32,
        help="Safety cap for matched modules per group.",
    )
    parser.add_argument(
        "--ablation_scale",
        type=float,
        default=0.0,
        help="Output multiplier for ablated modules. 0 means zero ablation.",
    )
    parser.add_argument(
        "--ablation_scales",
        default=None,
        help=(
            "Optional comma-separated output multipliers, e.g. 0,0.25,0.5,0.75. "
            "When set, overrides --ablation_scale and evaluates every scale."
        ),
    )
    parser.add_argument("--max_prefix_chars", type=int, default=6000)
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_fast_processor", action="store_true")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    return parser.parse_args()


def parse_module_groups(value: str) -> List[Tuple[str, str]]:
    groups = []
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Module group must be label=regex, got: {part}")
        label, regex = part.split("=", 1)
        label = label.strip()
        regex = regex.strip()
        if not label or not regex:
            raise ValueError(f"Invalid module group: {part}")
        groups.append((label, regex))
    if not groups:
        raise ValueError("No valid module groups were provided")
    return groups


def first_tensor(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return output[0]
    raise TypeError(f"Unsupported output type for module ablation: {type(output)}")


def replace_first_tensor(output: Any, replacement: torch.Tensor) -> Any:
    if torch.is_tensor(output):
        return replacement
    if isinstance(output, tuple):
        return (replacement,) + output[1:]
    raise TypeError(f"Unsupported output type for module ablation: {type(output)}")


def has_children(module: nn.Module) -> bool:
    return any(True for _ in module.children())


def find_modules(
    model: nn.Module,
    regex: str,
    leaf_only: bool,
    max_modules: int,
) -> List[Tuple[str, nn.Module]]:
    pattern = re.compile(regex, re.IGNORECASE)
    matches = []
    for name, module in model.named_modules():
        if not name:
            continue
        if leaf_only and has_children(module):
            continue
        class_name = module.__class__.__name__
        text = f"{name} {class_name}"
        if pattern.search(name) or pattern.search(class_name) or pattern.search(text):
            matches.append((name, module))
    if len(matches) > max_modules:
        preview = [name for name, _module in matches[:80]]
        raise ValueError(
            f"Regex '{regex}' matched {len(matches)} modules, exceeding "
            f"--max_modules_per_group={max_modules}. Narrow the regex. "
            f"First matched module names: {preview}"
        )
    if not matches:
        raise ValueError(f"Regex '{regex}' did not match any modules")
    return matches


def make_ablation_hook(scale: float):
    def hook(_module, _inputs, output):
        hidden = first_tensor(output)
        return replace_first_tensor(output, hidden * scale)
    return hook


def run_forward_logits(
    model: nn.Module,
    processor: AutoProcessor,
    sample: Dict[str, Any],
    task_type: str,
    prefix: str,
    option_token_ids: Dict[str, List[int]],
    sampling_rate: int,
) -> Tuple[str, Dict[str, float]]:
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
        outputs = model(**batch, return_dict=True)
    logits = outputs.logits[0, -1]
    scores = score_option_logits(logits, option_token_ids)
    pred = max(scores.items(), key=lambda item: item[1])[0]
    return pred, scores


def run_ablated_condition(
    model: nn.Module,
    modules: List[Tuple[str, nn.Module]],
    scale: float,
    processor: AutoProcessor,
    sample: Dict[str, Any],
    prefix: str,
    option_token_ids: Dict[str, List[int]],
    sampling_rate: int,
) -> Tuple[str, Dict[str, float]]:
    handles = [
        module.register_forward_hook(make_ablation_hook(scale))
        for _name, module in modules
    ]
    try:
        return run_forward_logits(
            model=model,
            processor=processor,
            sample=sample,
            task_type="text_bench_interference",
            prefix=prefix,
            option_token_ids=option_token_ids,
            sampling_rate=sampling_rate,
        )
    finally:
        for handle in handles:
            handle.remove()


def finite_delta(a: float, b: float) -> float:
    if (
        isinstance(a, (int, float))
        and isinstance(b, (int, float))
        and math.isfinite(a)
        and math.isfinite(b)
    ):
        return b - a
    return math.nan


def parse_scales(value: str, fallback: float) -> List[float]:
    if value is None:
        return [fallback]
    scales = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        scales.append(float(part))
    if not scales:
        raise ValueError("--ablation_scales did not contain any valid values")
    return scales


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    fractions = parse_fractions(args.fractions)
    ablation_scales = parse_scales(args.ablation_scales, args.ablation_scale)
    module_group_specs = parse_module_groups(args.module_groups)

    with open(args.flip_path, "r", encoding="utf-8") as f:
        flip_data = json.load(f)
    samples = [
        sample for sample in flip_data["samples"]
        if sample["condition"] == args.condition
        and (not args.flip_class or sample.get("flip_class", "") == args.flip_class)
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

    module_groups = []
    for label, regex in module_group_specs:
        modules = find_modules(
            model=model,
            regex=regex,
            leaf_only=args.leaf_only,
            max_modules=args.max_modules_per_group,
        )
        module_groups.append((label, regex, modules))
        print(
            f"Module group {label}: matched {len(modules)} modules: "
            f"{[name for name, _module in modules]}"
        )

    clean_dataset = get_dataset(args.task_id, args.task_split, mitigate_prompt=False)
    cond_dataset = get_dataset(args.task_id, args.condition, mitigate_prompt=False)
    labels = ["A", "B", "C", "D", "E"] if "arc" in args.task_id else ["A", "B", "C", "D"]
    option_token_ids = get_option_token_ids(processor.tokenizer, labels)

    rows = []
    for item in tqdm(samples, desc="Module ablation"):
        idx = item["idx"]
        clean_sample = clean_dataset[idx]
        cond_sample = cond_dataset[idx]
        response = item[f"{args.response_source}_response"]
        gold = item["gold"]

        for fraction in fractions:
            prefix, prefix_info = build_stage_prefix(
                response=response,
                fraction=fraction,
                max_prefix_chars=args.max_prefix_chars,
            )
            clean_pred, clean_scores = run_forward_logits(
                model=model,
                processor=processor,
                sample=clean_sample,
                task_type="text_bench",
                prefix=prefix,
                option_token_ids=option_token_ids,
                sampling_rate=args.sampling_rate,
            )
            condition_pred, condition_scores = run_forward_logits(
                model=model,
                processor=processor,
                sample=cond_sample,
                task_type="text_bench_interference",
                prefix=prefix,
                option_token_ids=option_token_ids,
                sampling_rate=args.sampling_rate,
            )
            clean_margin = option_margin(clean_scores, gold)
            condition_margin = option_margin(condition_scores, gold)

            for scale in ablation_scales:
                for label, regex, modules in module_groups:
                    ablated_pred, ablated_scores = run_ablated_condition(
                        model=model,
                        modules=modules,
                        scale=scale,
                        processor=processor,
                        sample=cond_sample,
                        prefix=prefix,
                        option_token_ids=option_token_ids,
                        sampling_rate=args.sampling_rate,
                    )
                    ablated_margin = option_margin(ablated_scores, gold)
                    rows.append({
                        "idx": idx,
                        "condition": args.condition,
                        "flip_class": item.get("flip_class", ""),
                        "response_source": args.response_source,
                        "fraction": fraction,
                        "patch_group": label,
                        "patch_layer": label,
                        "patch_layers": [],
                        "patch_span": "module_output",
                        "patch_span_tokens": 0,
                        "clean_patch_token_count": 0,
                        "module_regex": regex,
                        "matched_modules": [name for name, _module in modules],
                        "num_matched_modules": len(modules),
                        "ablation_scale": scale,
                        "gold": gold,
                        "clean_jsonl_pred": item["clean_pred"],
                        "condition_jsonl_pred": item["condition_pred"],
                        "clean_forward_pred": clean_pred,
                        "condition_forward_pred": condition_pred,
                        "patched_forward_pred": ablated_pred,
                        "gold_margin_clean": clean_margin,
                        "gold_margin_condition": condition_margin,
                        "gold_margin_patched": ablated_margin,
                        "gold_margin_delta_condition_minus_clean": finite_delta(
                            clean_margin,
                            condition_margin,
                        ),
                        "gold_margin_delta_patched_minus_condition": finite_delta(
                            condition_margin,
                            ablated_margin,
                        ),
                        "clean_option_logits": clean_scores,
                        "condition_option_logits": condition_scores,
                        "patched_option_logits": ablated_scores,
                        "patch_restores_clean_pred": ablated_pred == clean_pred,
                        "patch_restores_gold": ablated_pred == gold,
                        "patch_changes_condition_pred": ablated_pred != condition_pred,
                        **prefix_info,
                    })

    output = {
        "model_id": args.model_id,
        "task_id": args.task_id,
        "task_split": args.task_split,
        "condition": args.condition,
        "flip_class_filter": args.flip_class,
        "response_source": args.response_source,
        "fractions": fractions,
        "layers": [],
        "patch_groups": [
            {
                "label": label,
                "regex": regex,
                "matched_modules": [name for name, _module in modules],
            }
            for label, regex, modules in module_groups
        ],
        "patch_last_k": 0,
        "patch_span": "module_output",
        "patch_spans": ["module_output"],
        "patch_span_tokens": 0,
        "patch_span_tokens_list": [0],
        "ablation_scale": args.ablation_scale,
        "ablation_scales": ablation_scales,
        "option_token_ids": option_token_ids,
        "max_prefix_chars": args.max_prefix_chars,
        "num_samples": len(samples),
        "num_rows": len(rows),
        "samples": rows,
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    write_json(args.output_path, output)
    print(f"Saved module ablation output to {args.output_path}")


if __name__ == "__main__":
    main()
