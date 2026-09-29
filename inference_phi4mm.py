import json
import os
from argparse import ArgumentParser, Namespace
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from huggingface_hub import snapshot_download
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoProcessor, set_seed
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from mechanistic.ablate_qwen_modules import (
    find_modules,
    first_tensor,
    parse_module_groups,
    replace_first_tensor,
)
from mechanistic.common import move_to_device
from mechanistic.inspect_hf_modules import (
    force_attention_backend,
    patch_prepare_inputs_for_generation,
)
from utils.audio import decode_audio
from utils.common import save_jsonl
from utils.datasets import get_dataset


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "HF remote-class Phi-4-MM inference with fixed soft gates on "
            "candidate audio modules. Intended for small mechanistic pilots."
        )
    )
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--task_split", required=True)
    parser.add_argument("--task_type", required=True)
    parser.add_argument("--model_id", default="microsoft/Phi-4-multimodal-instruct")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--mitigate_prompt", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument(
        "--use_cache",
        action="store_true",
        help=(
            "Enable generation KV cache. Disabled by default because the cached "
            "Phi-4-MM remote code may expect an older transformers cache API."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--sample_strategy",
        choices=["first", "random"],
        default="first",
    )
    parser.add_argument("--sample_seed", type=int, default=None)
    parser.add_argument(
        "--module_groups",
        default="",
        help=(
            "Semicolon-separated gates: label=regex;label2=regex2. "
            "Example: proj=^model\\.embed_tokens_extend\\.audio_embed\\.audio_projection\\.speech$"
        ),
    )
    parser.add_argument("--gate_scale", type=float, default=1.0)
    parser.add_argument(
        "--dynamic_gate_mode",
        choices=["none", "query_audio_relevance"],
        default="none",
        help=(
            "Sample-level relevance gate. query_audio_relevance keeps audio "
            "modules open when the query instruction asks for audio "
            "understanding and suppresses them otherwise."
        ),
    )
    parser.add_argument("--dynamic_low_scale", type=float, default=0.25)
    parser.add_argument("--dynamic_high_scale", type=float, default=1.0)
    parser.add_argument(
        "--query_relevance_keywords",
        default=(
            "audio,speech,sound,voice,spoken,listen,hear,hearing,transcribe,"
            "transcription,recording,speaker,asr"
        ),
    )
    parser.add_argument(
        "--query_relevance_scope",
        choices=["full_query", "instruction", "task_content"],
        default="instruction",
    )
    parser.add_argument("--max_modules_per_group", type=int, default=8)
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--device_map", default="auto")
    parser.add_argument(
        "--attn_implementation",
        choices=["eager", "sdpa", "flash_attention_2"],
        default="eager",
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--remote_class",
        default="modeling_phi4mm.Phi4MMForCausalLM",
    )
    parser.add_argument(
        "--apply_speech_lora",
        action="store_true",
        help="Apply the checkpoint's speech-lora adapter through PEFT.",
    )
    parser.add_argument(
        "--speech_lora_path",
        default=None,
        help="Optional explicit speech LoRA path. Defaults to <snapshot>/speech-lora.",
    )
    parser.add_argument(
        "--stop_strings",
        nargs="*",
        default=["<|end|>", "Human:", "User:", "Assistant:"],
    )
    parser.add_argument(
        "--truncate_response_mode",
        choices=["none", "first_human", "first_answer_line"],
        default="first_human",
    )
    parser.add_argument(
        "--stream_output",
        action="store_true",
        help="Write each result immediately instead of saving only after all samples finish.",
    )
    return parser.parse_args()


class GateController:
    def __init__(self, scale: float):
        self.scale = scale
        self.matched_modules: List[str] = []


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def make_gate_hook(controller: GateController):
    def hook(_module, _inputs, output):
        hidden = first_tensor(output)
        return replace_first_tensor(output, hidden * controller.scale)

    return hook


def exact_regex_to_suffix(regex: str) -> Optional[str]:
    if not (regex.startswith("^") and regex.endswith("$")):
        return None
    body = regex[1:-1]
    for special in ["[", "]", "(", ")", "|", "+", "*", "?", "{", "}"]:
        if special in body:
            return None
    return body.replace(r"\.", ".").replace(r"\\", "\\")


def find_modules_for_gate(
    model: nn.Module,
    regex: str,
    max_modules: int,
) -> Tuple[List[Tuple[str, nn.Module]], Optional[str]]:
    try:
        return find_modules(
            model=model,
            regex=regex,
            leaf_only=False,
            max_modules=max_modules,
        ), None
    except ValueError as exc:
        if "did not match any modules" not in str(exc):
            raise

    suffix = exact_regex_to_suffix(regex)
    if suffix:
        suffixes = [suffix]
        if suffix.startswith("model."):
            suffixes.append(suffix[len("model."):])
        matches = [
            (name, module)
            for name, module in model.named_modules()
            if any(name == item or name.endswith("." + item) for item in suffixes)
        ]
        if matches:
            if len(matches) > max_modules:
                names = [name for name, _ in matches[:max_modules]]
                raise ValueError(
                    f"Suffix fallback for regex '{regex}' matched {len(matches)} modules, "
                    f"exceeding --max_modules_per_group={max_modules}. "
                    f"First matched module names: {names}"
                )
            return matches, f"suffix {suffix!r}"

    fallback_regex = regex
    if regex.startswith("^"):
        fallback_regex = f"(^|.*\\.){regex[1:]}"
    return find_modules(
        model=model,
        regex=fallback_regex,
        leaf_only=False,
        max_modules=max_modules,
    ), f"regex {fallback_regex!r}"


def relevance_text_span(query: str, scope: str) -> str:
    normalized = query.replace("\r\n", "\n")
    if scope == "full_query":
        return normalized
    if scope == "instruction":
        return normalized.split("\n\n", 1)[0]
    if scope == "task_content":
        parts = normalized.split("\n\n", 1)
        return parts[1] if len(parts) == 2 else normalized
    raise ValueError(f"Unsupported query relevance scope: {scope}")


def query_audio_relevance(query: str, keywords: str, scope: str) -> Dict[str, Any]:
    relevance_text = relevance_text_span(query, scope)
    lowered = relevance_text.lower()
    keyword_list = [
        keyword.strip().lower()
        for keyword in keywords.split(",")
        if keyword.strip()
    ]
    matched = [keyword for keyword in keyword_list if keyword in lowered]
    return {
        "audio_relevant": bool(matched),
        "matched_relevance_keywords": matched,
        "query_relevance_keywords": keyword_list,
        "query_relevance_scope": scope,
        "query_relevance_text": relevance_text,
    }


def choose_gate_scale(args: Namespace, relevance: Dict[str, Any]) -> float:
    if args.dynamic_gate_mode == "none":
        return args.gate_scale
    if args.dynamic_gate_mode == "query_audio_relevance":
        return args.dynamic_high_scale if relevance["audio_relevant"] else args.dynamic_low_scale
    raise ValueError(f"Unsupported dynamic_gate_mode: {args.dynamic_gate_mode}")


def install_gates(
    model: nn.Module,
    module_groups: str,
    controller: GateController,
    max_modules_per_group: int,
) -> List[Any]:
    if not module_groups.strip():
        controller.matched_modules = []
        return []

    handles = []
    matched_module_names: List[str] = []
    for label, regex in parse_module_groups(module_groups):
        modules, fallback = find_modules_for_gate(
            model=model,
            regex=regex,
            max_modules=max_modules_per_group,
        )
        if fallback:
            print(
                f"Gate group {label}: original regex did not match after wrappers; "
                f"using fallback {fallback}"
            )
        print(
            f"Gate group {label}: scale={controller.scale}, "
            f"matched {len(modules)} modules: {[name for name, _ in modules]}"
        )
        for name, module in modules:
            matched_module_names.append(name)
            handles.append(module.register_forward_hook(make_gate_hook(controller)))
    controller.matched_modules = matched_module_names
    return handles


def load_phi4mm_model(args: Namespace) -> nn.Module:
    config = AutoConfig.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if args.attn_implementation:
        force_attention_backend(config, args.attn_implementation)

    model_cls = get_class_from_dynamic_module(
        args.remote_class,
        args.model_id,
        local_files_only=args.local_files_only,
    )
    patch_prepare_inputs_for_generation(model_cls)
    # Phi-4-MM creates PEFT LoRA wrappers around the inner Phi4MMModel during
    # __init__. PEFT expects that inner base model to expose this generation
    # helper before the outer Phi4MMForCausalLM object exists.
    try:
        inner_model_cls = get_class_from_dynamic_module(
            "modeling_phi4mm.Phi4MMModel",
            args.model_id,
            local_files_only=args.local_files_only,
        )
        patch_prepare_inputs_for_generation(inner_model_cls)
    except Exception as exc:
        print(f"Warning: failed to patch Phi4MMModel.prepare_inputs_for_generation: {exc}")

    kwargs: Dict[str, Any] = {
        "config": config,
        "torch_dtype": dtype_from_name(args.dtype),
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.device_map:
        kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    model = model_cls.from_pretrained(args.model_id, **kwargs)
    if args.apply_speech_lora:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                "--apply_speech_lora requires peft. Install with `python -m pip install peft`."
            ) from exc
        lora_path = args.speech_lora_path
        if lora_path is None:
            snapshot_path = snapshot_download(
                args.model_id,
                local_files_only=args.local_files_only,
            )
            lora_path = os.path.join(snapshot_path, "speech-lora")
        if not os.path.isdir(lora_path):
            raise FileNotFoundError(f"Speech LoRA directory not found: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path)

    model.eval()
    return model


def get_inputs(
    sample: Dict[str, Any],
    task_type: str,
    sampling_rate: int,
) -> Tuple[str, Optional[List[Tuple[np.ndarray, int]]]]:
    query = sample["query"]
    if task_type == "text_bench":
        return f"<|user|>{query}<|end|><|assistant|>", None

    audio_array, audio_sr = decode_audio(sample["audio"])
    audio_array = librosa.resample(
        audio_array,
        orig_sr=audio_sr,
        target_sr=sampling_rate,
    )
    return f"<|user|><|audio_1|>{query}<|end|><|assistant|>", [(audio_array, sampling_rate)]


def processor_call_phi(
    processor: AutoProcessor,
    prompt: str,
    audios: Optional[List[Tuple[np.ndarray, int]]],
) -> Dict[str, torch.Tensor]:
    attempts = []
    if audios is None:
        attempts.extend(
            [
                lambda: processor(text=prompt, return_tensors="pt"),
                lambda: processor(prompt, return_tensors="pt"),
            ]
        )
    else:
        attempts.extend(
            [
                lambda: processor(
                    text=prompt,
                    audios=audios,
                    return_tensors="pt",
                ),
                lambda: processor(
                    prompt,
                    audios=audios,
                    return_tensors="pt",
                ),
            ]
        )

    errors = []
    for attempt in attempts:
        try:
            return attempt()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    raise RuntimeError("All Phi-4-MM processor calls failed:\n" + "\n".join(errors))


def decode_generation(
    processor: AutoProcessor,
    input_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    stop_strings: Optional[List[str]],
    truncate_response_mode: str,
) -> str:
    new_tokens = generated_ids[0, input_ids.shape[-1]:]
    tokenizer = getattr(processor, "tokenizer", processor)
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if stop_strings:
        cut = len(text)
        for stop in stop_strings:
            if not stop:
                continue
            pos = text.find(stop)
            if pos >= 0:
                cut = min(cut, pos)
        text = text[:cut].strip()
    if truncate_response_mode == "first_human":
        for marker in ["\nHuman:", "\nUser:", "\nAssistant:", "Human:", "User:", "Assistant:"]:
            pos = text.find(marker)
            if pos >= 0:
                text = text[:pos].strip()
    elif truncate_response_mode == "first_answer_line":
        lines = []
        for line in text.splitlines():
            lines.append(line)
            if "answer:" in line.lower():
                break
        text = "\n".join(lines).strip()
    return text


def generate_one(
    model: nn.Module,
    processor: AutoProcessor,
    sample: Dict[str, Any],
    args: Namespace,
    controller: GateController,
) -> Tuple[str, Dict[str, Any]]:
    prompt, audios = get_inputs(sample, args.task_type, args.sampling_rate)

    gate_info: Dict[str, Any] = {
        "dynamic_gate_mode": args.dynamic_gate_mode,
        "gate_scale": args.gate_scale,
        "applied_gate_scale": controller.scale,
        "module_groups": args.module_groups,
        "matched_modules": list(controller.matched_modules),
    }
    if args.dynamic_gate_mode == "query_audio_relevance" and args.task_type != "text_bench":
        relevance = query_audio_relevance(
            query=sample["query"],
            keywords=args.query_relevance_keywords,
            scope=args.query_relevance_scope,
        )
        controller.scale = choose_gate_scale(args, relevance)
        gate_info = {
            "dynamic_gate_mode": args.dynamic_gate_mode,
            "dynamic_low_scale": args.dynamic_low_scale,
            "dynamic_high_scale": args.dynamic_high_scale,
            "applied_gate_scale": controller.scale,
            "gate_decision": "high_scale" if relevance["audio_relevant"] else "low_scale",
            "module_groups": args.module_groups,
            "matched_modules": list(controller.matched_modules),
            **relevance,
        }
    else:
        controller.scale = args.gate_scale
        gate_info["applied_gate_scale"] = controller.scale

    batch = processor_call_phi(processor, prompt, audios)
    device = next(model.parameters()).device
    batch = move_to_device(batch, device)

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "use_cache": args.use_cache,
        # Phi-4-MM remote code indexes hidden_states[:, -num_logits_to_keep:, :].
        # Newer transformers generation may pass None here, which breaks that
        # unary minus. One token is sufficient for autoregressive generation.
        "num_logits_to_keep": 1,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    with torch.inference_mode():
        generated_ids = model.generate(**batch, **generation_kwargs)
    return decode_generation(
        processor,
        batch["input_ids"],
        generated_ids,
        args.stop_strings,
        args.truncate_response_mode,
    ), gate_info


def json_safe_metadata(sample: Dict[str, Any]) -> Dict[str, Any]:
    metadata = {}
    for key, value in sample.items():
        if key in {
            "audio",
            "query",
            "prompt",
            "question",
            "choices",
            "answer",
            "response",
        }:
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            metadata[key] = value
    return metadata


def generate_responses(args: Namespace) -> List[Dict[str, Any]]:
    if args.dynamic_gate_mode != "none" and not args.module_groups.strip():
        raise ValueError(
            "Dynamic gating requires --module_groups so the selected scale is "
            "applied to a real model module."
        )
    if args.dynamic_gate_mode == "query_audio_relevance" and args.task_type == "text_bench":
        raise ValueError(
            "--dynamic_gate_mode query_audio_relevance is intended for "
            "text_bench_interference samples with audio. Use --dynamic_gate_mode none "
            "for clean text-only runs."
        )
    dataset = get_dataset(args.task_id, args.task_split, args.mitigate_prompt)
    selected_indices = list(range(len(dataset)))
    if args.max_samples is not None:
        sample_count = min(args.max_samples, len(dataset))
        if args.sample_strategy == "first":
            selected_indices = list(range(sample_count))
        else:
            rng = np.random.default_rng(args.sample_seed if args.sample_seed is not None else args.seed)
            selected_indices = rng.choice(len(dataset), size=sample_count, replace=False).tolist()
        dataset = dataset.select(selected_indices)

    processor = AutoProcessor.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model = load_phi4mm_model(args)
    initial_scale = 1.0 if args.dynamic_gate_mode != "none" else args.gate_scale
    controller = GateController(initial_scale)
    handles = install_gates(
        model=model,
        module_groups=args.module_groups,
        controller=controller,
        max_modules_per_group=args.max_modules_per_group,
    )
    if args.dynamic_gate_mode != "none" and not handles:
        raise RuntimeError("Dynamic gating installed no forward hooks.")

    run_config = {
        "backend": "transformers_remote_code",
        "model_id": args.model_id,
        "task_id": args.task_id,
        "task_split": args.task_split,
        "task_type": args.task_type,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": args.use_cache,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "sample_strategy": args.sample_strategy,
        "sample_seed": args.sample_seed,
        "module_groups": args.module_groups,
        "matched_modules": list(controller.matched_modules),
        "gate_scale": args.gate_scale,
        "dynamic_gate_mode": args.dynamic_gate_mode,
        "dynamic_low_scale": args.dynamic_low_scale,
        "dynamic_high_scale": args.dynamic_high_scale,
        "query_relevance_scope": args.query_relevance_scope,
        "query_relevance_keywords": args.query_relevance_keywords,
        "truncate_response_mode": args.truncate_response_mode,
        "stop_strings": args.stop_strings,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "attn_implementation": args.attn_implementation,
        "apply_speech_lora": args.apply_speech_lora,
    }

    results = []
    stream_handle = None
    if args.stream_output:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        stream_handle = open(args.output_path, "w", encoding="utf-8")
    try:
        for row_idx, sample in enumerate(tqdm(dataset, desc="Phi-4-MM gated inference")):
            response, gate_info = generate_one(model, processor, sample, args, controller)
            metadata = json_safe_metadata(sample)
            result = {
                "source_index": selected_indices[row_idx],
                "id": sample.get("id", ""),
                "subject": sample.get("subject", ""),
                "task": sample.get("task", ""),
                "prompt": sample.get("prompt", ""),
                "query": sample["query"],
                "prefix": sample.get("prefix", ""),
                "choices": sample.get("choices", ""),
                "response": response,
                "answer": sample["answer"],
                "gate_info": gate_info,
                "metadata": metadata,
                "run_config": run_config,
            }
            for metadata_key in [
                "speech_source",
                "speech_index",
                "audio_source",
                "audio_index",
                "speaker_id",
                "utterance_id",
            ]:
                if metadata_key in metadata:
                    result[metadata_key] = metadata[metadata_key]
            if stream_handle is not None:
                stream_handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                stream_handle.flush()
            else:
                results.append(result)
    finally:
        if stream_handle is not None:
            stream_handle.close()
        for handle in handles:
            handle.remove()
    return results


def main(args: Namespace) -> None:
    set_seed(args.seed)
    if args.task_type not in {"text_bench", "text_bench_interference"}:
        raise ValueError(f"Unsupported task_type: {args.task_type}")
    results = generate_responses(args)
    if not args.stream_output:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        save_jsonl(args.output_path, results)


if __name__ == "__main__":
    main(parse_args())
