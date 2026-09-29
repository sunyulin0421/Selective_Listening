import json
import io
import os
import tempfile
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from mistral_common.audio import Audio
from mistral_common.protocol.instruct.messages import AudioChunk, RawAudio, TextChunk, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from torch import nn
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoModelForCausalLM, AutoProcessor, set_seed

from mechanistic.ablate_qwen_modules import (
    find_modules,
    first_tensor,
    parse_module_groups,
    replace_first_tensor,
)
from mechanistic.common import move_to_device
from mechanistic.inspect_hf_modules import force_attention_backend
from utils.audio import decode_audio
from utils.common import save_jsonl
from utils.datasets import get_dataset


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Transformers Voxtral inference with forward-hook gates. This is "
            "intended for mechanistic pilots; keep vLLM inference_voxtral.py "
            "as the baseline reproduction path."
        )
    )
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--task_split", required=True)
    parser.add_argument("--task_type", required=True)
    parser.add_argument("--model_id", default="mistralai/Voxtral-Mini-3B-2507")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--mitigate_prompt", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--sample_strategy", choices=["first", "random"], default="first")
    parser.add_argument("--sample_seed", type=int, default=None)
    parser.add_argument(
        "--module_groups",
        default="",
        help="Semicolon-separated gates: label=regex;label2=regex2.",
    )
    parser.add_argument("--gate_scale", type=float, default=1.0)
    parser.add_argument(
        "--dynamic_gate_mode",
        choices=[
            "none",
            "answer_margin",
            "answer_margin_soft",
            "clean_audio_disagreement",
            "query_audio_relevance",
        ],
        default="none",
    )
    parser.add_argument(
        "--dynamic_probe_prompt_mode",
        choices=["direct_answer", "generation_prompt"],
        default="direct_answer",
        help=(
            "Prompt used for answer-margin preflight. direct_answer appends "
            "'Answer:' before scoring MCQ letter logits; generation_prompt "
            "scores the next-token distribution under the normal prompt."
        ),
    )
    parser.add_argument(
        "--dynamic_margin_threshold",
        type=float,
        default=1.0,
        help=(
            "Low margin threshold for --dynamic_gate_mode answer_margin or "
            "answer_margin_soft. In answer_margin, samples with top1-top2 "
            "MCQ logit margin below this value use --dynamic_low_scale; "
            "other samples use --dynamic_high_scale. In answer_margin_soft, "
            "this is the margin where the scale starts rising from "
            "--dynamic_low_scale."
        ),
    )
    parser.add_argument(
        "--dynamic_margin_high_threshold",
        type=float,
        default=3.0,
        help=(
            "High margin threshold for --dynamic_gate_mode answer_margin_soft. "
            "Margins at or above this value use --dynamic_high_scale; margins "
            "between low/high thresholds linearly interpolate the scale."
        ),
    )
    parser.add_argument(
        "--dynamic_disagreement_on_missing",
        choices=["low_scale", "high_scale"],
        default="low_scale",
        help=(
            "Fallback for --dynamic_gate_mode clean_audio_disagreement when "
            "either clean or audio preflight cannot produce an MCQ prediction."
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
    parser.add_argument("--max_modules_per_group", type=int, default=16)
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
        default=None,
    )
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--cache_dir",
        default=None,
        help=(
            "Explicit HuggingFace cache root, e.g. "
            "~/.cache/huggingface (or another user-provided cache directory). "
            "When set, the script also exports HF_HOME and "
            "HUGGINGFACE_HUB_CACHE for mistral_common tokenizer lookup."
        ),
    )
    parser.add_argument(
        "--allow_processor_download",
        action="store_true",
        help=(
            "Allow AutoProcessor/tokenizer files to be fetched even when "
            "--local_files_only is used for model weights. Voxtral's "
            "mistral_common tokenizer may live in a separate cache path."
        ),
    )
    parser.add_argument(
        "--loader",
        choices=["auto", "voxtral", "image_text", "causal_lm", "auto_model"],
        default="auto",
        help="Loader order for Transformers. auto tries Voxtral-specific classes first.",
    )
    parser.add_argument(
        "--voxtral_input_mode",
        choices=["processor", "mistral_features"],
        default="mistral_features",
        help=(
            "processor uses AutoProcessor directly. mistral_features builds "
            "prompt tokens with mistral_common and audio input_features with "
            "the Voxtral feature extractor; use this for mechanistic gating."
        ),
    )
    parser.add_argument(
        "--stop_strings",
        nargs="*",
        default=["</s>", "[/INST]", "Human:", "User:", "Assistant:"],
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
    parser.add_argument(
        "--debug_processor_errors",
        action="store_true",
        help="Print processor fallback errors for the first failing sample.",
    )
    parser.add_argument(
        "--debug_processor_batch",
        action="store_true",
        help="Print processor output keys/shapes for each sample. Use only for small diagnostics.",
    )
    parser.add_argument(
        "--debug_gate_stats",
        action="store_true",
        help="Record hook call counts, tensor shapes, and pre/post gate norms in gate_info.",
    )
    parser.add_argument(
        "--debug_print_samples",
        type=int,
        default=5,
        help="Maximum number of samples to print debug summaries for.",
    )
    parser.add_argument(
        "--voxtral_feature_time_alignment",
        choices=["auto", "none"],
        default="auto",
        help=(
            "For --voxtral_input_mode mistral_features, check whether the "
            "number of audio placeholder tokens matches the number of audio "
            "embeddings implied by input_features. auto repeats the feature "
            "time axis only for exact integer-ratio mismatches."
        ),
    )
    parser.add_argument(
        "--voxtral_feature_frames_per_audio_token",
        type=int,
        default=8,
        help=(
            "Voxtral audio tower temporal downsample factor used to estimate "
            "audio embedding count from input_features.shape[-1]."
        ),
    )
    parser.add_argument(
        "--voxtral_audio_token_index",
        type=int,
        default=None,
        help=(
            "Override Voxtral audio placeholder token id. By default the "
            "script reads config.audio_token_index or related config fields."
        ),
    )
    return parser.parse_args()


def configure_hf_cache(cache_dir: Optional[str]) -> Dict[str, str]:
    if not cache_dir:
        return {}
    cache_root = os.path.abspath(cache_dir)
    hub_cache = cache_root if os.path.basename(cache_root) == "hub" else os.path.join(cache_root, "hub")
    os.environ["HF_HOME"] = cache_root if os.path.basename(cache_root) != "hub" else os.path.dirname(cache_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = hub_cache
    os.environ["TRANSFORMERS_CACHE"] = hub_cache
    return {
        "cache_root": os.environ["HF_HOME"],
        "hub_cache": hub_cache,
    }


def hf_from_pretrained_cache_dir(cache_dir: Optional[str]) -> Optional[str]:
    """Return the actual HF hub cache directory used by from_pretrained.

    The project convention passes HF_HOME, e.g. .../hf_cache, while
    huggingface_hub stores model snapshots under HF_HOME/hub. Passing the root
    directory as cache_dir creates a second cache tree at .../hf_cache/models--*
    that mistral_common will not find when local_files_only=True.
    """
    if not cache_dir:
        return None
    cache_root = os.path.abspath(cache_dir)
    return cache_root if os.path.basename(cache_root) == "hub" else os.path.join(cache_root, "hub")


def repo_cache_name(model_id: str) -> str:
    return "models--" + model_id.replace("/", "--")


def resolve_local_snapshot(model_id: str, cache_dir: Optional[str]) -> Optional[str]:
    """Resolve a local HF snapshot directory for offline processor/model loading."""
    if os.path.isdir(model_id):
        return os.path.abspath(model_id)
    hub_cache = hf_from_pretrained_cache_dir(cache_dir)
    if not hub_cache:
        return None
    repo_dir = Path(hub_cache) / repo_cache_name(model_id)
    ref_path = repo_dir / "refs" / "main"
    if not ref_path.exists():
        return None
    revision = ref_path.read_text(encoding="utf-8").strip()
    snapshot = repo_dir / "snapshots" / revision
    return str(snapshot.resolve()) if snapshot.exists() else None


class GateController:
    def __init__(self, scale: float, debug: bool = False):
        self.scale = scale
        self.debug = debug
        self.stats: Dict[str, Dict[str, Any]] = {}
        self.matched_modules: List[str] = []

    def reset_stats(self) -> None:
        self.stats = {}

    def record_call(
        self,
        label: str,
        module_name: str,
        hidden: torch.Tensor,
        scaled: torch.Tensor,
    ) -> None:
        if not self.debug:
            return
        key = f"{label}:{module_name}"
        stat = self.stats.setdefault(
            key,
            {
                "label": label,
                "module_name": module_name,
                "calls": 0,
                "shape": list(hidden.shape),
                "dtype": str(hidden.dtype),
                "device": str(hidden.device),
                "scales": [],
                "pre_norm_first": None,
                "post_norm_first": None,
                "pre_norm_last": None,
                "post_norm_last": None,
            },
        )
        pre_norm = float(hidden.detach().float().norm().item())
        post_norm = float(scaled.detach().float().norm().item())
        stat["calls"] += 1
        if len(stat["scales"]) < 8:
            stat["scales"].append(float(self.scale))
        if stat["pre_norm_first"] is None:
            stat["pre_norm_first"] = pre_norm
            stat["post_norm_first"] = post_norm
        stat["pre_norm_last"] = pre_norm
        stat["post_norm_last"] = post_norm

    def summary(self) -> Dict[str, Any]:
        return {
            "total_hook_calls": sum(stat["calls"] for stat in self.stats.values()),
            "modules_called": len([stat for stat in self.stats.values() if stat["calls"] > 0]),
            "modules": list(self.stats.values()),
        }


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def make_gate_hook(controller: GateController, label: str, module_name: str):
    def hook(_module, _inputs, output):
        hidden = voxtral_first_tensor(output)
        scaled = hidden * controller.scale
        controller.record_call(label, module_name, hidden, scaled)
        return voxtral_replace_first_tensor(output, scaled)

    return hook


def voxtral_first_tensor(output: Any) -> torch.Tensor:
    """Return the primary tensor from tensor/tuple/ModelOutput module outputs."""
    if isinstance(output, torch.Tensor):
        return output
    last_hidden_state = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        return last_hidden_state
    try:
        return first_tensor(output)
    except TypeError:
        pass
    raise TypeError(f"Unsupported output type for Voxtral gate: {type(output)}")


def voxtral_replace_first_tensor(output: Any, replacement: torch.Tensor) -> Any:
    """Replace the primary tensor while preserving Voxtral ModelOutput wrappers."""
    if isinstance(output, torch.Tensor):
        return replacement
    last_hidden_state = getattr(output, "last_hidden_state", None)
    if isinstance(last_hidden_state, torch.Tensor):
        values: Dict[str, Any] = {}
        if hasattr(output, "items"):
            values.update(dict(output.items()))
        values["last_hidden_state"] = replacement
        for attr in ["hidden_states", "attentions"]:
            if attr not in values and hasattr(output, attr):
                value = getattr(output, attr)
                if value is not None:
                    values[attr] = value
        return output.__class__(**values)
    return replace_first_tensor(output, replacement)


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
        matches = [
            (name, module)
            for name, module in model.named_modules()
            if name == suffix or name.endswith("." + suffix)
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


def probe_prompt(prompt: str, mode: str) -> str:
    if mode == "direct_answer":
        return prompt.rstrip() + "\nAnswer:"
    if mode == "generation_prompt":
        return prompt
    raise ValueError(f"Unsupported dynamic probe prompt mode: {mode}")


def encode_token_candidates(tokenizer: Any, text: str) -> List[int]:
    attempts = [
        lambda: tokenizer.encode(text, add_special_tokens=False),
        lambda: tokenizer.encode(text),
        lambda: tokenizer(text, add_special_tokens=False),
        lambda: tokenizer(text),
    ]
    for attempt in attempts:
        try:
            encoded = attempt()
        except Exception:
            continue
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.detach().cpu().view(-1).tolist()
        elif hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        elif isinstance(encoded, dict) and "input_ids" in encoded:
            encoded = encoded["input_ids"]
        if isinstance(encoded, torch.Tensor):
            encoded = encoded.detach().cpu().view(-1).tolist()
        if isinstance(encoded, (list, tuple)):
            if encoded and isinstance(encoded[0], (list, tuple)):
                encoded = encoded[0]
            token_ids = [int(token) for token in encoded if isinstance(token, int)]
            if token_ids:
                return token_ids
    return []


def letter_token_ids(processor: AutoProcessor, letters: List[str]) -> Dict[str, List[int]]:
    tokenizer = getattr(processor, "tokenizer", processor)
    ids: Dict[str, List[int]] = {}
    for letter in letters:
        token_ids: List[int] = []
        for variant in [letter, " " + letter, letter + ")", " " + letter + ")"]:
            encoded = encode_token_candidates(tokenizer, variant)
            if encoded:
                token_ids.append(encoded[-1])
        ids[letter] = sorted(set(token_ids))
    return ids


def build_voxtral_batch(
    processor: AutoProcessor,
    voxtral_tokenizer: Optional[MistralTokenizer],
    query: str,
    audio_array: Optional[np.ndarray],
    sampling_rate: Optional[int],
    args: Namespace,
    model: nn.Module,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    if args.voxtral_input_mode == "mistral_features":
        if voxtral_tokenizer is None:
            raise ValueError("--voxtral_input_mode mistral_features requires MistralTokenizer")
        batch = processor_call_voxtral_mistral_features(
            processor=processor,
            tokenizer=voxtral_tokenizer,
            query=query,
            audio_array=audio_array,
            sampling_rate=sampling_rate,
            model_id=args.model_id,
            fallback_sampling_rate=args.sampling_rate,
            debug_errors=args.debug_processor_errors,
        )
    else:
        batch = processor_call_voxtral(
            processor=processor,
            query=query,
            audio_array=audio_array,
            sampling_rate=sampling_rate,
            debug_errors=args.debug_processor_errors,
        )
    alignment_info = align_voxtral_mistral_features(batch, model, args)
    return batch, alignment_info


def compute_answer_margin(
    model: nn.Module,
    processor: AutoProcessor,
    voxtral_tokenizer: Optional[MistralTokenizer],
    prompt: str,
    audio_array: Optional[np.ndarray],
    sampling_rate: Optional[int],
    args: Namespace,
) -> Dict[str, Any]:
    letters = ["A", "B", "C", "D", "E"]
    batch, alignment_info = build_voxtral_batch(
        processor=processor,
        voxtral_tokenizer=voxtral_tokenizer,
        query=probe_prompt(prompt, args.dynamic_probe_prompt_mode),
        audio_array=audio_array,
        sampling_rate=sampling_rate,
        args=args,
        model=model,
    )
    device = next(model.parameters()).device
    batch = move_to_device(batch, device)
    token_ids = letter_token_ids(processor, letters)

    with torch.inference_mode():
        outputs = model(**batch, use_cache=False)
    logits = outputs.logits[0, -1].float()

    scores: Dict[str, float] = {}
    for letter, ids in token_ids.items():
        if not ids:
            continue
        scores[letter] = float(torch.max(logits[ids]).item())
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    result: Dict[str, Any] = {
        "dynamic_probe_prompt_mode": args.dynamic_probe_prompt_mode,
        "direct_pred": ranked[0][0] if ranked else None,
        "answer_scores": scores,
        "answer_margin": None,
    }
    if alignment_info:
        result["preflight_voxtral_feature_alignment"] = alignment_info
    if len(ranked) >= 2:
        result["answer_margin"] = ranked[0][1] - ranked[1][1]
    return result


def choose_gate_scale(args: Namespace, relevance: Dict[str, Any]) -> float:
    if args.dynamic_gate_mode == "none":
        return args.gate_scale
    if args.dynamic_gate_mode == "answer_margin":
        margin = relevance.get("answer_margin")
        if margin is None or margin < args.dynamic_margin_threshold:
            return args.dynamic_low_scale
        return args.dynamic_high_scale
    if args.dynamic_gate_mode == "answer_margin_soft":
        margin = relevance.get("answer_margin")
        if margin is None:
            return args.dynamic_high_scale
        low_t = args.dynamic_margin_threshold
        high_t = args.dynamic_margin_high_threshold
        if high_t <= low_t:
            raise ValueError(
                "--dynamic_margin_high_threshold must be greater than "
                "--dynamic_margin_threshold for answer_margin_soft"
            )
        if margin <= low_t:
            return args.dynamic_low_scale
        if margin >= high_t:
            return args.dynamic_high_scale
        ratio = (margin - low_t) / (high_t - low_t)
        return args.dynamic_low_scale + ratio * (
            args.dynamic_high_scale - args.dynamic_low_scale
        )
    if args.dynamic_gate_mode == "clean_audio_disagreement":
        if relevance.get("missing_disagreement_prediction"):
            return (
                args.dynamic_low_scale
                if args.dynamic_disagreement_on_missing == "low_scale"
                else args.dynamic_high_scale
            )
        return (
            args.dynamic_low_scale
            if relevance.get("clean_audio_disagree")
            else args.dynamic_high_scale
        )
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
        modules, fallback = find_modules_for_gate(model, regex, max_modules_per_group)
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
            handles.append(module.register_forward_hook(make_gate_hook(controller, label, name)))
    controller.matched_modules = matched_module_names
    return handles


def load_voxtral_model(args: Namespace) -> nn.Module:
    pretrained_id = resolve_local_snapshot(args.model_id, args.cache_dir) if args.local_files_only else None
    pretrained_id = pretrained_id or args.model_id

    config_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.cache_dir:
        config_kwargs["cache_dir"] = hf_from_pretrained_cache_dir(args.cache_dir)
    config = AutoConfig.from_pretrained(pretrained_id, **config_kwargs)
    if args.attn_implementation:
        force_attention_backend(config, args.attn_implementation)

    kwargs: Dict[str, Any] = {
        "config": config,
        "torch_dtype": dtype_from_name(args.dtype),
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.cache_dir:
        kwargs["cache_dir"] = hf_from_pretrained_cache_dir(args.cache_dir)
    if args.device_map:
        kwargs["device_map"] = args.device_map
    if args.attn_implementation:
        kwargs["attn_implementation"] = args.attn_implementation

    import transformers

    loader_candidates: List[Tuple[str, Any]] = []
    voxtral_cls = getattr(transformers, "VoxtralForConditionalGeneration", None)
    if args.loader in {"auto", "voxtral"} and voxtral_cls is not None:
        loader_candidates.append(("VoxtralForConditionalGeneration", voxtral_cls))
    image_text_cls = getattr(transformers, "AutoModelForImageTextToText", None)
    if args.loader in {"auto", "image_text"} and image_text_cls is not None:
        loader_candidates.append(("AutoModelForImageTextToText", image_text_cls))
    if args.loader in {"auto", "causal_lm"}:
        loader_candidates.append(("AutoModelForCausalLM", AutoModelForCausalLM))
    if args.loader in {"auto", "auto_model"}:
        loader_candidates.append(("AutoModel", AutoModel))

    errors = []
    for label, loader in loader_candidates:
        try:
            model = loader.from_pretrained(pretrained_id, **kwargs)
            print(f"Loaded Voxtral model with {label}")
            model.eval()
            return model
        except Exception as exc:
            errors.append(f"{label}: {type(exc).__name__}: {exc}")
    raise RuntimeError("All Voxtral HF loaders failed:\n" + "\n".join(errors))


def get_prompt_and_audio(
    sample: Dict[str, Any],
    task_type: str,
    sampling_rate: int,
) -> Tuple[str, Optional[np.ndarray], Optional[int]]:
    query = sample["query"]
    if task_type == "text_bench":
        return query, None, None
    if task_type not in {"text_bench_interference", "audio_understanding", "asr"}:
        raise ValueError(f"Unsupported task_type for prompt/audio construction: {task_type}")
    audio_array, audio_sr = decode_audio(sample["audio"])
    audio_array = librosa.resample(
        audio_array,
        orig_sr=audio_sr,
        target_sr=sampling_rate,
    )
    return query, audio_array, sampling_rate


def build_mistral_common_prompt(
    query: str,
    audio_array: Optional[np.ndarray],
    sampling_rate: Optional[int],
    tokenizer: MistralTokenizer,
    model_id: str,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """Build Voxtral prompt tokens with the same mistral_common path as vLLM.

    Voxtral's HF processor currently does not accept Qwen/Phi-style
    ``audios=...`` inputs. The reliable path is to use mistral_common for text
    and audio placeholders, then provide mel features through ``input_features``.
    """
    if audio_array is None:
        messages = [UserMessage(content=[TextChunk(text=query)])]
    else:
        assert sampling_rate is not None
        buffer = io.BytesIO()
        sf.write(buffer, audio_array, sampling_rate, format="WAV")
        audio = Audio.from_bytes(buffer.getvalue())
        messages = [
            UserMessage(
                content=[
                    AudioChunk(input_audio=RawAudio.from_audio(audio)),
                    TextChunk(text=query),
                ]
            )
        ]
    request = ChatCompletionRequest(messages=messages, model=model_id)
    encoded = tokenizer.encode_chat_completion(request)
    token_audios = encoded.audios or []
    audio_payloads = [
        {
            "audio_array": getattr(audio, "audio_array", None),
            "sampling_rate": getattr(audio, "sampling_rate", None),
        }
        for audio in token_audios
    ]
    return encoded.tokens, audio_payloads


def processor_component(processor: AutoProcessor, names: Tuple[str, ...]) -> Optional[Any]:
    for name in names:
        component = getattr(processor, name, None)
        if component is not None:
            return component
    return None


def normalize_feature_output(output: Any) -> Dict[str, torch.Tensor]:
    if hasattr(output, "data") and isinstance(output.data, dict):
        output = output.data
    if not isinstance(output, dict):
        output_dict = {
            key: getattr(output, key)
            for key in ["input_features", "attention_mask", "feature_attention_mask"]
            if hasattr(output, key)
        }
    else:
        output_dict = output

    normalized: Dict[str, torch.Tensor] = {}
    for key, value in output_dict.items():
        if isinstance(value, torch.Tensor):
            normalized[key] = value
        elif isinstance(value, np.ndarray):
            normalized[key] = torch.from_numpy(value)
        elif isinstance(value, (list, tuple)):
            try:
                normalized[key] = torch.tensor(value)
            except Exception:
                continue
    return normalized


def extract_voxtral_input_features(
    processor: AutoProcessor,
    audio_payloads: List[Dict[str, Any]],
    fallback_sampling_rate: int,
    debug_errors: bool,
) -> Dict[str, torch.Tensor]:
    if not audio_payloads:
        return {}

    arrays = []
    sampling_rates = []
    for payload in audio_payloads:
        audio_array = payload.get("audio_array")
        if audio_array is None:
            continue
        arrays.append(np.asarray(audio_array, dtype=np.float32))
        sampling_rates.append(int(payload.get("sampling_rate") or fallback_sampling_rate))
    if not arrays:
        raise ValueError("mistral_common produced audio placeholders but no audio arrays")

    if len(set(sampling_rates)) != 1:
        target_sr = sampling_rates[0]
        arrays = [
            librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            if sr != target_sr else audio
            for audio, sr in zip(arrays, sampling_rates)
        ]
    else:
        target_sr = sampling_rates[0]

    extractor = processor_component(
        processor,
        ("feature_extractor", "audio_processor", "audio_feature_extractor"),
    )
    if extractor is None:
        raise AttributeError(
            "Voxtral processor has no feature_extractor/audio_processor component"
        )

    attempts = [
        lambda: extractor(
            arrays,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        ),
        lambda: extractor(
            raw_speech=arrays,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        ),
        lambda: extractor(
            audio=arrays,
            sampling_rate=target_sr,
            return_tensors="pt",
            padding=True,
        ),
    ]
    errors = []
    for attempt in attempts:
        try:
            features = normalize_feature_output(attempt())
            if "input_features" in features:
                return {"input_features": features["input_features"].float()}
            errors.append(f"missing input_features in keys={sorted(features.keys())}")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    if debug_errors:
        print("Voxtral feature extraction errors:\n" + "\n".join(errors))
    raise RuntimeError("All Voxtral feature extraction attempts failed:\n" + "\n".join(errors))


def processor_call_voxtral_mistral_features(
    processor: AutoProcessor,
    tokenizer: MistralTokenizer,
    query: str,
    audio_array: Optional[np.ndarray],
    sampling_rate: Optional[int],
    model_id: str,
    fallback_sampling_rate: int,
    debug_errors: bool,
) -> Dict[str, torch.Tensor]:
    prompt_tokens, audio_payloads = build_mistral_common_prompt(
        query=query,
        audio_array=audio_array,
        sampling_rate=sampling_rate,
        tokenizer=tokenizer,
        model_id=model_id,
    )
    batch: Dict[str, torch.Tensor] = {
        "input_ids": torch.tensor([prompt_tokens], dtype=torch.long),
    }
    batch["attention_mask"] = torch.ones_like(batch["input_ids"])
    batch.update(
        extract_voxtral_input_features(
            processor=processor,
            audio_payloads=audio_payloads,
            fallback_sampling_rate=fallback_sampling_rate,
            debug_errors=debug_errors,
        )
    )
    return batch


def get_nested_attr(root: Any, path: str) -> Optional[Any]:
    value = root
    for part in path.split("."):
        value = getattr(value, part, None)
        if value is None:
            return None
    return value


def infer_voxtral_audio_token_index(model: nn.Module, override: Optional[int]) -> Optional[int]:
    if override is not None:
        return int(override)
    config = getattr(model, "config", None)
    for path in [
        "audio_token_index",
        "audio_token_id",
        "audio_token",
        "text_config.audio_token_index",
        "text_config.audio_token_id",
    ]:
        value = get_nested_attr(config, path)
        if isinstance(value, int):
            return int(value)
    return None


def infer_repeated_audio_token_from_input_ids(
    input_ids: torch.Tensor,
    min_run_length: int,
) -> Tuple[Optional[int], int]:
    flat_ids = input_ids.detach().cpu().view(-1).tolist()
    best_token = None
    best_run = 0
    current_token = None
    current_run = 0
    for token in flat_ids:
        if token == current_token:
            current_run += 1
        else:
            if current_run > best_run:
                best_token = current_token
                best_run = current_run
            current_token = token
            current_run = 1
    if current_run > best_run:
        best_token = current_token
        best_run = current_run
    if best_token is not None and best_run >= min_run_length:
        return int(best_token), int(best_run)
    return None, int(best_run)


def align_voxtral_mistral_features(
    batch: Dict[str, torch.Tensor],
    model: nn.Module,
    args: Namespace,
) -> Dict[str, Any]:
    """Repair exact audio-token/input-feature length mismatches.

    Voxtral replaces audio placeholder token positions with audio tower
    embeddings during forward. For some FSD clips, mistral_common can insert
    twice as many audio placeholders as the fixed 3000-frame HF feature path
    produces embeddings for. Keep the mel features unchanged and trim only the
    surplus placeholder tokens so the assignment in Voxtral forward is aligned.
    """
    info: Dict[str, Any] = {
        "voxtral_feature_time_alignment": args.voxtral_feature_time_alignment,
    }
    if args.voxtral_input_mode != "mistral_features":
        return info
    if args.voxtral_feature_time_alignment == "none":
        return info
    input_ids = batch.get("input_ids")
    input_features = batch.get("input_features")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(input_features, torch.Tensor):
        return info

    frames_per_token = int(args.voxtral_feature_frames_per_audio_token)
    feature_time = int(input_features.shape[-1])
    feature_audio_token_estimate = feature_time // frames_per_token if frames_per_token > 0 else 0

    audio_token_index = infer_voxtral_audio_token_index(
        model,
        args.voxtral_audio_token_index,
    )
    inferred_from_repeated_run = False
    longest_repeated_run = None
    if audio_token_index is None and feature_audio_token_estimate > 0:
        audio_token_index, longest_repeated_run = infer_repeated_audio_token_from_input_ids(
            input_ids,
            min_run_length=max(16, feature_audio_token_estimate // 2),
        )
        inferred_from_repeated_run = audio_token_index is not None
    info["audio_token_index"] = audio_token_index
    info["audio_token_index_inferred_from_repeated_run"] = inferred_from_repeated_run
    if longest_repeated_run is not None:
        info["longest_repeated_token_run"] = longest_repeated_run
    if audio_token_index is None:
        info["alignment_status"] = "skipped_missing_audio_token_index"
        return info

    audio_token_mask = input_ids == audio_token_index
    audio_token_count = int(audio_token_mask.sum().item())
    if audio_token_count == 0 and feature_audio_token_estimate > 0:
        repeated_token, repeated_run = infer_repeated_audio_token_from_input_ids(
            input_ids,
            min_run_length=max(16, feature_audio_token_estimate // 2),
        )
        if repeated_token is not None:
            audio_token_index = repeated_token
            inferred_from_repeated_run = True
            longest_repeated_run = repeated_run
            audio_token_mask = input_ids == audio_token_index
            audio_token_count = int(audio_token_mask.sum().item())
            info["audio_token_index"] = audio_token_index
            info["audio_token_index_inferred_from_repeated_run"] = True
            info["longest_repeated_token_run"] = repeated_run
    info.update(
        {
            "audio_token_count": audio_token_count,
            "input_feature_time": feature_time,
            "feature_frames_per_audio_token": frames_per_token,
            "feature_audio_token_estimate": feature_audio_token_estimate,
        }
    )
    if audio_token_count <= 0 or feature_audio_token_estimate <= 0:
        info["alignment_status"] = "skipped_no_audio_tokens_or_features"
        return info
    if audio_token_count == feature_audio_token_estimate:
        info["alignment_status"] = "already_aligned"
        return info

    if audio_token_count > feature_audio_token_estimate:
        if input_ids.ndim != 2 or int(input_ids.shape[0]) != 1:
            info["alignment_status"] = "unsupported_batch_audio_token_trim"
            return info
        audio_positions = torch.nonzero(audio_token_mask[0], as_tuple=False).flatten()
        if int(audio_positions.numel()) <= feature_audio_token_estimate:
            info["alignment_status"] = "skipped_inconsistent_audio_token_positions"
            return info
        keep = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
        keep[audio_positions[feature_audio_token_estimate:]] = False
        old_input_length = int(input_ids.shape[1])
        batch["input_ids"] = input_ids[:, keep]
        attention_mask = batch.get("attention_mask")
        if isinstance(attention_mask, torch.Tensor) and attention_mask.ndim == 2:
            if int(attention_mask.shape[1]) == old_input_length:
                batch["attention_mask"] = attention_mask[:, keep]
            else:
                info["attention_mask_alignment_warning"] = {
                    "attention_mask_length": int(attention_mask.shape[1]),
                    "old_input_length": old_input_length,
                }
        info.update(
            {
                "alignment_status": "trimmed_audio_placeholder_tokens",
                "trimmed_audio_tokens": int(audio_token_count - feature_audio_token_estimate),
                "old_input_token_length": old_input_length,
                "aligned_input_token_length": int(batch["input_ids"].shape[1]),
                "aligned_audio_token_count": int(
                    (batch["input_ids"] == audio_token_index).sum().item()
                ),
                "input_features_unchanged": True,
            }
        )
        return info

    info["alignment_status"] = "unsupported_feature_token_estimate_exceeds_placeholders"
    return info


def write_temp_wav(audio_array: np.ndarray, sampling_rate: int) -> str:
    handle = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    handle.close()
    sf.write(handle.name, audio_array, sampling_rate)
    return handle.name


def processor_call_voxtral(
    processor: AutoProcessor,
    query: str,
    audio_array: Optional[np.ndarray],
    sampling_rate: Optional[int],
    debug_errors: bool,
) -> Dict[str, torch.Tensor]:
    attempts = []
    temp_paths: List[str] = []
    audio_markers = ("audio", "input_features", "feature", "waveform", "spectrogram")

    def has_audio_like_key(batch: Dict[str, Any]) -> bool:
        return any(
            any(marker in key.lower() for marker in audio_markers)
            for key in batch.keys()
        )

    def normalize_processor_output(output: Any) -> Dict[str, torch.Tensor]:
        if isinstance(output, dict):
            normalized = {}
            for key, value in output.items():
                if isinstance(value, torch.Tensor):
                    normalized[key] = value
                elif isinstance(value, (list, tuple)) and value and all(isinstance(item, int) for item in value):
                    normalized[key] = torch.tensor([list(value)], dtype=torch.long)
                else:
                    normalized[key] = value
            return normalized
        if isinstance(output, torch.Tensor):
            tensor = output.unsqueeze(0) if output.ndim == 1 else output
            return {"input_ids": tensor.long()}
        if isinstance(output, (list, tuple)) and output and all(isinstance(item, int) for item in output):
            return {"input_ids": torch.tensor([list(output)], dtype=torch.long)}
        input_ids = getattr(output, "input_ids", None)
        if input_ids is not None:
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            if input_ids.ndim == 1:
                input_ids = input_ids.unsqueeze(0)
            return {"input_ids": input_ids.long()}
        raise TypeError(f"Unsupported Voxtral processor output type: {type(output).__name__}")

    def checked_output(output: Any) -> Dict[str, torch.Tensor]:
        batch = normalize_processor_output(output)
        if audio_array is not None and not has_audio_like_key(batch):
            raise ValueError(
                f"Processor output has no audio-like keys: {sorted(batch.keys())}"
            )
        return batch

    if audio_array is None:
        conversation = [{"role": "user", "content": [{"type": "text", "text": query}]}]
        attempts.extend(
            [
                lambda: processor.apply_chat_template(
                    conversation,
                    return_tensors="pt",
                    return_dict=True,
                    tokenize=True,
                ),
                lambda: processor.apply_chat_template(
                    conversation,
                ),
                lambda: processor(text=query, return_tensors="pt"),
                lambda: processor(query, return_tensors="pt"),
            ]
        )
    else:
        assert sampling_rate is not None
        wav_path = write_temp_wav(audio_array, sampling_rate)
        temp_paths.append(wav_path)
        conversations = [
            [{"role": "user", "content": [{"type": "audio", "url": wav_path}, {"type": "text", "text": query}]}],
            [{"role": "user", "content": [{"type": "audio", "url": "file://" + wav_path}, {"type": "text", "text": query}]}],
            [{"role": "user", "content": [{"type": "audio", "path": wav_path}, {"type": "text", "text": query}]}],
            [{"role": "user", "content": [{"type": "input_audio", "path": wav_path}, {"type": "text", "text": query}]}],
            [{"role": "user", "content": [{"type": "audio", "audio": wav_path}, {"type": "text", "text": query}]}],
        ]
        for conversation in conversations:
            attempts.append(
                lambda conversation=conversation: processor.apply_chat_template(
                    conversation,
                    return_tensors="pt",
                    return_dict=True,
                    tokenize=True,
                    processor_kwargs={
                        "audio_kwargs": {
                            "sampling_rate": sampling_rate,
                            "padding": True,
                            "truncation": False,
                        },
                        "text_kwargs": {
                            "padding": True,
                        },
                    },
                )
            )
        for conversation in conversations:
            attempts.append(
                lambda conversation=conversation: processor.apply_chat_template(
                    conversation,
                )
            )
        attempts.extend(
            [
                lambda: processor(
                    text=query,
                    audios=[(audio_array, sampling_rate)],
                    return_tensors="pt",
                ),
                lambda: processor(
                    text=query,
                    audio=audio_array,
                    sampling_rate=sampling_rate,
                    return_tensors="pt",
                ),
                lambda: processor(
                    query,
                    audios=[(audio_array, sampling_rate)],
                    return_tensors="pt",
                ),
            ]
        )

    errors = []
    try:
        for attempt in attempts:
            try:
                return checked_output(attempt())
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        if debug_errors:
            print("Voxtral processor fallback errors:\n" + "\n".join(errors))
        raise RuntimeError("All Voxtral processor calls failed:\n" + "\n".join(errors))
    finally:
        for path in temp_paths:
            try:
                os.remove(path)
            except OSError:
                pass


def summarize_processor_batch(batch: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "keys": sorted(batch.keys()),
        "tensor_shapes": {},
        "non_tensor_types": {},
        "has_audio_like_key": False,
    }
    audio_markers = ("audio", "input_features", "feature", "waveform", "spectrogram")
    for key, value in batch.items():
        lowered_key = key.lower()
        if any(marker in lowered_key for marker in audio_markers):
            summary["has_audio_like_key"] = True
        if isinstance(value, torch.Tensor):
            summary["tensor_shapes"][key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
        else:
            summary["non_tensor_types"][key] = type(value).__name__
    return summary


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
    voxtral_tokenizer: Optional[MistralTokenizer],
    sample: Dict[str, Any],
    args: Namespace,
    controller: GateController,
    row_idx: int,
) -> Tuple[str, Dict[str, Any]]:
    query, audio_array, sampling_rate = get_prompt_and_audio(
        sample,
        args.task_type,
        args.sampling_rate,
    )

    gate_info: Dict[str, Any] = {
        "dynamic_gate_mode": args.dynamic_gate_mode,
        "gate_scale": args.gate_scale,
        "applied_gate_scale": controller.scale,
        "module_groups": args.module_groups,
    }
    if args.dynamic_gate_mode in {"answer_margin", "answer_margin_soft"} and args.task_type != "text_bench":
        controller.scale = 1.0
        preflight = compute_answer_margin(
            model=model,
            processor=processor,
            voxtral_tokenizer=voxtral_tokenizer,
            prompt=query,
            audio_array=audio_array,
            sampling_rate=sampling_rate,
            args=args,
        )
        controller.scale = choose_gate_scale(args, preflight)
        answer_margin = preflight.get("answer_margin")
        if args.dynamic_gate_mode == "answer_margin_soft":
            if answer_margin is None:
                gate_decision = "missing_high_scale"
            elif answer_margin <= args.dynamic_margin_threshold:
                gate_decision = "low_scale"
            elif answer_margin >= args.dynamic_margin_high_threshold:
                gate_decision = "high_scale"
            else:
                gate_decision = "interpolated_scale"
        else:
            gate_decision = (
                "low_scale"
                if answer_margin is None or answer_margin < args.dynamic_margin_threshold
                else "high_scale"
            )
        gate_info = {
            "dynamic_gate_mode": args.dynamic_gate_mode,
            "dynamic_probe_prompt_mode": args.dynamic_probe_prompt_mode,
            "dynamic_margin_threshold": args.dynamic_margin_threshold,
            "dynamic_margin_high_threshold": (
                args.dynamic_margin_high_threshold
                if args.dynamic_gate_mode == "answer_margin_soft"
                else None
            ),
            "dynamic_low_scale": args.dynamic_low_scale,
            "dynamic_high_scale": args.dynamic_high_scale,
            "applied_gate_scale": controller.scale,
            "gate_decision": gate_decision,
            "module_groups": args.module_groups,
            **preflight,
        }
    elif args.dynamic_gate_mode == "clean_audio_disagreement" and args.task_type != "text_bench":
        controller.scale = 1.0
        clean_preflight = compute_answer_margin(
            model=model,
            processor=processor,
            voxtral_tokenizer=voxtral_tokenizer,
            prompt=query,
            audio_array=None,
            sampling_rate=None,
            args=args,
        )
        audio_preflight = compute_answer_margin(
            model=model,
            processor=processor,
            voxtral_tokenizer=voxtral_tokenizer,
            prompt=query,
            audio_array=audio_array,
            sampling_rate=sampling_rate,
            args=args,
        )
        clean_pred = clean_preflight.get("direct_pred")
        audio_pred = audio_preflight.get("direct_pred")
        missing_prediction = clean_pred is None or audio_pred is None
        disagreement = (
            False
            if missing_prediction
            else clean_pred != audio_pred
        )
        relevance = {
            "clean_direct_pred": clean_pred,
            "audio_direct_pred": audio_pred,
            "clean_answer_margin": clean_preflight.get("answer_margin"),
            "audio_answer_margin": audio_preflight.get("answer_margin"),
            "clean_answer_scores": clean_preflight.get("answer_scores"),
            "audio_answer_scores": audio_preflight.get("answer_scores"),
            "clean_audio_disagree": disagreement,
            "missing_disagreement_prediction": missing_prediction,
        }
        if "preflight_voxtral_feature_alignment" in clean_preflight:
            relevance["clean_preflight_voxtral_feature_alignment"] = clean_preflight[
                "preflight_voxtral_feature_alignment"
            ]
        if "preflight_voxtral_feature_alignment" in audio_preflight:
            relevance["audio_preflight_voxtral_feature_alignment"] = audio_preflight[
                "preflight_voxtral_feature_alignment"
            ]
        controller.scale = choose_gate_scale(args, relevance)
        low_scale_decision = (
            missing_prediction and args.dynamic_disagreement_on_missing == "low_scale"
        ) or (not missing_prediction and disagreement)
        gate_info = {
            "dynamic_gate_mode": args.dynamic_gate_mode,
            "dynamic_probe_prompt_mode": args.dynamic_probe_prompt_mode,
            "dynamic_low_scale": args.dynamic_low_scale,
            "dynamic_high_scale": args.dynamic_high_scale,
            "dynamic_disagreement_on_missing": args.dynamic_disagreement_on_missing,
            "applied_gate_scale": controller.scale,
            "gate_decision": "low_scale" if low_scale_decision else "high_scale",
            "module_groups": args.module_groups,
            **relevance,
        }
    elif args.dynamic_gate_mode == "query_audio_relevance" and args.task_type != "text_bench":
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
            **relevance,
        }
    else:
        controller.scale = args.gate_scale
        gate_info["applied_gate_scale"] = controller.scale

    gate_info["matched_modules"] = list(controller.matched_modules)

    batch, alignment_info = build_voxtral_batch(
        processor=processor,
        voxtral_tokenizer=voxtral_tokenizer,
        query=query,
        audio_array=audio_array,
        sampling_rate=sampling_rate,
        args=args,
        model=model,
    )
    if alignment_info:
        gate_info["voxtral_feature_alignment"] = alignment_info
    processor_batch_summary = summarize_processor_batch(batch)
    gate_info["voxtral_input_mode"] = args.voxtral_input_mode
    gate_info["has_audio_input_features"] = "input_features" in batch
    if isinstance(batch.get("input_features"), torch.Tensor):
        gate_info["input_features_shape"] = list(batch["input_features"].shape)
    if args.debug_processor_batch and row_idx < args.debug_print_samples:
        print(
            f"Processor batch row={row_idx}: "
            f"{json.dumps(processor_batch_summary, ensure_ascii=False)}"
        )
    device = next(model.parameters()).device
    batch = move_to_device(batch, device)

    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
    }
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    controller.reset_stats()
    with torch.inference_mode():
        generated_ids = model.generate(**batch, **generation_kwargs)
    generated_new_tokens = int(generated_ids.shape[-1] - batch["input_ids"].shape[-1])
    if args.debug_processor_batch:
        gate_info["processor_batch_summary"] = processor_batch_summary
    if args.debug_gate_stats:
        gate_debug_stats = controller.summary()
        gate_info["gate_debug_stats"] = gate_debug_stats
        if row_idx < args.debug_print_samples:
            print(
                f"Gate stats row={row_idx}: "
                f"{json.dumps(gate_debug_stats, ensure_ascii=False)}"
            )
    response = decode_generation(
        processor,
        batch["input_ids"],
        generated_ids,
        args.stop_strings,
        args.truncate_response_mode,
    )
    gate_info["generated_new_tokens"] = generated_new_tokens
    gate_info["response_chars"] = len(response)
    return response, gate_info


def json_safe_metadata(sample: Dict[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
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
        elif isinstance(value, list) and all(
            isinstance(item, (str, int, float, bool)) or item is None
            for item in value
        ):
            metadata[key] = value
        elif isinstance(value, dict) and all(
            isinstance(k, str)
            and (isinstance(v, (str, int, float, bool)) or v is None)
            for k, v in value.items()
        ):
            metadata[key] = value
    return metadata


def generate_responses(args: Namespace) -> List[Dict[str, Any]]:
    if args.dynamic_gate_mode != "none" and not args.module_groups.strip():
        raise ValueError(
            "Dynamic gating requires --module_groups so the selected scale is "
            "applied to a real model module."
        )
    if args.module_groups.strip() and args.voxtral_input_mode != "mistral_features":
        raise ValueError(
            "Voxtral mechanistic gating requires --voxtral_input_mode "
            "mistral_features so real audio input_features reach the audio tower."
        )
    if args.dynamic_gate_mode in {
        "answer_margin",
        "answer_margin_soft",
        "clean_audio_disagreement",
    }:
        task_name = f"{args.task_id} {args.task_split}".lower()
        if "arc" not in task_name and "mmlu" not in task_name:
            raise ValueError(
                f"--dynamic_gate_mode {args.dynamic_gate_mode} currently supports only MCQ "
                "tasks with A/B/C/D/E answer labels, e.g. ARC or MMLU."
            )
        if args.task_type == "text_bench":
            raise ValueError(
                f"--dynamic_gate_mode {args.dynamic_gate_mode} is intended for "
                "text_bench_interference samples with audio. Use "
                "--dynamic_gate_mode none for clean text-only runs."
            )
    if args.dynamic_gate_mode == "query_audio_relevance" and args.task_type == "text_bench":
        raise ValueError(
            "--dynamic_gate_mode query_audio_relevance is intended for "
            "text_bench_interference samples with audio."
        )

    cache_info = configure_hf_cache(args.cache_dir)
    if cache_info:
        print(f"Using HF cache root: {cache_info['cache_root']}")
        print(f"Using HF hub cache: {cache_info['hub_cache']}")

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

    run_config = {
        "backend": "transformers",
        "model_id": args.model_id,
        "task_id": args.task_id,
        "task_split": args.task_split,
        "task_type": args.task_type,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "sample_strategy": args.sample_strategy,
        "sample_seed": args.sample_seed,
        "voxtral_input_mode": args.voxtral_input_mode,
        "truncate_response_mode": args.truncate_response_mode,
        "stop_strings": args.stop_strings,
        "module_groups": args.module_groups,
        "gate_scale": args.gate_scale,
        "dynamic_gate_mode": args.dynamic_gate_mode,
        "dynamic_probe_prompt_mode": args.dynamic_probe_prompt_mode,
        "dynamic_margin_threshold": args.dynamic_margin_threshold,
        "dynamic_margin_high_threshold": args.dynamic_margin_high_threshold,
        "dynamic_low_scale": args.dynamic_low_scale,
        "dynamic_high_scale": args.dynamic_high_scale,
        "dynamic_disagreement_on_missing": args.dynamic_disagreement_on_missing,
        "query_relevance_scope": args.query_relevance_scope,
        "query_relevance_keywords": args.query_relevance_keywords,
        "voxtral_feature_time_alignment": args.voxtral_feature_time_alignment,
        "voxtral_feature_frames_per_audio_token": args.voxtral_feature_frames_per_audio_token,
    }

    processor_pretrained_id = (
        resolve_local_snapshot(args.model_id, args.cache_dir)
        if args.local_files_only and not args.allow_processor_download
        else None
    )
    processor_pretrained_id = processor_pretrained_id or args.model_id
    if processor_pretrained_id != args.model_id:
        print(f"Loading Voxtral processor from local snapshot: {processor_pretrained_id}")

    processor_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "local_files_only": args.local_files_only and not args.allow_processor_download,
    }
    if args.cache_dir:
        processor_kwargs["cache_dir"] = hf_from_pretrained_cache_dir(args.cache_dir)
    processor = AutoProcessor.from_pretrained(processor_pretrained_id, **processor_kwargs)
    voxtral_tokenizer: Optional[MistralTokenizer] = None
    if args.voxtral_input_mode == "mistral_features":
        voxtral_tokenizer = MistralTokenizer.from_hf_hub(
            args.model_id,
            local_files_only=args.local_files_only and not args.allow_processor_download,
        )
    model = load_voxtral_model(args)
    initial_scale = 1.0 if args.dynamic_gate_mode != "none" else args.gate_scale
    controller = GateController(initial_scale, debug=args.debug_gate_stats)
    handles = install_gates(
        model=model,
        module_groups=args.module_groups,
        controller=controller,
        max_modules_per_group=args.max_modules_per_group,
    )
    if args.module_groups.strip() and not handles:
        raise RuntimeError("Voxtral gating installed no forward hooks.")
    run_config["matched_modules"] = list(controller.matched_modules)

    results = []
    stream_handle = None
    if args.stream_output:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        stream_handle = open(args.output_path, "w", encoding="utf-8")
    try:
        for row_idx, sample in enumerate(tqdm(dataset, desc="Voxtral gated inference")):
            response, gate_info = generate_one(
                model,
                processor,
                voxtral_tokenizer,
                sample,
                args,
                controller,
                row_idx,
            )
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
                elif isinstance(metadata.get("metadata"), dict) and metadata_key in metadata["metadata"]:
                    result[metadata_key] = metadata["metadata"][metadata_key]
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
    if args.task_type not in {
        "text_bench",
        "text_bench_interference",
        "audio_understanding",
        "asr",
    }:
        raise ValueError(f"Unsupported task_type: {args.task_type}")
    results = generate_responses(args)
    if not args.stream_output:
        output_dir = os.path.dirname(args.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        save_jsonl(args.output_path, results)


if __name__ == "__main__":
    main(parse_args())
