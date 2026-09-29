import json
import math
import os
import re
import time
from argparse import ArgumentParser, Namespace
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoProcessor, StoppingCriteria, StoppingCriteriaList, set_seed

from mechanistic.ablate_qwen_modules import (
    find_modules,
    first_tensor,
    parse_module_groups,
    replace_first_tensor,
)
from mechanistic.common import move_to_device
from mechanistic.extract_qwen_logit_hidden_drift import (
    load_qwen_thinker,
    processor_call,
)
from utils.audio import decode_audio
from utils.common import extract_jsonl
from utils.datasets import get_dataset
from utils.eval_common import (
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
    MULTILINGUAL_ANSWER_REGEXES,
    normalize_extracted_answer,
    normalize_response,
)
from utils.hf_local import resolve_local_snapshot
from utils.mc_scoring import extract_mc_answer, majority_vote


SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating "
    "text and speech."
)


RELEVANCE_STOPWORDS = {
    "a",
    "about",
    "above",
    "after",
    "again",
    "against",
    "all",
    "also",
    "am",
    "an",
    "and",
    "any",
    "answer",
    "are",
    "around",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "being",
    "below",
    "between",
    "both",
    "but",
    "by",
    "can",
    "choice",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "few",
    "following",
    "for",
    "format",
    "from",
    "had",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "itself",
    "last",
    "letter",
    "line",
    "may",
    "might",
    "more",
    "most",
    "must",
    "no",
    "nor",
    "not",
    "now",
    "off",
    "on",
    "only",
    "multiple",
    "of",
    "one",
    "or",
    "other",
    "our",
    "out",
    "over",
    "question",
    "response",
    "same",
    "she",
    "should",
    "some",
    "such",
    "step",
    "text",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "through",
    "think",
    "to",
    "too",
    "under",
    "up",
    "very",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "whom",
    "why",
    "will",
    "with",
    "without",
    "would",
    "you",
    "your",
}


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Qwen2.5-Omni inference with fixed soft gates on selected audio "
            "modules. This is a method pilot derived from causal mapping."
        )
    )
    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--task_split", type=str, required=True)
    parser.add_argument("--task_type", type=str, required=True)
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--mitigate_prompt", action="store_true")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--self_consistency_samples",
        type=int,
        default=1,
        help=(
            "Number of stochastic responses per example for self-consistency. "
            "When greater than 1, response is saved as a list and downstream "
            "evaluation must use --scs. Requires temperature > 0."
        ),
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap for fast pilot runs. Default runs the full split.",
    )
    parser.add_argument(
        "--sample_strategy",
        type=str,
        default="first",
        choices=["first", "random"],
        help="Subset selection strategy when --max_samples is set.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=None,
        help="Seed for --sample_strategy random. Defaults to --seed.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--benchmark_timing_path",
        type=str,
        default=None,
        help="Write per-measured-query efficiency records to this JSONL path.",
    )
    parser.add_argument(
        "--benchmark_summary_path",
        type=str,
        default=None,
        help="Write the method-level efficiency summary to this JSON path.",
    )
    parser.add_argument("--benchmark_method", type=str, default=None)
    parser.add_argument("--benchmark_warmup", type=int, default=0)
    parser.add_argument("--benchmark_measure", type=int, default=None)
    parser.add_argument("--benchmark_repeat_id", type=int, default=0)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from completed JSONL rows matched by source_index.",
    )
    parser.add_argument(
        "--overwrite_output",
        action="store_true",
        help="Allow truncating an existing output when --resume is not used.",
    )
    parser.add_argument(
        "--module_groups",
        default="",
        help=(
            "Semicolon-separated gates: label=regex;label2=regex2. "
            "Example: late25=^audio_tower\\.layers\\.25$"
        ),
    )
    parser.add_argument(
        "--gate_scale",
        type=float,
        default=1.0,
        help="Multiplier applied to matched module outputs. 1.0 disables gating.",
    )
    parser.add_argument(
        "--dynamic_gate_mode",
        type=str,
        default="none",
            choices=[
                "none",
                "answer_margin",
                "query_audio_relevance",
                "learned_query_relevance",
                "audio_text_relevance",
            ],
        help=(
            "Sample-level gate policy. 'answer_margin' runs a direct-answer "
            "preflight forward pass and applies --dynamic_low_scale when the "
            "MCQ answer margin is below --dynamic_margin_threshold. "
            "'query_audio_relevance' keeps audio modules open when the query "
            "appears to require audio understanding, and suppresses them otherwise. "
            "'learned_query_relevance' uses a lightweight classifier loaded from "
            "--relevance_classifier_path. 'audio_text_relevance' additionally "
            "uses available speech transcripts/metadata to estimate whether "
            "the concrete audio content overlaps with the query."
        ),
    )
    parser.add_argument(
        "--dynamic_probe_prompt_mode",
        type=str,
        default="direct_answer",
        choices=["direct_answer", "generation_prompt"],
        help=(
            "Prompt used for answer-margin preflight. direct_answer appends "
            "'Answer:' to force MCQ-letter logits; generation_prompt uses the "
            "normal chat prompt and is mainly diagnostic."
        ),
    )
    parser.add_argument(
        "--dynamic_margin_threshold",
        type=float,
        default=1.0,
        help="Margin threshold for --dynamic_gate_mode answer_margin.",
    )
    parser.add_argument(
        "--dynamic_low_scale",
        type=float,
        default=0.25,
        help="Gate scale used for low-margin samples in dynamic mode.",
    )
    parser.add_argument(
        "--dynamic_high_scale",
        type=float,
        default=1.0,
        help="Gate scale used for high-margin samples in dynamic mode.",
    )
    parser.add_argument(
        "--query_relevance_keywords",
        type=str,
        default=(
            "audio,speech,spoken,transcribe,transcription,utterance,asr"
        ),
        help=(
            "Comma-separated lowercase keywords for --dynamic_gate_mode "
            "query_audio_relevance. If any keyword appears in the query, audio "
            "is treated as task-relevant and --dynamic_high_scale is used."
        ),
    )
    parser.add_argument(
        "--query_relevance_scope",
        type=str,
        default="instruction",
        choices=["full_query", "instruction"],
        help=(
            "Text span used by --dynamic_gate_mode query_audio_relevance. "
            "'full_query' matches the whole query. 'instruction' matches only "
            "the first paragraph before the first blank line, which avoids "
            "false positives from task content such as legal 'speech' or "
            "biology 'transcription'."
        ),
    )
    parser.add_argument(
        "--relevance_classifier_path",
        type=str,
        default=None,
        help=(
            "Path to a JSON lightweight relevance classifier produced by "
            "mechanistic.train_relevance_classifier. Required for "
            "--dynamic_gate_mode learned_query_relevance."
        ),
    )
    parser.add_argument(
        "--learned_relevance_threshold",
        type=float,
        default=None,
        help=(
            "Decision threshold for learned_query_relevance. If omitted, the "
            "threshold stored in the classifier JSON is used, falling back to 0.5."
        ),
    )
    parser.add_argument(
        "--audio_text_relevance_threshold",
        type=float,
        default=0.08,
        help=(
            "Cosine-style lexical overlap threshold for --dynamic_gate_mode "
            "audio_text_relevance when a speech transcript is available."
        ),
    )
    parser.add_argument(
        "--audio_text_relevance_min_overlap",
        type=int,
        default=2,
        help=(
            "Minimum number of non-stopword overlaps required for "
            "--dynamic_gate_mode audio_text_relevance to keep audio open."
        ),
    )
    parser.add_argument(
        "--audio_text_relevance_idf_path",
        type=str,
        default=None,
        help=(
            "Optional JSON IDF weights for --dynamic_gate_mode "
            "audio_text_relevance. Use the IDF file emitted by "
            "mechanistic.replay_audio_text_relevance_gate_decisions so "
            "offline auditing and inference share the same weighting."
        ),
    )
    parser.add_argument(
        "--audio_text_relevance_min_idf",
        type=float,
        default=0.0,
        help=(
            "Optional lower bound on IDF for terms used by audio-text "
            "relevance. This filters high-frequency residual function words."
        ),
    )
    parser.add_argument(
        "--audio_text_relevance_scope",
        type=str,
        default="task_content",
        choices=["full_query", "instruction", "task_content"],
        help=(
            "Query span compared against an audio transcript. task_content "
            "drops the instruction paragraph and compares the actual question "
            "and choices."
        ),
    )
    parser.add_argument("--max_modules_per_group", type=int, default=8)
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--use_fast_processor", action="store_true")
    parser.add_argument(
        "--stop_strings",
        nargs="*",
        default=None,
        help=(
            "Optional generated-text stop strings. Use this for models that "
            "continue with synthetic turns, e.g. --stop_strings Human:"
        ),
    )
    parser.add_argument(
        "--truncate_response_mode",
        type=str,
        default="none",
        choices=["none", "first_human", "first_answer_line"],
        help=(
            "Optional post-decode truncation before saving responses. This is "
            "useful for diagnostics or for models that continue into extra "
            "Human turns after answering."
        ),
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
    )
    return parser.parse_args()


class GateController:
    def __init__(self, scale: float):
        self.scale = scale
        self.matched_modules: List[str] = []


class StopOnTokenSequences(StoppingCriteria):
    def __init__(self, stop_sequences: List[List[int]]):
        super().__init__()
        self.stop_sequences = [seq for seq in stop_sequences if seq]

    def __call__(self, input_ids: torch.LongTensor, _scores, **_kwargs) -> bool:
        if not self.stop_sequences:
            return False
        sequence = input_ids[0]
        for stop_sequence in self.stop_sequences:
            stop_len = len(stop_sequence)
            if stop_len <= sequence.shape[-1] and sequence[-stop_len:].tolist() == stop_sequence:
                return True
        return False


def get_inputs(
    sample: Dict[str, Any],
    processor: AutoProcessor,
    task_type: str,
    sampling_rate: int,
) -> Tuple[str, Optional[List[np.ndarray]]]:
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
    return prompt, audios


def make_gate_hook(controller: GateController):
    def hook(_module, _inputs, output):
        hidden = first_tensor(output)
        return replace_first_tensor(output, hidden * controller.scale)

    return hook


def install_gates(
    model: nn.Module,
    module_groups: str,
    controller: GateController,
    max_modules_per_group: int,
):
    if not module_groups:
        print("Audio soft gating disabled.")
        controller.matched_modules = []
        return []

    handles = []
    matched_module_names: List[str] = []
    for label, regex in parse_module_groups(module_groups):
        modules = find_modules(
            model=model,
            regex=regex,
            leaf_only=False,
            max_modules=max_modules_per_group,
        )
        print(
            f"Gate group {label}: initial_scale={controller.scale}, matched "
            f"{len(modules)} modules: {[name for name, _module in modules]}"
        )
        for name, module in modules:
            matched_module_names.append(name)
            handles.append(module.register_forward_hook(make_gate_hook(controller)))
    controller.matched_modules = matched_module_names
    return handles


def decode_generation(
    processor: AutoProcessor,
    input_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    stop_strings: Optional[List[str]] = None,
    truncate_response_mode: str = "none",
) -> str:
    prompt_len = input_ids.shape[-1]
    new_ids = generated_ids[0, prompt_len:]
    text = processor.tokenizer.decode(
        new_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return truncate_response(text, stop_strings, truncate_response_mode).strip()


def truncate_response(
    text: str,
    stop_strings: Optional[List[str]] = None,
    mode: str = "none",
) -> str:
    cut = len(text)
    if stop_strings:
        for stop_string in stop_strings:
            if not stop_string:
                continue
            pos = text.find(stop_string)
            if pos >= 0:
                cut = min(cut, pos)

    if mode == "first_human":
        pos = text.find("Human:")
        if pos >= 0:
            cut = min(cut, pos)
    elif mode == "first_answer_line":
        lines = text.splitlines(keepends=True)
        answer_seen = False
        char_count = 0
        for line in lines:
            char_count += len(line)
            stripped = line.strip().lower()
            if stripped.startswith("answer:") or stripped.startswith("**answer:"):
                answer_seen = True
                cut = min(cut, char_count)
                break
        if not answer_seen:
            pos = text.find("Human:")
            if pos >= 0:
                cut = min(cut, pos)
    elif mode != "none":
        raise ValueError(f"Unsupported truncate_response_mode: {mode}")

    return text[:cut]


def set_generation_seed(seed: int) -> None:
    """Set a per-example seed so self-consistency samples are reproducible."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize_cuda() -> None:
    """Make wall-clock timing include all queued CUDA work."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def summarize_self_consistency(
    responses: List[str], sample_seeds: Optional[List[int]] = None
) -> Dict[str, Any]:
    """Return auditable vote counts and the winning extracted answer."""
    extracted = [extract_mc_answer(response) for response in responses]
    vote = majority_vote(extracted)
    return {
        "sample_count": len(responses),
        "extracted_answers": extracted,
        **vote,
        "sample_seeds": sample_seeds or [],
    }


def make_stopping_criteria(
    processor: AutoProcessor,
    stop_strings: Optional[List[str]],
) -> Optional[StoppingCriteriaList]:
    if not stop_strings:
        return None
    stop_sequences = []
    tokenizer = processor.tokenizer
    for stop_string in stop_strings:
        if not stop_string:
            continue
        encoded = tokenizer.encode(stop_string, add_special_tokens=False)
        if encoded:
            stop_sequences.append(encoded)
    if not stop_sequences:
        return None
    return StoppingCriteriaList([StopOnTokenSequences(stop_sequences)])


def json_safe_metadata(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Keep lightweight dataset metadata without copying audio or prompt fields."""
    skip_keys = {
        "audio",
        "prompt",
        "query",
        "prefix",
        "choices",
        "answer",
        "response",
        "gate_info",
    }
    metadata: Dict[str, Any] = {}
    for key, value in sample.items():
        if key in skip_keys or key.startswith("_"):
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


def letter_token_ids(processor: AutoProcessor, letters: List[str]) -> Dict[str, List[int]]:
    tokenizer = processor.tokenizer
    ids: Dict[str, List[int]] = {}
    for letter in letters:
        variants = [letter, f" {letter}"]
        token_ids = []
        for variant in variants:
            encoded = tokenizer.encode(variant, add_special_tokens=False)
            if encoded:
                token_ids.append(encoded[-1])
        ids[letter] = sorted(set(token_ids))
    return ids


def probe_prompt(prompt: str, mode: str) -> str:
    if mode == "direct_answer":
        return prompt.rstrip() + "\nAnswer:"
    if mode == "generation_prompt":
        return prompt
    raise ValueError(f"Unsupported dynamic probe prompt mode: {mode}")


def compute_answer_margin(
    model: nn.Module,
    processor: AutoProcessor,
    prompt: str,
    audios: Optional[List[np.ndarray]],
    probe_mode: str,
) -> Dict[str, Any]:
    letters = ["A", "B", "C", "D", "E"]
    batch = processor_call(processor, probe_prompt(prompt, probe_mode), audios)
    device = next(model.parameters()).device
    batch = move_to_device(batch, device)
    token_ids = letter_token_ids(processor, letters)

    with torch.inference_mode():
        outputs = model(**batch, use_cache=False)
    logits = outputs.logits[0, -1].float()

    scores = {}
    for letter, ids in token_ids.items():
        if not ids:
            continue
        scores[letter] = float(torch.max(logits[ids]).item())
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) < 2:
        return {
            "direct_pred": ranked[0][0] if ranked else None,
            "answer_margin": None,
            "answer_scores": scores,
        }
    return {
        "direct_pred": ranked[0][0],
        "answer_margin": ranked[0][1] - ranked[1][1],
        "answer_scores": scores,
    }


def relevance_text_span(query: str, scope: str) -> str:
    if scope == "full_query":
        return query
    if scope == "instruction":
        normalized = query.replace("\r\n", "\n")
        return normalized.split("\n\n", 1)[0]
    if scope == "task_content":
        normalized = query.replace("\r\n", "\n")
        parts = normalized.split("\n\n", 1)
        return parts[1] if len(parts) == 2 else normalized
    raise ValueError(f"Unsupported query relevance scope: {scope}")


def query_audio_relevance(
    query: str,
    keywords: str,
    scope: str,
) -> Dict[str, Any]:
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


def _string_from_nested(sample: Dict[str, Any], keys: List[str]) -> Optional[str]:
    for key in keys:
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        for key in keys:
            value = metadata.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def transcript_from_speech_source(speech_source: str) -> Optional[str]:
    if not speech_source:
        return None
    utterance_id = os.path.splitext(os.path.basename(speech_source))[0]
    directory = os.path.dirname(speech_source)
    if not utterance_id or not directory:
        return None
    parts = utterance_id.split("-")
    candidates = []
    if len(parts) >= 2:
        candidates.append(os.path.join(directory, f"{parts[0]}-{parts[1]}.trans.txt"))
    candidates.append(os.path.join(directory, f"{utterance_id}.trans.txt"))
    for path in candidates:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line.startswith(utterance_id + " "):
                    continue
                return line[len(utterance_id) :].strip()
    return None


def extract_audio_transcript(sample: Dict[str, Any]) -> Tuple[Optional[str], str]:
    direct = _string_from_nested(
        sample,
        [
            "speech_transcript",
            "audio_transcript",
            "transcript",
            "asr_transcript",
        ],
    )
    if direct:
        return direct, "metadata"
    speech_source = _string_from_nested(sample, ["speech_source"])
    if speech_source:
        transcript = transcript_from_speech_source(speech_source)
        if transcript:
            return transcript, "speech_source_trans_file"
    return None, "missing"


def relevance_terms(text: str) -> List[str]:
    terms = []
    for term in re.findall(r"[a-z0-9]+", text.lower()):
        if len(term) < 3:
            continue
        if term in RELEVANCE_STOPWORDS:
            continue
        terms.append(term)
    return terms


def load_audio_text_idf(path: str) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "idf" in payload:
        payload = payload["idf"]
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid audio-text IDF JSON at {path}")
    idf: Dict[str, float] = {}
    for key, value in payload.items():
        try:
            idf[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return idf


def lexical_overlap_relevance(
    query_text: str,
    transcript: str,
    idf: Optional[Dict[str, float]] = None,
    min_idf: float = 0.0,
) -> Dict[str, Any]:
    query_terms = set(relevance_terms(query_text))
    transcript_terms = set(relevance_terms(transcript))
    raw_overlap = sorted(query_terms.intersection(transcript_terms))
    default_idf = max(idf.values()) if idf else 1.0
    if idf is not None:
        query_terms = {
            term for term in query_terms if idf.get(term, default_idf) >= min_idf
        }
        transcript_terms = {
            term for term in transcript_terms if idf.get(term, default_idf) >= min_idf
        }
    overlap = sorted(query_terms.intersection(transcript_terms))
    if not query_terms or not transcript_terms:
        score = 0.0
        numerator = 0.0
        query_norm = 0.0
        transcript_norm = 0.0
    else:
        query_weights = {term: idf.get(term, default_idf) if idf else 1.0 for term in query_terms}
        transcript_weights = {
            term: idf.get(term, default_idf) if idf else 1.0 for term in transcript_terms
        }
        numerator = sum(query_weights[term] * transcript_weights[term] for term in overlap)
        query_norm = math.sqrt(sum(weight * weight for weight in query_weights.values()))
        transcript_norm = math.sqrt(
            sum(weight * weight for weight in transcript_weights.values())
        )
        denominator = query_norm * transcript_norm
        score = numerator / denominator if denominator else 0.0
    overlap_weights = {
        term: idf.get(term, default_idf) if idf else 1.0 for term in overlap
    }
    return {
        "audio_text_relevance_score": score,
        "audio_text_overlap_terms": overlap,
        "audio_text_raw_overlap_terms": raw_overlap,
        "audio_text_overlap_weights": overlap_weights,
        "audio_text_query_term_count": len(query_terms),
        "audio_text_transcript_term_count": len(transcript_terms),
        "audio_text_weighted_overlap": numerator,
        "audio_text_query_weight_norm": query_norm,
        "audio_text_transcript_weight_norm": transcript_norm,
        "audio_text_uses_idf": idf is not None,
        "audio_text_min_idf": min_idf,
    }


def audio_text_relevance(
    sample: Dict[str, Any],
    query_keywords: str,
    query_scope: str,
    text_scope: str,
    threshold: float,
    min_overlap: int,
    idf: Optional[Dict[str, float]] = None,
    min_idf: float = 0.0,
) -> Dict[str, Any]:
    query = str(sample.get("query", ""))
    instruction_signal = query_audio_relevance(
        query=query,
        keywords=query_keywords,
        scope=query_scope,
    )
    if instruction_signal["audio_relevant"]:
        return {
            **instruction_signal,
            "audio_relevant": True,
            "audio_text_relevance_reason": "instruction_requires_audio",
            "audio_text_relevance_threshold": threshold,
            "audio_text_relevance_min_overlap": min_overlap,
            "audio_text_relevance_scope": text_scope,
            "audio_text_uses_idf": idf is not None,
            "audio_text_min_idf": min_idf,
        }

    transcript, transcript_source = extract_audio_transcript(sample)
    query_text = relevance_text_span(query, text_scope)
    if not transcript:
        return {
            **instruction_signal,
            "audio_relevant": False,
            "audio_text_relevance_reason": "no_audio_transcript",
            "audio_text_relevance_threshold": threshold,
            "audio_text_relevance_min_overlap": min_overlap,
            "audio_text_relevance_scope": text_scope,
            "audio_text_transcript_source": transcript_source,
            "audio_text_uses_idf": idf is not None,
            "audio_text_min_idf": min_idf,
            "query_relevance_text": query_text,
        }

    overlap_info = lexical_overlap_relevance(
        query_text,
        transcript,
        idf=idf,
        min_idf=min_idf,
    )
    overlap_count = len(overlap_info["audio_text_overlap_terms"])
    audio_relevant = (
        overlap_info["audio_text_relevance_score"] >= threshold
        and overlap_count >= min_overlap
    )
    return {
        **instruction_signal,
        **overlap_info,
        "audio_relevant": audio_relevant,
        "audio_text_relevance_reason": (
            "transcript_overlap" if audio_relevant else "transcript_low_overlap"
        ),
        "audio_text_relevance_threshold": threshold,
        "audio_text_relevance_min_overlap": min_overlap,
        "audio_text_relevance_scope": text_scope,
        "audio_text_transcript_source": transcript_source,
        "audio_text_transcript_excerpt": transcript[:240],
        "query_relevance_text": query_text,
    }


def tokenize_for_relevance(text: str, ngram_max: int = 2) -> List[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    tokens = list(words)
    if ngram_max >= 2:
        tokens.extend(
            f"{words[i]}_{words[i + 1]}" for i in range(max(0, len(words) - 1))
        )
    return tokens


def load_relevance_classifier(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        model = json.load(f)
    required_keys = {"class_log_prior", "feature_log_prob", "default_log_prob"}
    missing = required_keys.difference(model)
    if missing:
        raise ValueError(
            f"Invalid relevance classifier at {path}; missing keys: {sorted(missing)}"
        )
    return model


def learned_query_audio_relevance(
    query: str,
    scope: str,
    classifier: Dict[str, Any],
    threshold: Optional[float],
) -> Dict[str, Any]:
    relevance_text = relevance_text_span(query, scope)
    ngram_max = int(classifier.get("ngram_max", 2))
    tokens = tokenize_for_relevance(relevance_text, ngram_max=ngram_max)
    token_counts: Dict[str, int] = {}
    for token in tokens:
        token_counts[token] = token_counts.get(token, 0) + 1

    class_log_prior = classifier["class_log_prior"]
    feature_log_prob = classifier["feature_log_prob"]
    default_log_prob = classifier["default_log_prob"]
    classes = list(class_log_prior.keys())
    scores: Dict[str, float] = {}
    top_features: Dict[str, List[Tuple[str, float]]] = {}
    for class_name in classes:
        score = float(class_log_prior[class_name])
        feature_scores = []
        class_features = feature_log_prob.get(class_name, {})
        default = float(default_log_prob[class_name])
        for token, count in token_counts.items():
            token_log_prob = float(class_features.get(token, default))
            contribution = count * token_log_prob
            score += contribution
            if token in class_features:
                feature_scores.append((token, contribution))
        scores[class_name] = score
        top_features[class_name] = sorted(
            feature_scores, key=lambda item: item[1], reverse=True
        )[:10]

    relevant_score = scores.get("relevant", float("-inf"))
    irrelevant_score = scores.get("irrelevant", float("-inf"))
    max_score = max(relevant_score, irrelevant_score)
    relevant_exp = math.exp(relevant_score - max_score)
    irrelevant_exp = math.exp(irrelevant_score - max_score)
    relevant_prob = relevant_exp / (relevant_exp + irrelevant_exp)
    decision_threshold = (
        float(threshold)
        if threshold is not None
        else float(classifier.get("threshold", 0.5))
    )
    audio_relevant = relevant_prob >= decision_threshold
    return {
        "audio_relevant": audio_relevant,
        "learned_relevance_prob": relevant_prob,
        "learned_relevance_threshold": decision_threshold,
        "learned_relevance_scores": scores,
        "learned_relevance_top_features": top_features,
        "query_relevance_scope": scope,
        "query_relevance_text": relevance_text,
        "relevance_classifier_path": classifier.get("model_path"),
        "relevance_classifier_type": classifier.get("model_type", "multinomial_nb"),
    }


def choose_dynamic_gate_scale(
    args: Namespace,
    preflight: Dict[str, Any],
) -> float:
    if args.dynamic_gate_mode == "none":
        return args.gate_scale
    if args.dynamic_gate_mode == "answer_margin":
        margin = preflight.get("answer_margin")
        if margin is None or margin < args.dynamic_margin_threshold:
            return args.dynamic_low_scale
        return args.dynamic_high_scale
    if args.dynamic_gate_mode == "query_audio_relevance":
        if preflight.get("audio_relevant"):
            return args.dynamic_high_scale
        return args.dynamic_low_scale
    if args.dynamic_gate_mode == "learned_query_relevance":
        if preflight.get("audio_relevant"):
            return args.dynamic_high_scale
        return args.dynamic_low_scale
    if args.dynamic_gate_mode == "audio_text_relevance":
        if preflight.get("audio_relevant"):
            return args.dynamic_high_scale
        return args.dynamic_low_scale
    raise ValueError(f"Unsupported dynamic gate mode: {args.dynamic_gate_mode}")


def generate_one(
    model: nn.Module,
    processor: AutoProcessor,
    sample: Dict[str, Any],
    task_type: str,
    sampling_rate: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    args: Namespace,
    gate_controller: GateController,
) -> Tuple[str, Dict[str, Any], Dict[str, int]]:
    prompt, audios = get_inputs(
        sample=sample,
        processor=processor,
        task_type=task_type,
        sampling_rate=sampling_rate,
    )

    gate_info: Dict[str, Any] = {
        "dynamic_gate_mode": args.dynamic_gate_mode,
        "gate_scale": args.gate_scale,
        "applied_gate_scale": gate_controller.scale,
        "module_groups": args.module_groups,
        "matched_modules": list(gate_controller.matched_modules),
    }
    if task_type != "text_bench" and args.dynamic_gate_mode == "answer_margin":
        gate_controller.scale = 1.0
        preflight = compute_answer_margin(
            model=model,
            processor=processor,
            prompt=prompt,
            audios=audios,
            probe_mode=args.dynamic_probe_prompt_mode,
        )
        gate_controller.scale = choose_dynamic_gate_scale(args, preflight)
        answer_margin = preflight.get("answer_margin")
        gate_info = {
            "dynamic_gate_mode": args.dynamic_gate_mode,
            "dynamic_probe_prompt_mode": args.dynamic_probe_prompt_mode,
            "dynamic_margin_threshold": args.dynamic_margin_threshold,
            "dynamic_low_scale": args.dynamic_low_scale,
            "dynamic_high_scale": args.dynamic_high_scale,
            "applied_gate_scale": gate_controller.scale,
            "module_groups": args.module_groups,
            "matched_modules": list(gate_controller.matched_modules),
            "gate_decision": (
                "low_scale"
                if answer_margin is None or answer_margin < args.dynamic_margin_threshold
                else "high_scale"
            ),
            **preflight,
        }
    elif task_type != "text_bench" and args.dynamic_gate_mode == "query_audio_relevance":
        preflight = query_audio_relevance(
            query=sample["query"],
            keywords=args.query_relevance_keywords,
            scope=args.query_relevance_scope,
        )
        gate_controller.scale = choose_dynamic_gate_scale(args, preflight)
        gate_info = {
            "dynamic_gate_mode": args.dynamic_gate_mode,
            "dynamic_low_scale": args.dynamic_low_scale,
            "dynamic_high_scale": args.dynamic_high_scale,
            "applied_gate_scale": gate_controller.scale,
            "module_groups": args.module_groups,
            "matched_modules": list(gate_controller.matched_modules),
            "gate_decision": (
                "high_scale" if preflight["audio_relevant"] else "low_scale"
            ),
            **preflight,
        }
    elif task_type != "text_bench" and args.dynamic_gate_mode == "learned_query_relevance":
        preflight = learned_query_audio_relevance(
            query=sample["query"],
            scope=args.query_relevance_scope,
            classifier=args._relevance_classifier,
            threshold=args.learned_relevance_threshold,
        )
        gate_controller.scale = choose_dynamic_gate_scale(args, preflight)
        gate_info = {
            "dynamic_gate_mode": args.dynamic_gate_mode,
            "dynamic_low_scale": args.dynamic_low_scale,
            "dynamic_high_scale": args.dynamic_high_scale,
            "applied_gate_scale": gate_controller.scale,
            "module_groups": args.module_groups,
            "matched_modules": list(gate_controller.matched_modules),
            "gate_decision": (
                "high_scale" if preflight["audio_relevant"] else "low_scale"
            ),
            **preflight,
        }
    elif task_type != "text_bench" and args.dynamic_gate_mode == "audio_text_relevance":
        preflight = audio_text_relevance(
            sample=sample,
            query_keywords=args.query_relevance_keywords,
            query_scope=args.query_relevance_scope,
            text_scope=args.audio_text_relevance_scope,
            threshold=args.audio_text_relevance_threshold,
            min_overlap=args.audio_text_relevance_min_overlap,
            idf=args._audio_text_relevance_idf,
            min_idf=args.audio_text_relevance_min_idf,
        )
        gate_controller.scale = choose_dynamic_gate_scale(args, preflight)
        gate_info = {
            "dynamic_gate_mode": args.dynamic_gate_mode,
            "dynamic_low_scale": args.dynamic_low_scale,
            "dynamic_high_scale": args.dynamic_high_scale,
            "applied_gate_scale": gate_controller.scale,
            "module_groups": args.module_groups,
            "matched_modules": list(gate_controller.matched_modules),
            "gate_decision": (
                "high_scale" if preflight["audio_relevant"] else "low_scale"
            ),
            **preflight,
        }
    else:
        gate_controller.scale = args.gate_scale
        gate_info["applied_gate_scale"] = gate_controller.scale

    batch = processor_call(processor, prompt, audios)
    device = next(model.parameters()).device
    batch = move_to_device(batch, device)

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "use_cache": True,
    }
    stopping_criteria = make_stopping_criteria(processor, args.stop_strings)
    if stopping_criteria is not None:
        generation_kwargs["stopping_criteria"] = stopping_criteria
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p

    with torch.inference_mode():
        generated_ids = model.generate(**batch, **generation_kwargs)
    response = decode_generation(
        processor,
        batch["input_ids"],
        generated_ids,
        stop_strings=args.stop_strings,
        truncate_response_mode=args.truncate_response_mode,
    )
    input_tokens = int(batch["input_ids"].shape[-1])
    generated_tokens = max(int(generated_ids.shape[-1]) - input_tokens, 0)
    return response, gate_info, {
        "input_tokens": input_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": input_tokens + generated_tokens,
    }


def generate_responses(
    args: Namespace,
) -> List[Dict[str, Any]]:
    benchmark_enabled = args.benchmark_timing_path is not None
    if benchmark_enabled != (args.benchmark_summary_path is not None):
        raise ValueError(
            "--benchmark_timing_path and --benchmark_summary_path must be "
            "provided together."
        )
    if benchmark_enabled:
        if args.resume:
            raise ValueError("Benchmark runs do not support --resume.")
        if not args.benchmark_method:
            raise ValueError("Benchmark runs require --benchmark_method.")
        if args.max_samples is None:
            raise ValueError(
                "Benchmark runs require --max_samples >= warmup + measured samples."
            )
        if args.benchmark_warmup < 0:
            raise ValueError("--benchmark_warmup must be non-negative.")
        if args.benchmark_measure is None:
            args.benchmark_measure = args.max_samples - args.benchmark_warmup
        if args.benchmark_measure <= 0:
            raise ValueError("--benchmark_measure must be positive.")
        required_samples = args.benchmark_warmup + args.benchmark_measure
        if args.max_samples < required_samples:
            raise ValueError(
                f"--max_samples={args.max_samples} is smaller than the required "
                f"warmup + measured count {required_samples}."
            )
    if args.self_consistency_samples < 1:
        raise ValueError("--self_consistency_samples must be at least 1.")
    if args.self_consistency_samples > 1 and args.temperature <= 0:
        raise ValueError(
            "Self-consistency requires stochastic decoding; set --temperature "
            "to a value greater than 0."
        )
    if args.dynamic_gate_mode != "none" and not args.module_groups.strip():
        raise ValueError(
            "Dynamic gating requires --module_groups so the selected scale is "
            "applied to a real model module."
        )
    if args.dynamic_gate_mode == "answer_margin" and not (
        "arc" in args.task_id or "mmlu" in args.task_id
    ):
        raise ValueError(
            "--dynamic_gate_mode answer_margin currently supports only MCQ "
            "tasks with A/B/C/D/E answer labels, e.g. ARC or MMLU. "
            "Use it only for ARC or MMLU multiple-choice tasks."
        )
    if args.dynamic_gate_mode in {
        "query_audio_relevance",
        "learned_query_relevance",
        "audio_text_relevance",
    } and args.task_type == "text_bench":
        raise ValueError(
            f"--dynamic_gate_mode {args.dynamic_gate_mode} is intended for "
            "text_bench_interference samples with an audio input. Use "
            "--dynamic_gate_mode none for clean text-only runs."
        )
    if args.dynamic_gate_mode == "learned_query_relevance":
        if not args.relevance_classifier_path:
            raise ValueError(
                "--relevance_classifier_path is required for "
                "--dynamic_gate_mode learned_query_relevance."
            )
        args._relevance_classifier = load_relevance_classifier(
            args.relevance_classifier_path
        )
        args._relevance_classifier["model_path"] = args.relevance_classifier_path
    else:
        args._relevance_classifier = None
    if args.audio_text_relevance_idf_path:
        args._audio_text_relevance_idf = load_audio_text_idf(
            args.audio_text_relevance_idf_path
        )
    else:
        args._audio_text_relevance_idf = None

    dataset = get_dataset(args.task_id, args.task_split, args.mitigate_prompt)
    selected_indices = list(range(len(dataset)))
    if args.max_samples is not None:
        sample_count = min(args.max_samples, len(dataset))
        if args.sample_strategy == "first":
            selected_indices = list(range(sample_count))
        elif args.sample_strategy == "random":
            rng = np.random.default_rng(args.sample_seed if args.sample_seed is not None else args.seed)
            selected_indices = rng.choice(len(dataset), size=sample_count, replace=False).tolist()
        else:
            raise ValueError(f"Unsupported sample_strategy: {args.sample_strategy}")
        dataset = dataset.select(selected_indices)

    if benchmark_enabled and len(selected_indices) < (
        args.benchmark_warmup + args.benchmark_measure
    ):
        raise ValueError("The selected dataset is smaller than the benchmark plan.")

    model_source = resolve_local_snapshot(args.model_id)
    processor = AutoProcessor.from_pretrained(
        model_source,
        trust_remote_code=True,
        local_files_only=True,
        use_fast=args.use_fast_processor,
    )
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.dtype]
    model = load_qwen_thinker(model_source, dtype=dtype)
    model.eval()

    initial_scale = args.gate_scale
    if args.dynamic_gate_mode != "none":
        initial_scale = 1.0
    gate_controller = GateController(initial_scale)
    handles = install_gates(
        model=model,
        module_groups=args.module_groups,
        controller=gate_controller,
        max_modules_per_group=args.max_modules_per_group,
    )
    if args.dynamic_gate_mode != "none" and not handles:
        raise RuntimeError("Dynamic gating installed no forward hooks.")

    run_config = {
        "backend": "transformers",
        "model_id": args.model_id,
        "task_id": args.task_id,
        "task_split": args.task_split,
        "task_type": args.task_type,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "self_consistency_samples": args.self_consistency_samples,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "max_samples": args.max_samples,
        "sample_strategy": args.sample_strategy,
        "sample_seed": args.sample_seed,
        "module_groups": args.module_groups,
        "matched_modules": list(gate_controller.matched_modules),
        "gate_scale": args.gate_scale,
        "dynamic_gate_mode": args.dynamic_gate_mode,
        "dynamic_low_scale": args.dynamic_low_scale,
        "dynamic_high_scale": args.dynamic_high_scale,
        "query_relevance_scope": args.query_relevance_scope,
        "query_relevance_keywords": args.query_relevance_keywords,
        "truncate_response_mode": args.truncate_response_mode,
        "stop_strings": args.stop_strings,
        "dtype": args.dtype,
    }
    if benchmark_enabled:
        run_config["benchmark"] = {
            "method": args.benchmark_method,
            "warmup_queries": args.benchmark_warmup,
            "measured_queries": args.benchmark_measure,
            "repeat_id": args.benchmark_repeat_id,
            "timing_boundary": (
                "cached sample -> input formatting -> audio preprocessing -> "
                "routing -> generation -> answer extraction"
            ),
            "model_loading_included": False,
            "disk_io_included": False,
        }

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    existing_results: List[Dict[str, Any]] = []
    completed_source_indices = set()
    if os.path.exists(args.output_path):
        if args.resume:
            existing_results = extract_jsonl(args.output_path)
            for row in existing_results:
                source_index = row.get("source_index")
                if source_index in completed_source_indices:
                    raise ValueError(
                        f"Duplicate source_index in existing output: {source_index}"
                    )
                if source_index not in selected_indices:
                    raise ValueError(
                        f"Existing source_index {source_index} is not in the selected dataset."
                    )
                completed_source_indices.add(source_index)
        elif args.overwrite_output:
            open(args.output_path, "w", encoding="utf-8").close()
        else:
            raise FileExistsError(
                f"Output already exists: {args.output_path}. Use --resume or "
                "--overwrite_output explicitly."
            )

    if benchmark_enabled:
        for path in [args.benchmark_timing_path, args.benchmark_summary_path]:
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
        open(args.benchmark_timing_path, "w", encoding="utf-8").close()

    results = list(existing_results)
    benchmark_records: List[Dict[str, Any]] = []
    benchmark_peak_reset = not benchmark_enabled
    try:
        for row_idx, sample in enumerate(tqdm(dataset, desc="Qwen gated inference")):
            source_index = selected_indices[row_idx]
            if source_index in completed_source_indices:
                continue
            measured_query = (
                benchmark_enabled
                and args.benchmark_warmup
                <= row_idx
                < args.benchmark_warmup + args.benchmark_measure
            )
            if benchmark_enabled and not benchmark_peak_reset and row_idx >= args.benchmark_warmup:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                benchmark_peak_reset = True

            query_start = None
            gpu_start = None
            if measured_query:
                synchronize_cuda()
                query_start = time.perf_counter()
                if torch.cuda.is_available():
                    gpu_start = torch.cuda.Event(enable_timing=True)
                    gpu_start.record()
            sample_responses: List[str] = []
            sample_gate_infos: List[Dict[str, Any]] = []
            sample_seeds: List[int] = []
            sample_token_metas: List[Dict[str, int]] = []
            sample_completion_wall_times: List[float] = []
            for sample_idx in range(args.self_consistency_samples):
                if args.self_consistency_samples > 1:
                    sample_seed = (
                        args.seed
                        + source_index * args.self_consistency_samples
                        + sample_idx
                    )
                    sample_seeds.append(sample_seed)
                    set_generation_seed(
                        sample_seed
                    )
                completion_start = time.perf_counter() if measured_query else None
                generated_text, gate_info, token_meta = generate_one(
                    model=model,
                    processor=processor,
                    sample=sample,
                    task_type=args.task_type,
                    sampling_rate=args.sampling_rate,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_new_tokens=args.max_new_tokens,
                    args=args,
                    gate_controller=gate_controller,
                )
                if measured_query and completion_start is not None:
                    sample_completion_wall_times.append(
                        time.perf_counter() - completion_start
                    )
                sample_responses.append(generated_text)
                sample_gate_infos.append(gate_info)
                sample_token_metas.append(token_meta)
            if args.self_consistency_samples > 1:
                response_value: Any = sample_responses
                gate_info = {
                    "self_consistency_samples": args.self_consistency_samples,
                    "sample_gate_info": sample_gate_infos,
                    "vote_summary": summarize_self_consistency(sample_responses, sample_seeds),
                }
            else:
                response_value = sample_responses[0]
                gate_info = sample_gate_infos[0]
            if measured_query:
                gpu_end = None
                if torch.cuda.is_available() and gpu_start is not None:
                    gpu_end = torch.cuda.Event(enable_timing=True)
                    gpu_end.record()
                synchronize_cuda()
                query_end = time.perf_counter()
                latency_sec = query_end - query_start
                gpu_compute_sec = (
                    gpu_start.elapsed_time(gpu_end) / 1000.0
                    if gpu_start is not None and gpu_end is not None
                    else None
                )
            else:
                latency_sec = None
                gpu_compute_sec = None
            metadata = json_safe_metadata(sample)
            result = {
                "source_index": source_index,
                "id": sample.get("id", ""),
                "subject": sample.get("subject", ""),
                "task": sample.get("task", ""),
                "prompt": sample.get("prompt", ""),
                "query": sample["query"],
                "prefix": sample.get("prefix", ""),
                "choices": sample.get("choices", ""),
                "response": response_value,
                "answer": sample["answer"],
                "gate_info": gate_info,
                "metadata": metadata,
                "run_config": run_config,
            }
            if args.self_consistency_samples > 1:
                result["self_consistency"] = gate_info["vote_summary"]
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
            if measured_query:
                winner = (
                    gate_info["vote_summary"].get("winner")
                    if args.self_consistency_samples > 1
                    else extract_mc_answer(response_value)
                )
                timing_record = {
                    "method": args.benchmark_method,
                    "task_id": args.task_id,
                    "task_split": args.task_split,
                    "task_type": args.task_type,
                    "source_index": source_index,
                    "id": sample.get("id", ""),
                    "repeat_id": args.benchmark_repeat_id,
                    "latency_sec": latency_sec,
                    "gpu_compute_sec": gpu_compute_sec,
                    "input_tokens": int(
                        sum(meta["input_tokens"] for meta in sample_token_metas)
                    ),
                    "generated_tokens": int(
                        sum(meta["generated_tokens"] for meta in sample_token_metas)
                    ),
                    "total_tokens": int(
                        sum(meta["total_tokens"] for meta in sample_token_metas)
                    ),
                    "completions": args.self_consistency_samples,
                    "completion_latency_sec": sample_completion_wall_times,
                    "completion_input_tokens": [
                        meta["input_tokens"] for meta in sample_token_metas
                    ],
                    "completion_generated_tokens": [
                        meta["generated_tokens"] for meta in sample_token_metas
                    ],
                    "prediction": winner,
                }
                benchmark_records.append(timing_record)
                with open(args.benchmark_timing_path, "a", encoding="utf-8") as timing_handle:
                    timing_handle.write(
                        json.dumps(timing_record, ensure_ascii=False) + "\n"
                    )
            results.append(result)
            with open(args.output_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            completed_source_indices.add(source_index)
    finally:
        for handle in handles:
            handle.remove()

    if benchmark_enabled:
        latencies = np.asarray(
            [record["latency_sec"] for record in benchmark_records], dtype=np.float64
        )
        gpu_times = [
            record["gpu_compute_sec"]
            for record in benchmark_records
            if record["gpu_compute_sec"] is not None
        ]
        benchmark_summary = {
            "method": args.benchmark_method,
            "task_id": args.task_id,
            "task_split": args.task_split,
            "repeat_id": args.benchmark_repeat_id,
            "warmup_queries": args.benchmark_warmup,
            "measured_queries": len(benchmark_records),
            "latency_total_sec": float(latencies.sum()),
            "latency_mean_sec": float(latencies.mean()),
            "latency_median_sec": float(np.median(latencies)),
            "latency_p95_sec": float(np.percentile(latencies, 95)),
            "throughput_queries_per_sec": float(len(latencies) / latencies.sum()),
            "gpu_compute_mean_sec": float(np.mean(gpu_times)) if gpu_times else None,
            "gpu_compute_total_sec": float(np.sum(gpu_times)) if gpu_times else None,
            "input_tokens_mean": float(
                np.mean([record["input_tokens"] for record in benchmark_records])
            ),
            "generated_tokens_mean": float(
                np.mean([record["generated_tokens"] for record in benchmark_records])
            ),
            "total_tokens_mean": float(
                np.mean([record["total_tokens"] for record in benchmark_records])
            ),
            "completions_per_query": args.self_consistency_samples,
            "peak_memory_allocated_bytes": (
                int(torch.cuda.max_memory_allocated())
                if torch.cuda.is_available()
                else None
            ),
            "peak_memory_reserved_bytes": (
                int(torch.cuda.max_memory_reserved())
                if torch.cuda.is_available()
                else None
            ),
            "timing_path": args.benchmark_timing_path,
        }
        with open(args.benchmark_summary_path, "w", encoding="utf-8") as summary_handle:
            json.dump(benchmark_summary, summary_handle, indent=2, ensure_ascii=False)
            summary_handle.write("\n")

    return results


def main(args: Namespace) -> None:
    set_seed(args.seed)
    assert args.task_type in ["text_bench", "text_bench_interference"]
    results = generate_responses(args)
    print(f"Saved {len(results)} completed rows to {args.output_path}")


if __name__ == "__main__":
    main(parse_args())
