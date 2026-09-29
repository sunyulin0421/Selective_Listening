import json
import math
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch

from utils.eval_common import (MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
                               MULTILINGUAL_ANSWER_REGEXES,
                               normalize_extracted_answer, normalize_response)


MC_LABELS = ["A", "B", "C", "D", "E"]

FINAL_MC_ANSWER_PATTERNS = [
    # Answer: C, **Answer:** C, Answer: **C**
    r"(?is)(?:\*{1,2}\s*)?answer\s*(?:\*{1,2}\s*)?\s*:\s*"
    r"(?:\*{1,2}\s*)?([A-E])(?:\s*\*{1,2})?\b",
    # The answer is C, correct answer should be option C
    r"(?is)(?:the\s+)?(?:correct\s+)?answer\s+(?:is|should\s+be)\s*"
    r"(?:option\s*)?(?:\*{1,2}\s*)?([A-E])(?:\s*\*{1,2})?\b",
    # option C, choice C
    r"(?is)(?:option|choice)\s*(?:\*{1,2}\s*)?([A-E])"
    r"(?:\s*\*{1,2})?\b",
    # A final standalone letter line.
    r"(?im)^\s*(?:\*{1,2}\s*)?([A-E])(?:\s*\*{1,2})?\s*[\).:]?\s*$",
]


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_final_mc_answer_letter_span(response: str) -> Optional[Tuple[int, str]]:
    best_match = None
    for pattern in FINAL_MC_ANSWER_PATTERNS:
        for match in re.finditer(pattern, response):
            if best_match is None or match.start(1) > best_match.start(1):
                best_match = match
    if best_match is None:
        return None
    return best_match.start(1), best_match.group(1).upper()


def iter_condition_args(values: Iterable[str]) -> Dict[str, str]:
    parsed = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                f"Condition must be formatted as name=path, got: {value}"
            )
        name, path = value.split("=", 1)
        parsed[name.strip()] = path.strip()
    return parsed


def extract_mc_answer(response: str) -> Optional[str]:
    normalized = normalize_response(response)
    for pattern in MULTILINGUAL_ANSWER_REGEXES:
        regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(pattern)
        match = re.search(regex, normalized)
        if match:
            return normalize_extracted_answer(match.group(1)).upper()
    return None


def normalize_gold_mc(answer: Any) -> Optional[str]:
    if isinstance(answer, int):
        if 0 <= answer < len(MC_LABELS):
            return MC_LABELS[answer]
        return None
    if isinstance(answer, str):
        stripped = answer.strip().upper()
        if stripped in MC_LABELS:
            return stripped
    return None

def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    moved = {}
    for key, value in batch.items():
        if hasattr(value, "to"):
            moved[key] = value.to(device)
        elif isinstance(value, list):
            moved[key] = [
                item.to(device) if hasattr(item, "to") else item for item in value
            ]
        else:
            moved[key] = value
    return moved


def parse_layer_list(value: str, num_layers: Optional[int] = None) -> List[int]:
    value = value.strip()
    if value == "all":
        if num_layers is None:
            raise ValueError("num_layers is required when layers='all'")
        return list(range(num_layers))

    layers = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        layer = int(part)
        if layer < 0 and num_layers is not None:
            layer = num_layers + layer
        layers.append(layer)
    return layers


def tensor_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.linalg.vector_norm((a - b).float()).item()


def tensor_cosine_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().flatten()
    b = b.float().flatten()
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if denom.item() == 0:
        return math.nan
    return float(1.0 - torch.dot(a, b).item() / denom.item())
