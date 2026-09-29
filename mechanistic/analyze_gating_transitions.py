import json
import math
import os
import re
from argparse import ArgumentParser, Namespace
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from utils.common import extract_jsonl, save_jsonl
from utils.eval_common import (
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE,
    MULTILINGUAL_ANSWER_REGEXES,
    normalize_extracted_answer,
    normalize_response,
)
from utils.mc_scoring import majority_vote


CHOICES = ["A", "B", "C", "D", "E"]


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Analyze clean/ungated/gated transition patterns for a gating "
            "pilot without re-running inference."
        )
    )
    parser.add_argument("--clean_path", type=str, required=True)
    parser.add_argument("--ungated_path", type=str, required=True)
    parser.add_argument("--gated_path", type=str, required=True)
    parser.add_argument("--task_id", type=str, required=True)
    parser.add_argument("--output_summary", type=str, required=True)
    parser.add_argument("--output_cases", type=str, default=None)
    return parser.parse_args()


def normalize_gold(answer: Any) -> str:
    if isinstance(answer, int):
        return CHOICES[answer]
    return str(answer).strip().upper()


def extract_mc_answer(response: str) -> Optional[str]:
    response = normalize_response(response)
    for pattern in MULTILINGUAL_ANSWER_REGEXES:
        regex = MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(pattern)
        match = re.search(regex, response)
        if match:
            return normalize_extracted_answer(match.group(1)).upper()
    return None


def extract_prediction(result: Dict[str, Any]) -> Optional[str]:
    response = result["response"]
    if isinstance(response, list):
        extracted = [extract_prediction({**result, "response": r}) for r in response]
        return majority_vote(extracted)["winner"]
    return extract_mc_answer(str(response))


def score_result(result: Dict[str, Any]) -> Tuple[Optional[str], str, int]:
    pred = extract_prediction(result)
    gold = normalize_gold(result["answer"])
    if pred is None:
        return pred, gold, 0
    return pred, gold, int(pred == gold)


def first_present(*values: Any, default: Any = "") -> Any:
    for value in values:
        if value not in ("", None):
            return value
    return default


def pct(count: int, total: int) -> float:
    return count / total if total else 0.0


def add_rate(summary: Dict[str, Any], key: str, count: int, total: int) -> None:
    summary[key] = {
        "count": count,
        "total": total,
        "rate": pct(count, total),
        "percent": f"{pct(count, total):.2%}",
    }


def add_delta(summary: Dict[str, Any], key: str, numerator: int, total: int) -> None:
    rate = pct(numerator, total)
    summary[key] = {
        "count_delta": numerator,
        "total": total,
        "rate_delta": rate,
        "percentage_point_delta": f"{rate * 100:.2f}",
    }


def sign_test_p_value(wins: int, losses: int) -> Optional[float]:
    """Two-sided exact sign test under p=0.5, ignoring ties."""
    n = wins + losses
    if n == 0:
        return None
    tail = min(wins, losses)
    prob = sum(math.comb(n, k) for k in range(tail + 1)) / (2 ** n)
    return min(1.0, 2.0 * prob)


def add_sign_test(summary: Dict[str, Any], key: str, wins: int, losses: int) -> None:
    p_value = sign_test_p_value(wins, losses)
    summary[key] = {
        "wins": wins,
        "losses": losses,
        "ties_ignored": True,
        "p_value": p_value,
        "p_value_text": "NA" if p_value is None else f"{p_value:.6g}",
    }


def analyze(args: Namespace) -> Dict[str, Any]:
    clean = extract_jsonl(args.clean_path)
    ungated = extract_jsonl(args.ungated_path)
    gated = extract_jsonl(args.gated_path)

    if not (len(clean) == len(ungated) == len(gated)):
        raise ValueError(
            f"Length mismatch: clean={len(clean)}, ungated={len(ungated)}, "
            f"gated={len(gated)}"
        )

    cases: List[Dict[str, Any]] = []
    counters: Counter = Counter()
    transition_counter: Counter = Counter()

    for idx, (c, u, g) in enumerate(zip(clean, ungated, gated)):
        if c["query"] != u["query"] or c["query"] != g["query"]:
            raise ValueError(f"Query mismatch at index {idx}")

        c_pred, gold, c_score = score_result(c)
        u_pred, _, u_score = score_result(u)
        g_pred, _, g_score = score_result(g)

        counters["clean_correct"] += c_score
        counters["ungated_correct"] += u_score
        counters["gated_correct"] += g_score
        counters["ungated_score_ir"] += int(c_score != u_score)
        counters["gated_score_ir"] += int(c_score != g_score)
        counters["ungated_answer_flip"] += int(c_pred != u_pred)
        counters["gated_answer_flip"] += int(c_pred != g_pred)
        counters["ungated_fail_extract"] += int(u_pred is None)
        counters["gated_fail_extract"] += int(g_pred is None)
        counters["clean_fail_extract"] += int(c_pred is None)

        transition = f"c{c_score}_u{u_score}_g{g_score}"
        transition_counter[transition] += 1

        counters["repair_clean_c2w"] += int(c_score == 1 and u_score == 0 and g_score == 1)
        counters["damage_clean_correct"] += int(c_score == 1 and u_score == 1 and g_score == 0)
        counters["repair_overall"] += int(u_score == 0 and g_score == 1)
        counters["damage_overall"] += int(u_score == 1 and g_score == 0)
        counters["restore_clean_answer"] += int(c_pred != u_pred and c_pred == g_pred)
        counters["break_clean_answer"] += int(c_pred == u_pred and c_pred != g_pred)

        cases.append(
            {
                "index": idx,
                "id": first_present(c.get("id"), u.get("id"), g.get("id")),
                "source_index": c.get("source_index", idx),
                "subject": c.get("subject", ""),
                "task": c.get("task", ""),
                "speech_source": first_present(
                    c.get("speech_source"),
                    u.get("speech_source"),
                    g.get("speech_source"),
                    (c.get("metadata") or {}).get("speech_source"),
                    (u.get("metadata") or {}).get("speech_source"),
                    (g.get("metadata") or {}).get("speech_source"),
                ),
                "speech_index": first_present(
                    c.get("speech_index"),
                    u.get("speech_index"),
                    g.get("speech_index"),
                    (c.get("metadata") or {}).get("speech_index"),
                    (u.get("metadata") or {}).get("speech_index"),
                    (g.get("metadata") or {}).get("speech_index"),
                ),
                "query": c["query"],
                "choices": c.get("choices", ""),
                "gold": gold,
                "clean_pred": c_pred,
                "ungated_pred": u_pred,
                "gated_pred": g_pred,
                "clean_score": c_score,
                "ungated_score": u_score,
                "gated_score": g_score,
                "clean_gate_info": c.get("gate_info", {}),
                "ungated_gate_info": u.get("gate_info", {}),
                "gated_gate_info": g.get("gate_info", {}),
                "transition": transition,
                "repair_clean_c2w": c_score == 1 and u_score == 0 and g_score == 1,
                "damage_clean_correct": c_score == 1 and u_score == 1 and g_score == 0,
                "restore_clean_answer": c_pred != u_pred and c_pred == g_pred,
                "break_clean_answer": c_pred == u_pred and c_pred != g_pred,
            }
        )

    total = len(clean)
    summary: Dict[str, Any] = {
        "clean_path": args.clean_path,
        "ungated_path": args.ungated_path,
        "gated_path": args.gated_path,
        "task_id": args.task_id,
        "total": total,
        "transitions": dict(sorted(transition_counter.items())),
    }
    add_rate(summary, "clean_accuracy", counters["clean_correct"], total)
    add_rate(summary, "ungated_accuracy", counters["ungated_correct"], total)
    add_rate(summary, "gated_accuracy", counters["gated_correct"], total)
    add_rate(summary, "ungated_score_ir", counters["ungated_score_ir"], total)
    add_rate(summary, "gated_score_ir", counters["gated_score_ir"], total)
    add_rate(summary, "ungated_answer_flip_to_clean", counters["ungated_answer_flip"], total)
    add_rate(summary, "gated_answer_flip_to_clean", counters["gated_answer_flip"], total)
    add_rate(summary, "repair_clean_c2w", counters["repair_clean_c2w"], total)
    add_rate(summary, "damage_clean_correct", counters["damage_clean_correct"], total)
    add_rate(summary, "repair_overall", counters["repair_overall"], total)
    add_rate(summary, "damage_overall", counters["damage_overall"], total)
    add_rate(summary, "restore_clean_answer", counters["restore_clean_answer"], total)
    add_rate(summary, "break_clean_answer", counters["break_clean_answer"], total)
    add_rate(summary, "clean_fail_extract", counters["clean_fail_extract"], total)
    add_rate(summary, "ungated_fail_extract", counters["ungated_fail_extract"], total)
    add_rate(summary, "gated_fail_extract", counters["gated_fail_extract"], total)
    add_delta(
        summary,
        "accuracy_delta_gated_vs_ungated",
        counters["gated_correct"] - counters["ungated_correct"],
        total,
    )
    add_delta(
        summary,
        "accuracy_delta_gated_vs_clean",
        counters["gated_correct"] - counters["clean_correct"],
        total,
    )
    add_delta(
        summary,
        "score_ir_reduction_gated_vs_ungated",
        counters["ungated_score_ir"] - counters["gated_score_ir"],
        total,
    )
    add_delta(
        summary,
        "answer_flip_reduction_gated_vs_ungated",
        counters["ungated_answer_flip"] - counters["gated_answer_flip"],
        total,
    )
    summary["net_clean_c2w_repair"] = (
        counters["repair_clean_c2w"] - counters["damage_clean_correct"]
    )
    summary["net_overall_repair"] = counters["repair_overall"] - counters["damage_overall"]
    summary["net_answer_restoration"] = (
        counters["restore_clean_answer"] - counters["break_clean_answer"]
    )
    add_sign_test(
        summary,
        "sign_test_clean_c2w_repair_vs_damage",
        counters["repair_clean_c2w"],
        counters["damage_clean_correct"],
    )
    add_sign_test(
        summary,
        "sign_test_overall_repair_vs_damage",
        counters["repair_overall"],
        counters["damage_overall"],
    )
    add_sign_test(
        summary,
        "sign_test_answer_restore_vs_break",
        counters["restore_clean_answer"],
        counters["break_clean_answer"],
    )

    if args.output_cases:
        os.makedirs(os.path.dirname(args.output_cases), exist_ok=True)
        save_jsonl(args.output_cases, cases)

    return summary


def main(args: Namespace) -> None:
    summary = analyze(args)
    os.makedirs(os.path.dirname(args.output_summary), exist_ok=True)
    with open(args.output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main(parse_args())
