import re
import string
from argparse import ArgumentParser, Namespace
from typing import List

from utils.common import extract_jsonl


def parse_args() -> Namespace:
    parser = ArgumentParser(description="Evaluate ASR-style JSONL outputs with WER.")
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument(
        "--extract_answer_line",
        action="store_true",
        help="If set, extract text after the first 'Answer:' marker before scoring.",
    )
    return parser.parse_args()


def normalize_text(text: str, extract_answer_line: bool = False) -> str:
    text = text.lower()
    if extract_answer_line and "answer:" in text:
        text = text.split("answer:", 1)[1]
    text = re.sub(r"\s+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def edit_distance(ref: List[str], hyp: List[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, ref_word in enumerate(ref, start=1):
        curr = [i] + [0] * len(hyp)
        for j, hyp_word in enumerate(hyp, start=1):
            cost = 0 if ref_word == hyp_word else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    return prev[-1]


def main(args: Namespace) -> None:
    rows = extract_jsonl(args.input_path)
    total_edits = 0
    total_ref_words = 0
    exact = 0
    empty_pred = 0

    for row in rows:
        ref = normalize_text(str(row["answer"]))
        hyp_source = str(row["response"])
        hyp = normalize_text(
            hyp_source,
            extract_answer_line=args.extract_answer_line,
        )
        ref_words = ref.split()
        hyp_words = hyp.split()
        total_edits += edit_distance(ref_words, hyp_words)
        total_ref_words += len(ref_words)
        exact += int(ref == hyp)
        empty_pred += int(not hyp)

    wer = total_edits / total_ref_words if total_ref_words else 0.0
    total = len(rows)
    print(f"WER: {wer:.2%} ({total_edits}/{total_ref_words})")
    print(f"Exact Match: {exact / total:.2%} ({exact}/{total})")
    print(f"Empty Prediction: {empty_pred / total:.2%} ({empty_pred}/{total})")


if __name__ == "__main__":
    main(parse_args())
