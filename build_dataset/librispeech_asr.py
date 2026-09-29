from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, List, Tuple

from datasets import Audio, Dataset, DatasetDict


SUPPORTED_AUDIO_EXTS = {".flac", ".wav"}


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Build a local LibriSpeech ASR dataset for audio-understanding "
            "side-effect tests. The output is a DatasetDict saved to disk."
        )
    )
    parser.add_argument(
        "--librispeech_path",
        type=str,
        required=True,
        help="Path to a LibriSpeech split folder, e.g. LibriSpeech/dev-clean.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Local directory to save the Hugging Face DatasetDict.",
    )
    parser.add_argument("--split_name", type=str, default="asr")
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap for a smaller side-effect pilot dataset.",
    )
    return parser.parse_args()


def read_transcripts(root: Path) -> Dict[str, str]:
    transcripts: Dict[str, str] = {}
    for path in sorted(root.rglob("*.trans.txt")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                utt_id, text = line.split(" ", 1)
                transcripts[utt_id] = text.strip()
    return transcripts


def collect_audio(root: Path) -> List[Tuple[str, Path]]:
    examples: List[Tuple[str, Path]] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTS:
            examples.append((path.stem, path))
    return examples


def main(args: Namespace) -> None:
    root = Path(args.librispeech_path)
    transcripts = read_transcripts(root)
    audio_files = collect_audio(root)
    if not transcripts:
        raise RuntimeError(f"No LibriSpeech transcript files found under {root}")
    if not audio_files:
        raise RuntimeError(f"No audio files found under {root}")

    rows = []
    for utt_id, audio_path in audio_files:
        transcript = transcripts.get(utt_id)
        if transcript is None:
            continue
        rows.append(
            {
                "id": utt_id,
                "audio": str(audio_path.resolve()),
                "query": (
                    "Transcribe the speech in the audio. The last line of your "
                    "response should be of the following format: 'Answer: "
                    "<transcription>' (without quotes)."
                ),
                "answer": transcript,
            }
        )

    if args.max_samples is not None:
        rows = rows[: args.max_samples]
    if not rows:
        raise RuntimeError("No matched audio/transcript examples were found.")

    dataset = Dataset.from_list(rows).cast_column("audio", Audio())
    dataset_dict = DatasetDict({args.split_name: dataset})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))

    print(dataset_dict)
    print(dataset_dict[args.split_name].features)
    print(f"Saved LibriSpeech ASR side-effect dataset to {output_dir.resolve()}")


if __name__ == "__main__":
    main(parse_args())
