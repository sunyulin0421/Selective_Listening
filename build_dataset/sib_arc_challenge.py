import random
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import List

import librosa
import numpy as np
import soundfile as sf
from datasets import Audio, Dataset, DatasetDict, load_dataset
from tqdm import tqdm


SUPPORTED_AUDIO_EXTS = {".flac", ".wav", ".mp3", ".ogg", ".m4a"}


def parse_args() -> Namespace:
    parser = ArgumentParser(
        description=(
            "Build ARC-Challenge Speech Interference Benchmark (SIB) by "
            "pairing each ARC test question with a random speech file."
        )
    )
    parser.add_argument(
        "--speech_path",
        type=str,
        required=True,
        help="Folder containing speech audio files, e.g. LibriSpeech dev-clean.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Local directory to save the Hugging Face DatasetDict.",
    )
    parser.add_argument("--split_name", type=str, default="speech_random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sampling_rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=5.0)
    parser.add_argument(
        "--rms",
        type=float,
        default=None,
        help="Optional RMS normalization target. Omit to keep natural loudness.",
    )
    parser.add_argument(
        "--pad_mode",
        choices=["zero", "repeat"],
        default="zero",
        help="How to pad speech shorter than --duration.",
    )
    parser.add_argument(
        "--overwrite_audio",
        action="store_true",
        help="Regenerate processed wavs even if they already exist.",
    )
    return parser.parse_args()


def collect_audio_files(speech_path: str) -> List[Path]:
    root = Path(speech_path)
    files = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTS
    ]
    return sorted(files)


def fit_duration(
    audio: np.ndarray,
    target_len: int,
    rng: random.Random,
    pad_mode: str,
) -> np.ndarray:
    if len(audio) >= target_len:
        max_start = len(audio) - target_len
        start = rng.randint(0, max_start) if max_start > 0 else 0
        return audio[start:start + target_len]

    if pad_mode == "repeat" and len(audio) > 0:
        repeats = int(np.ceil(target_len / len(audio)))
        audio = np.tile(audio, repeats)[:target_len]
        return audio

    padded = np.zeros(target_len, dtype=np.float32)
    padded[:len(audio)] = audio
    return padded


def normalize_rms(audio: np.ndarray, target_rms: float) -> np.ndarray:
    current = float(np.sqrt(np.mean(np.square(audio))) + 1e-8)
    audio = audio * (target_rms / current)
    return np.clip(audio, -1.0, 1.0)


def process_speech_file(
    input_path: Path,
    output_path: Path,
    sampling_rate: int,
    duration: float,
    rms: float,
    pad_mode: str,
    rng: random.Random,
    overwrite: bool,
) -> str:
    if output_path.exists() and not overwrite:
        return str(output_path.resolve())

    audio, _sr = librosa.load(input_path, sr=sampling_rate, mono=True)
    audio = audio.astype(np.float32)
    target_len = int(round(sampling_rate * duration))
    audio = fit_duration(audio, target_len, rng, pad_mode)
    if rms is not None:
        audio = normalize_rms(audio, rms)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(output_path, audio, sampling_rate)
    return str(output_path.resolve())


def main(args: Namespace) -> None:
    rng = random.Random(args.seed)
    speech_files = collect_audio_files(args.speech_path)
    if not speech_files:
        raise RuntimeError(f"No speech files found under {args.speech_path}")

    arc = load_dataset("allenai/ai2_arc", "ARC-Challenge")["test"]
    dataset_size = len(arc)
    if len(speech_files) < dataset_size:
        raise RuntimeError(
            f"Need at least {dataset_size} speech files for random pairing without "
            f"replacement, but found {len(speech_files)} under {args.speech_path}."
        )

    selected_files = rng.sample(speech_files, dataset_size)
    audio_dir = Path(args.output_dir) / "audio" / args.split_name
    processed_audio = []
    speech_sources = []
    speech_indices = []
    for idx, input_path in enumerate(tqdm(selected_files, desc="Processing speech")):
        output_path = audio_dir / f"speech_random_{idx:05d}.wav"
        processed_audio.append(
            process_speech_file(
                input_path=input_path,
                output_path=output_path,
                sampling_rate=args.sampling_rate,
                duration=args.duration,
                rms=args.rms,
                pad_mode=args.pad_mode,
                rng=rng,
                overwrite=args.overwrite_audio,
            )
        )
        speech_sources.append(str(input_path))
        speech_indices.append(speech_files.index(input_path))

    sib = Dataset.from_dict(
        {
            "id": arc["id"],
            "audio": processed_audio,
            "question": arc["question"],
            "choices": arc["choices"],
            "answer": arc["answerKey"],
            "speech_source": speech_sources,
            "speech_index": speech_indices,
        }
    ).cast_column("audio", Audio())

    dataset_dict = DatasetDict({args.split_name: sib})
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dict.save_to_disk(str(output_dir))

    print(dataset_dict)
    print(dataset_dict[args.split_name].features)
    print(f"Saved SIB ARC dataset to {output_dir.resolve()}")


if __name__ == "__main__":
    main(parse_args())
