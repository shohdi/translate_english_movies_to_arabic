#!/usr/bin/env python
"""Transcribe an English movie and translate subtitles to a target language."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
import whisper
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


MODEL_ID = "google/translategemma-4b-it"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / ".local_models"


def format_srt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def parse_srt_timestamp(value: str) -> float:
    time_part = value.strip()
    hh_mm_ss, ms = time_part.split(",")
    hours, minutes, seconds = hh_mm_ss.split(":")
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + (int(ms) / 1000.0)
    )


def write_srt(path: Path, segments: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as srt_file:
        for index, segment in enumerate(segments, start=1):
            start = format_srt_timestamp(float(segment["start"]))
            end = format_srt_timestamp(float(segment["end"]))
            text = str(segment["text"]).strip()
            srt_file.write(f"{index}\n{start} --> {end}\n{text}\n\n")


def append_srt_entry(srt_file, index: int, segment: dict, text: str) -> None:
    start = format_srt_timestamp(float(segment["start"]))
    end = format_srt_timestamp(float(segment["end"]))
    srt_file.write(f"{index}\n{start} --> {end}\n{text.strip()}\n\n")
    srt_file.flush()


def get_completed_subtitle_count(srt_path: Path) -> int:
    if not srt_path.exists():
        return 0

    content = srt_path.read_text(encoding="utf-8").strip()
    if not content:
        return 0

    blocks = [block.strip() for block in content.split("\n\n") if block.strip()]
    expected_index = 1
    for block in blocks:
        lines = [line for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            break
        if lines[0].strip() != str(expected_index):
            break
        if "-->" not in lines[1]:
            break
        expected_index += 1
    return expected_index - 1


def load_segments_from_srt(srt_path: Path) -> list[dict]:
    content = srt_path.read_text(encoding="utf-8").strip()
    if not content:
        return []

    segments: list[dict] = []
    blocks = [block.strip() for block in content.split("\n\n") if block.strip()]
    for block in blocks:
        lines = [line.rstrip() for line in block.splitlines() if line.strip()]
        if len(lines) < 3:
            continue
        if "-->" not in lines[1]:
            continue

        start_raw, end_raw = [part.strip() for part in lines[1].split("-->")]
        text = " ".join(lines[2:]).strip()
        segments.append(
            {
                "start": parse_srt_timestamp(start_raw),
                "end": parse_srt_timestamp(end_raw),
                "text": text,
            }
        )
    return segments


def ensure_local_translator_path(models_dir: Path) -> Path:
    local_model_dir = models_dir / "huggingface" / "translategemma-4b-it"
    local_model_dir.mkdir(parents=True, exist_ok=True)
    if not (local_model_dir / "config.json").exists():
        print("[prep] Downloading TranslateGemma model locally (first run only)...")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=str(local_model_dir),
        )
    return local_model_dir


def load_translator(local_model_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        str(local_model_path),
        local_files_only=True,
    )
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            str(local_model_path),
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=True,
        )
        device = None
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(local_model_path),
            torch_dtype=torch.float32,
            local_files_only=True,
        )
        device = -1

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
    )


def translate_line(
    translator_pipeline,
    english_text: str,
    destination_language: str,
) -> str:
    prompt = (
        f"Translate the following subtitle from English to {destination_language}. "
        "Keep it natural and concise for subtitles. Return only the translation.\n\n"
        f"English: {english_text}\n"
        f"{destination_language}:"
    )
    output = translator_pipeline(
        prompt,
        max_new_tokens=160,
        do_sample=False,
        temperature=0.0,
        return_full_text=False,
    )[0]["generated_text"].strip()
    return output.replace("\n", " ").strip(" \"'")


def transcribe_to_english_segments(
    movie_path: Path,
    whisper_model_name: str,
    whisper_cache_dir: Path,
) -> list[dict]:
    whisper_cache_dir.mkdir(parents=True, exist_ok=True)
    model = whisper.load_model(whisper_model_name, download_root=str(whisper_cache_dir))
    result = model.transcribe(
        str(movie_path),
        language="en",
        task="transcribe",
        verbose=False,
    )
    return result["segments"]


def build_output_paths(movie_path: Path, destination_language: str) -> tuple[Path, Path]:
    stem = movie_path.stem
    parent = movie_path.parent
    english_srt = parent / f"{stem}.en.srt"
    safe_lang = destination_language.strip().lower().replace(" ", "_")
    translated_srt = parent / f"{stem}.{safe_lang}.srt"
    return english_srt, translated_srt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create English subtitles with Whisper, then translate them using "
            "google/translategemma-4b-it while keeping original timestamps."
        )
    )
    parser.add_argument(
        "--movie-path",
        help="Path to the input movie file (mp4/mkv/etc).",
    )
    parser.add_argument(
        "--target-language",
        help="Destination language for subtitle translation (example: Arabic).",
    )
    parser.add_argument(
        "--whisper-model",
        default="base",
        help="Whisper model size/name to use (default: base).",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help=(
            "Local model storage path. Models are downloaded here once, then reused "
            "offline (default: .local_models beside script)."
        ),
    )
    parser.add_argument(
        "--prepare-models",
        action="store_true",
        help="Download required models to local storage and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir).expanduser().resolve()
    whisper_cache_dir = models_dir / "whisper"
    local_translator_path = ensure_local_translator_path(models_dir)

    if args.prepare_models:
        print(f"[prep] Downloading/loading Whisper model '{args.whisper_model}' into {whisper_cache_dir}")
        whisper_cache_dir.mkdir(parents=True, exist_ok=True)
        whisper.load_model(args.whisper_model, download_root=str(whisper_cache_dir))
        print(f"[prep] Models are ready in: {models_dir}")
        return

    movie_path_input = args.movie_path or input("Enter movie path: ").strip()
    destination_language = args.target_language or input("Enter target language: ").strip()

    movie_path = Path(movie_path_input).expanduser().resolve()
    if not movie_path.exists():
        raise FileNotFoundError(f"Movie file does not exist: {movie_path}")

    if not destination_language:
        raise ValueError("Target language must not be empty.")

    english_srt_path, translated_srt_path = build_output_paths(movie_path, destination_language)

    if english_srt_path.exists():
        print(f"[1/3] Reusing existing English subtitles: {english_srt_path}")
        segments = load_segments_from_srt(english_srt_path)
        if not segments:
            raise ValueError(
                f"English SRT exists but has no valid subtitle entries: {english_srt_path}"
            )
    else:
        print(f"[1/3] Transcribing {movie_path.name} with Whisper...")
        segments = transcribe_to_english_segments(movie_path, args.whisper_model, whisper_cache_dir)
        print(f"[2/3] Writing English subtitles to: {english_srt_path}")
        write_srt(english_srt_path, segments)

    print(f"[3/3] Translating subtitles to {destination_language} with {MODEL_ID}...")
    translator_pipeline = load_translator(local_translator_path)
    total = len(segments)
    completed_count = get_completed_subtitle_count(translated_srt_path)
    if completed_count > total:
        raise ValueError(
            f"Existing translated file has {completed_count} subtitles but transcription has {total}. "
            "Please remove the translated .srt or use another output name."
        )
    if completed_count > 0:
        print(f"Resuming from subtitle {completed_count + 1}/{total} using existing file: {translated_srt_path}")

    write_mode = "a" if completed_count > 0 else "w"
    with translated_srt_path.open(write_mode, encoding="utf-8") as target_srt_file:
        for idx, segment in enumerate(segments, start=1):
            if idx <= completed_count:
                continue

            english_text = str(segment["text"]).strip()
            print(f"[{idx}/{total}] EN: {english_text}")

            translated_text = translate_line(
                translator_pipeline,
                english_text,
                destination_language,
            )
            print(f"[{idx}/{total}] {destination_language}: {translated_text}")

            append_srt_entry(target_srt_file, idx, segment, translated_text)
            print(f"[{idx}/{total}] Saved to {translated_srt_path}")

    print(f"Done. English SRT: {english_srt_path}")
    print(f"Done. {destination_language} SRT: {translated_srt_path}")


if __name__ == "__main__":
    main()
