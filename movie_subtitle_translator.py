#!/usr/bin/env python
"""Transcribe an English movie and translate subtitles to a target language."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
import whisper
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


MODEL_ID = "google/translategemma-4b-it"


def format_srt_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours = total_ms // 3_600_000
    minutes = (total_ms % 3_600_000) // 60_000
    secs = (total_ms % 60_000) // 1000
    millis = total_ms % 1000
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def write_srt(path: Path, segments: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as srt_file:
        for index, segment in enumerate(segments, start=1):
            start = format_srt_timestamp(float(segment["start"]))
            end = format_srt_timestamp(float(segment["end"]))
            text = str(segment["text"]).strip()
            srt_file.write(f"{index}\n{start} --> {end}\n{text}\n\n")


def load_translator():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        device = None
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
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


def transcribe_to_english_segments(movie_path: Path, whisper_model_name: str) -> list[dict]:
    model = whisper.load_model(whisper_model_name)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    movie_path_input = args.movie_path or input("Enter movie path: ").strip()
    destination_language = args.target_language or input("Enter target language: ").strip()

    movie_path = Path(movie_path_input).expanduser().resolve()
    if not movie_path.exists():
        raise FileNotFoundError(f"Movie file does not exist: {movie_path}")

    if not destination_language:
        raise ValueError("Target language must not be empty.")

    english_srt_path, translated_srt_path = build_output_paths(movie_path, destination_language)

    print(f"[1/3] Transcribing {movie_path.name} with Whisper...")
    segments = transcribe_to_english_segments(movie_path, args.whisper_model)

    print(f"[2/3] Writing English subtitles to: {english_srt_path}")
    write_srt(english_srt_path, segments)

    print(f"[3/3] Translating subtitles to {destination_language} with {MODEL_ID}...")
    translator_pipeline = load_translator()
    translated_segments: list[dict] = []
    total = len(segments)
    for idx, segment in enumerate(segments, start=1):
        translated_text = translate_line(
            translator_pipeline,
            str(segment["text"]).strip(),
            destination_language,
        )
        translated_segments.append(
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": translated_text,
            }
        )
        if idx % 20 == 0 or idx == total:
            print(f"Translated {idx}/{total} subtitles...")

    write_srt(translated_srt_path, translated_segments)
    print(f"Done. English SRT: {english_srt_path}")
    print(f"Done. {destination_language} SRT: {translated_srt_path}")


if __name__ == "__main__":
    main()
