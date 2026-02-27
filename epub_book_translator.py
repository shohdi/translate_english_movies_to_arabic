#!/usr/bin/env python
"""Translate EPUB books paragraph-by-paragraph with resume support."""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
from typing import Iterable

import torch
from bs4 import BeautifulSoup
from ebooklib import ITEM_DOCUMENT, epub
from huggingface_hub import snapshot_download
from transformers import pipeline


MODEL_ID = "google/translategemma-4b-it"
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent / ".local_models"
ORIGINAL_MARKER = "\n[Original]\n"
LANGUAGE_CODE_MAP = {
    "arabic": "ar-EG",
    "english": "en",
    "french": "fr-FR",
    "spanish": "es-ES",
    "german": "de-DE",
    "italian": "it-IT",
    "portuguese": "pt-BR",
    "russian": "ru-RU",
    "turkish": "tr-TR",
    "urdu": "ur",
    "hindi": "hi-IN",
    "japanese": "ja-JP",
    "korean": "ko-KR",
    "chinese": "zh-CN",
}


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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32

    try:
        return pipeline(
            "text-generation",
            model=str(local_model_path),
            device=device,
            torch_dtype=dtype,
            model_kwargs={"local_files_only": True},
        )
    except TypeError:
        return pipeline(
            "text-generation",
            model=str(local_model_path),
            device=device,
            dtype=dtype,
            model_kwargs={"local_files_only": True},
        )


def normalize_lang_code(language: str) -> str:
    lang = language.strip().lower()
    if lang in LANGUAGE_CODE_MAP:
        return LANGUAGE_CODE_MAP[lang]
    if "-" in language and len(language) >= 4:
        return language
    if len(lang) == 2 and lang.isalpha():
        return lang
    return lang[:2] if len(lang) > 2 else lang


def extract_translated_text(pipe_output) -> str:
    if not pipe_output:
        return ""
    first_item = pipe_output[0]
    generated = first_item.get("generated_text")
    if isinstance(generated, list) and generated:
        last_item = generated[-1]
        if isinstance(last_item, dict):
            content = last_item.get("content", "")
            if isinstance(content, str):
                return content.strip()
    if isinstance(generated, str):
        return generated.strip()
    return ""


def split_long_text(text: str, max_words: int = 800) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text]
    chunks: list[str] = []
    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i : i + max_words]))
    return chunks


def translate_chunk(
    translator_pipe,
    text: str,
    source_lang_code: str,
    target_lang_code: str,
) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source_lang_code,
                    "target_lang_code": target_lang_code,
                    "text": text,
                }
            ],
        }
    ]
    output = translator_pipe(messages, max_new_tokens=1000, do_sample=False)
    return extract_translated_text(output).replace("\n", " ").strip(" \"'")


def build_context_block(previous_translations: list[str], context_window: int) -> str:
    if context_window <= 0 or not previous_translations:
        return ""
    recent = previous_translations[-context_window:]
    lines = [f"- {line}" for line in recent if line.strip()]
    if not lines:
        return ""
    return "Recent translated context:\n" + "\n".join(lines)


def build_glossary_block(glossary: dict) -> str:
    if not glossary:
        return ""
    lines: list[str] = []
    for name, meta in glossary.items():
        if isinstance(meta, dict):
            gender = str(meta.get("gender", "")).strip()
            target_form = str(meta.get("target_form", "")).strip()
            note = str(meta.get("note", "")).strip()
            parts = [p for p in [f"name={name}", f"gender={gender}", f"target_form={target_form}", f"note={note}"] if p and not p.endswith("=")]
            if parts:
                lines.append("- " + ", ".join(parts))
        else:
            lines.append(f"- name={name}, note={meta}")
    if not lines:
        return ""
    return "Character glossary (keep these consistent):\n" + "\n".join(lines)


def build_chapter_note(chapter_context: dict, file_name: str) -> str:
    if not chapter_context:
        return ""
    note = chapter_context.get(file_name) or chapter_context.get("__default__")
    if not note:
        return ""
    return f"Chapter context note:\n{note}"


def translate_paragraph_with_context(
    translator_pipe,
    paragraph_text: str,
    source_lang_code: str,
    target_lang_code: str,
    previous_translations: list[str],
    context_window: int,
    glossary: dict,
    chapter_note: str,
) -> str:
    context_block = build_context_block(previous_translations, context_window)
    glossary_block = build_glossary_block(glossary)
    sections = [
        "Translate ONLY the final paragraph below.",
        "Keep meaning accurate and natural.",
        "Preserve gender/pronoun consistency using context and glossary.",
        "Return only translated text without explanations.",
    ]
    if chapter_note:
        sections.append(chapter_note)
    if glossary_block:
        sections.append(glossary_block)
    if context_block:
        sections.append(context_block)
    sections.append(f"Paragraph to translate:\n{paragraph_text}")
    full_text = "\n\n".join(sections)
    return translate_chunk(
        translator_pipe,
        full_text,
        source_lang_code,
        target_lang_code,
    )


def translate_paragraph(
    translator_pipe,
    paragraph_text: str,
    source_lang_code: str,
    target_lang_code: str,
) -> str:
    chunks = split_long_text(paragraph_text, max_words=800)
    translated_chunks: list[str] = []
    for chunk in chunks:
        translated = translate_chunk(
            translator_pipe,
            chunk,
            source_lang_code,
            target_lang_code,
        )
        translated_chunks.append(translated if translated else chunk)
    return " ".join(translated_chunks).strip()


def refine_translation_consistency(
    translator_pipe,
    source_text: str,
    translated_text: str,
    source_lang_code: str,
    target_lang_code: str,
    previous_translations: list[str],
    context_window: int,
    glossary: dict,
    chapter_note: str,
) -> str:
    context_block = build_context_block(previous_translations, context_window)
    glossary_block = build_glossary_block(glossary)
    parts = [
        "Improve the translated paragraph only if needed.",
        "Focus on pronouns, gender agreement, and character consistency.",
        "Do not add or remove information.",
        "Return only the revised translated paragraph.",
        f"Source paragraph:\n{source_text}",
        f"Current translation:\n{translated_text}",
    ]
    if chapter_note:
        parts.append(chapter_note)
    if glossary_block:
        parts.append(glossary_block)
    if context_block:
        parts.append(context_block)
    review_prompt = "\n\n".join(parts)
    reviewed = translate_chunk(
        translator_pipe,
        review_prompt,
        source_lang_code,
        target_lang_code,
    )
    return reviewed if reviewed else translated_text


def parse_progress(progress_path: Path) -> dict[str, str]:
    if not progress_path.exists():
        return {}
    result: dict[str, str] = {}
    with progress_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.rstrip("\n")
            if not line or "\t" not in line:
                continue
            key, encoded = line.split("\t", 1)
            try:
                decoded = base64.b64decode(encoded.encode("ascii")).decode("utf-8")
            except Exception:
                continue
            result[key] = decoded
    return result


def append_progress(progress_path: Path, key: str, translated_text: str) -> None:
    encoded = base64.b64encode(translated_text.encode("utf-8")).decode("ascii")
    with progress_path.open("a", encoding="utf-8") as file:
        file.write(f"{key}\t{encoded}\n")
        file.flush()


def iter_document_items(book: epub.EpubBook) -> Iterable[epub.EpubHtml]:
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        yield item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Translate EPUB book paragraphs using google/translategemma-4b-it "
            "with resumable progress."
        )
    )
    parser.add_argument("--epub-path", help="Path to input EPUB file.")
    parser.add_argument(
        "--source-language",
        default="en",
        help="Source language (name or code, default: en).",
    )
    parser.add_argument(
        "--target-language",
        help="Target language (name or code, example: Arabic or ar-EG).",
    )
    parser.add_argument(
        "--models-dir",
        default=str(DEFAULT_MODELS_DIR),
        help="Local model storage path (default: .local_models beside script).",
    )
    parser.add_argument(
        "--output-path",
        help="Output translated EPUB path (default: same folder with target suffix).",
    )
    parser.add_argument(
        "--progress-path",
        help="Progress text file path for resume (default: beside EPUB output).",
    )
    parser.add_argument(
        "--prepare-models",
        action="store_true",
        help="Download required model to local storage and exit.",
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=4,
        help="Number of previous translated paragraphs to include as context (default: 4).",
    )
    parser.add_argument(
        "--character-glossary",
        help="Path to JSON glossary for names/gender/target forms.",
    )
    parser.add_argument(
        "--chapter-context",
        help="Path to JSON map of chapter file name -> context note.",
    )
    parser.add_argument(
        "--consistency-pass",
        action="store_true",
        help="Run a second pass per paragraph to improve gender/pronoun consistency.",
    )
    parser.add_argument(
        "--append-original",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Append the original paragraph after the translated paragraph "
            "(useful for language learning). Default: enabled. Use --no-append-original to disable."
        ),
    )
    return parser.parse_args()


def load_json_dict(path_str: str | None) -> dict:
    if not path_str:
        return {}
    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise ValueError(f"JSON file must contain an object/map at top level: {path}")
    return data


def build_output_paragraph(translated_text: str, original_text: str, append_original: bool) -> str:
    if not append_original:
        return translated_text
    return f"{translated_text}{ORIGINAL_MARKER}{original_text}"


def set_paragraph_content(soup: BeautifulSoup, paragraph, text: str) -> None:
    paragraph.clear()
    if ORIGINAL_MARKER not in text:
        paragraph.append(text)
        return

    translated_text, original_text = text.split(ORIGINAL_MARKER, 1)
    paragraph.append(translated_text.strip())
    paragraph.append(soup.new_tag("br"))
    paragraph.append(soup.new_tag("br"))
    label = soup.new_tag("span")
    label.string = "[Original]"
    paragraph.append(label)
    paragraph.append(soup.new_tag("br"))
    paragraph.append(original_text.strip())


def main() -> None:
    args = parse_args()
    models_dir = Path(args.models_dir).expanduser().resolve()
    local_translator_path = ensure_local_translator_path(models_dir)

    if args.prepare_models:
        print(f"[prep] Model is ready in: {local_translator_path}")
        return

    epub_path_input = args.epub_path or input("Enter EPUB path: ").strip()
    target_language = args.target_language or input("Enter target language: ").strip()
    source_language = args.source_language.strip()

    epub_path = Path(epub_path_input).expanduser().resolve()
    if not epub_path.exists():
        raise FileNotFoundError(f"EPUB file does not exist: {epub_path}")
    if not target_language:
        raise ValueError("Target language must not be empty.")

    target_lang_code = normalize_lang_code(target_language)
    source_lang_code = normalize_lang_code(source_language)
    glossary = load_json_dict(args.character_glossary)
    chapter_context = load_json_dict(args.chapter_context)

    output_path = (
        Path(args.output_path).expanduser().resolve()
        if args.output_path
        else epub_path.with_name(f"{epub_path.stem}.{target_language.strip().lower().replace(' ', '_')}.epub")
    )
    progress_path = (
        Path(args.progress_path).expanduser().resolve()
        if args.progress_path
        else output_path.with_suffix(".progress.txt")
    )

    print(f"[1/3] Loading EPUB: {epub_path}")
    book = epub.read_epub(str(epub_path))

    print(f"[2/3] Loading translator model: {MODEL_ID}")
    translator_pipe = load_translator(local_translator_path)

    print(f"[3/3] Translating paragraphs to {target_language} ({target_lang_code})")
    progress = parse_progress(progress_path)
    translated_count = 0
    total_count = 0
    doc_index = 0

    for item in iter_document_items(book):
        doc_index += 1
        soup = BeautifulSoup(item.get_content(), "lxml")
        paragraph_index = 0
        doc_previous_translations: list[str] = []
        chapter_note = build_chapter_note(chapter_context, item.file_name)
        for paragraph in soup.find_all("p"):
            original_text = paragraph.get_text(" ", strip=True)
            if not original_text:
                continue
            paragraph_index += 1
            total_count += 1
            key = f"{item.file_name}::d{doc_index}_p{paragraph_index}"
            if key in progress:
                set_paragraph_content(soup, paragraph, progress[key])
                doc_previous_translations.append(progress[key])
                continue

            print(f"[{total_count}] SRC: {original_text[:140]}")
            translated_text = translate_paragraph_with_context(
                translator_pipe,
                original_text,
                source_lang_code,
                target_lang_code,
                doc_previous_translations,
                args.context_window,
                glossary,
                chapter_note,
            )
            if args.consistency_pass and translated_text:
                translated_text = refine_translation_consistency(
                    translator_pipe,
                    original_text,
                    translated_text,
                    source_lang_code,
                    target_lang_code,
                    doc_previous_translations,
                    args.context_window,
                    glossary,
                    chapter_note,
                )
            if not translated_text:
                translated_text = original_text
            output_text = build_output_paragraph(
                translated_text,
                original_text,
                args.append_original,
            )
            print(f"[{total_count}] DST: {translated_text[:140]}")
            set_paragraph_content(soup, paragraph, output_text)
            append_progress(progress_path, key, output_text)
            progress[key] = output_text
            doc_previous_translations.append(translated_text)
            translated_count += 1
            print(f"[{total_count}] Saved progress to: {progress_path}")

        item.set_content(str(soup).encode("utf-8"))

    epub.write_epub(str(output_path), book)
    print(f"Done. Output EPUB: {output_path}")
    print(f"Progress file: {progress_path}")
    print(f"Newly translated paragraphs this run: {translated_count}")
    print(f"Total paragraphs in book: {total_count}")


if __name__ == "__main__":
    main()
