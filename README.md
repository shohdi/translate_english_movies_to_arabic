# Translate English Movies to Any Language Subtitles 🎬🌍

This project creates subtitles in 2 steps:

1. Whisper transcribes your movie/audio into English `.srt` subtitles.
2. `google/translategemma-4b-it` translates those subtitles into your target language (Arabic, French, Japanese... your call 😎).

Output files are created in the same folder as your movie, while keeping subtitle timings from Whisper.
Models are cached locally on first use, then loaded from local files only (offline-ready) 📴

## Features 🚀

- Transcribe English speech to subtitles using Whisper.
- Translate subtitle text with `google/translategemma-4b-it`.
- Preserve exact subtitle timing from the English transcription.
- Download models once into a local cache and reuse them offline.
- Interactive mode (asks for input path and language) or CLI arguments.

## Requirements 🧰

- Python 3.10+
- Conda
- `ffmpeg` installed and available in your `PATH` (needed by Whisper)
- Enough RAM/VRAM for model loading (especially for `translategemma-4b-it`)

## Offline-First Model Cache 🗃️

The app stores models in a local folder:

- `.local_models/whisper/...`
- `.local_models/huggingface/translategemma-4b-it/...`

First run downloads what is missing. After that, loading uses local files only, so it works without internet.

## Environment Setup (Conda) 🐍

```powershell
conda create -n modelEnv python=3.11 -y
conda activate modelEnv
pip install -r requirements.txt
```

Optional but useful for faster inference on NVIDIA GPUs:

```powershell
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
```

## Install FFmpeg (if missing) 🛠️

Windows (winget):

```powershell
winget install "Gyan.FFmpeg"
```

Then restart terminal and verify:

```powershell
ffmpeg -version
```

## Usage 🎯

Activate the environment first:

```powershell
conda activate modelEnv
```

### Optional: Pre-download Models Once (Recommended) 📦

```powershell
python movie_subtitle_translator.py --prepare-models --whisper-model base
```

Now you can disconnect internet and run normally.

### Option 1: Interactive Mode

```powershell
python movie_subtitle_translator.py
```

It will ask:

- Movie path
- Target language (example: `Arabic`)

### Option 2: CLI Arguments

```powershell
python movie_subtitle_translator.py --movie-path "C:\movies\my_film.mp4" --target-language "Arabic"
```

You can also choose Whisper size:

```powershell
python movie_subtitle_translator.py --movie-path "C:\movies\my_film.mp4" --target-language "Arabic" --whisper-model medium
```

Custom local model cache directory:

```powershell
python movie_subtitle_translator.py --movie-path "C:\movies\my_film.mp4" --target-language "Arabic" --models-dir "D:\models_cache"
```

## Output Files 📁

If input file is:

`C:\movies\my_film.mp4`

You will get:

- `C:\movies\my_film.en.srt` (English subtitles from Whisper)
- `C:\movies\my_film.arabic.srt` (Translated subtitles, same timings)

## Notes 🤓

- The script assumes spoken language in media is English for transcription.
- Translation is done subtitle-by-subtitle to keep timings aligned.
- Larger Whisper models improve quality but are slower.
- `.local_models/` is git-ignored, so model weights are never pushed to your repo.

## Quick Example

If target language is `Arabic`, the script creates Arabic subtitle text while preserving start/end times from English SRT. So your subtitles stay synced and your popcorn stays safe 🍿.
