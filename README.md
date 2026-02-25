# Translate English Movies to Any Language Subtitles 🎬🌍

## English Guide 🇺🇸

This project creates subtitles in 2 steps:

1. Whisper transcribes your movie/audio into English `.srt` subtitles.
2. `google/translategemma-4b-it` translates those subtitles into your target language (Arabic, French, Japanese... your call 😎).

Output files are created in the same folder as your movie, while keeping subtitle timings from Whisper.
Models are cached locally on first use, then loaded from local files only (offline-ready) 📴

### Features 🚀

- Transcribe English speech to subtitles using Whisper.
- Translate subtitle text with `google/translategemma-4b-it`.
- Preserve exact subtitle timing from the English transcription.
- Download models once into a local cache and reuse them offline.
- Interactive mode (asks for input path and language) or CLI arguments.
- Logs progress for every subtitle (shows current English + translated line).
- Writes translated `.srt` on the fly and resumes from last completed subtitle if interrupted.

### Requirements 🧰

- Python 3.10+
- Conda
- `ffmpeg` installed and available in your `PATH` (needed by Whisper)
- Enough RAM/VRAM for model loading (especially for `translategemma-4b-it`)

### Offline-First Model Cache 🗃️

The app stores models in a local folder:

- `.local_models/whisper/...`
- `.local_models/huggingface/translategemma-4b-it/...`

First run downloads what is missing. After that, loading uses local files only, so it works without internet.

### Environment Setup (Conda) 🐍

```powershell
conda create -n modelEnv python=3.11 -y
conda activate modelEnv
pip install -r requirements.txt
```

Optional but useful for faster inference on NVIDIA GPUs:

```powershell
pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu124
```

### Install FFmpeg (if missing) 🛠️

Windows (winget):

```powershell
winget install "Gyan.FFmpeg"
```

Then restart terminal and verify:

```powershell
ffmpeg -version
```

### Usage 🎯

Activate environment:

```powershell
conda activate modelEnv
```

Optional: pre-download models once:

```powershell
python movie_subtitle_translator.py --prepare-models --whisper-model base
```

Now you can disconnect internet and run normally.

Option 1 (interactive):

```powershell
python movie_subtitle_translator.py
```

Option 2 (arguments):

```powershell
python movie_subtitle_translator.py --movie-path "C:\movies\my_film.mp4" --target-language "Arabic"
```

Choose Whisper size:

```powershell
python movie_subtitle_translator.py --movie-path "C:\movies\my_film.mp4" --target-language "Arabic" --whisper-model medium
```

Custom local model cache directory:

```powershell
python movie_subtitle_translator.py --movie-path "C:\movies\my_film.mp4" --target-language "Arabic" --models-dir "D:\models_cache"
```

### Output Files 📁

If input file is:
`C:\movies\my_film.mp4`

You get:

- `C:\movies\my_film.en.srt` (English subtitles from Whisper)
- `C:\movies\my_film.arabic.srt` (Translated subtitles, same timings)

### Notes 🤓

- The script assumes spoken language in media is English for transcription.
- Translation is done subtitle-by-subtitle to keep timings aligned.
- Larger Whisper models improve quality but are slower.
- `.local_models/` is git-ignored, so model weights are never pushed to your repo.
- If translation stops mid-run, re-run the same command and it continues from the last saved subtitle.
- On resume, if `movie.en.srt` already exists, the script reuses it and skips Whisper to keep subtitle order stable.

## الدليل العربي 🇸🇦

هذا المشروع ينشئ الترجمة على مرحلتين:

1. يقوم Whisper بتفريغ الصوت إلى ملف ترجمة إنجليزي بصيغة `.srt`.
2. يقوم `google/translategemma-4b-it` بترجمة النص الإنجليزي إلى اللغة التي تختارها (مثل العربية 😎).

سيتم إنشاء الملفات في نفس مسار الفيديو مع الحفاظ على التوقيت الأصلي للسطر.
يتم حفظ النماذج محليًا من أول تشغيل ثم استخدامها لاحقًا بدون إنترنت 📴

### المميزات 🚀

- تفريغ الصوت الإنجليزي إلى ترجمة باستخدام Whisper.
- ترجمة النص باستخدام `google/translategemma-4b-it`.
- الحفاظ على نفس التوقيت بدقة.
- تنزيل النماذج مرة واحدة وإعادة استخدامها محليًا بدون إنترنت.
- دعم الوضع التفاعلي أو التشغيل بالأوامر.
- عرض التقدم لكل سطر ترجمة (النص الإنجليزي ثم النص المترجم).
- الكتابة إلى ملف الترجمة بشكل مباشر أثناء العمل مع إمكانية الاستكمال بعد الانقطاع.
- عند الاستكمال: إذا كان ملف `movie.en.srt` موجودًا، يتم استخدامه مباشرة بدون إعادة Whisper حتى يبقى ترتيب الأسطر ثابتًا.

### المتطلبات 🧰

- Python 3.10 أو أحدث
- Conda
- تثبيت `ffmpeg` وإضافته إلى `PATH`
- ذاكرة كافية (RAM/VRAM) خاصة لنموذج `translategemma-4b-it`

### التخزين المحلي للنماذج 🗃️

يتم حفظ النماذج في:

- `.local_models/whisper/...`
- `.local_models/huggingface/translategemma-4b-it/...`

أول تشغيل يحتاج تنزيل، وبعدها يعمل البرنامج من الملفات المحلية فقط.

### إعداد البيئة (Conda) 🐍

```powershell
conda create -n modelEnv python=3.11 -y
conda activate modelEnv
pip install -r requirements.txt
```

### طريقة الاستخدام 🎯

فعّل البيئة:

```powershell
conda activate modelEnv
```

تنزيل مسبق للنماذج (اختياري ومفضل):

```powershell
python movie_subtitle_translator.py --prepare-models --whisper-model base
```

تشغيل تفاعلي:

```powershell
python movie_subtitle_translator.py
```

تشغيل بالأوامر:

```powershell
python movie_subtitle_translator.py --movie-path "C:\movies\my_film.mp4" --target-language "Arabic"
```

تحديد حجم Whisper:

```powershell
python movie_subtitle_translator.py --movie-path "C:\movies\my_film.mp4" --target-language "Arabic" --whisper-model medium
```

تحديد مسار تخزين محلي مخصص للنماذج:

```powershell
python movie_subtitle_translator.py --movie-path "C:\movies\my_film.mp4" --target-language "Arabic" --models-dir "D:\models_cache"
```

### ملفات الإخراج 📁

إذا كان الملف:
`C:\movies\my_film.mp4`

سيتم إنشاء:

- `C:\movies\my_film.en.srt`
- `C:\movies\my_film.arabic.srt`

## Donation If You Like ❤️

USDT (BEP20):
`0x526b4d1c4f63f379991106d0bd82c402bed7aed1`
