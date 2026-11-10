# Setup

Tested on Windows 11 with an NVIDIA RTX A4000 (16 GB). Other NVIDIA cards with enough memory should work
the same way. Linux may work for the app and the command line, but only Windows has been tried; the
screensaver and the shortcuts use Windows features.

## What you need

| | |
|---|---|
| OS | Windows 10 or 11, 64-bit |
| GPU | NVIDIA, 8 GB or more of video memory (DALL·E Mega peaks at ≈ 6.5 GB, SD 1.5 at ≈ 5.7 GB) and a recent driver (the PyTorch build uses CUDA 12.8, so driver 570 or newer) |
| Disk | ≈ 9 GB for the model weights, plus room for images (the screensaver saves ≈ 0.6 GB per day of non-stop running; you can switch that off) |
| [uv](https://docs.astral.sh/uv/) | installs Python 3.12 and all packages; you don't need Python yourself |
| [Ollama](https://ollama.com) | optional: for the screensaver's random words, the "no text" check and the "words must show" check |
| [ffmpeg](https://ffmpeg.org) | optional: only for `daymovie.py` |

## Install

1. Install uv (PowerShell):

   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```

2. Get the code and install the packages (CUDA PyTorch, min-dalle, diffusers, PySide6 …, about 4 GB):

   ```bash
   git clone https://github.com/mvuijlst/images.git
   cd images
   uv sync
   ```

3. Download the model weights. Downloads resume when interrupted and are checked against their hashes.
   The app also fetches a missing model the first time it needs it, so this step is optional, but doing
   it now avoids a long wait later.

   ```bash
   uv run download_weights.py          # DALL·E Mega + Real-ESRGAN upscaler (≈ 5.2 GB)
   uv run download_weights.py --all    # + Stable Diffusion 1.5 (≈ 2.6 GB) and the name-recognition model (≈ 250 MB)
   ```

   Everything goes into `pretrained/` in this folder.

4. Make the double-click shortcuts (they start the programs without a console window):

   ```bash
   uv run make_shortcuts.py              # in this folder
   uv run make_shortcuts.py --desktop    # or on the desktop
   ```

5. Check that it works:

   ```bash
   uv run generate.py "a watercolor painting of a lighthouse at dusk" --open
   ```

## Optional: Ollama (screensaver words and image checks)

Install Ollama from [ollama.com](https://ollama.com). That's all: the screensaver starts `ollama serve`
when it isn't running and pulls its model (`qwen3.5:2b` by default, ≈ 2.7 GB) the first time. To pull it
ahead of time:

```bash
ollama pull qwen3.5:2b
```

The model has to understand images for the "no text" and "words must show" checks; `qwen3.5:2b` does.
Another model can be set in the screensaver's settings. Everything runs locally.

## Optional: ffmpeg (day movie)

```bash
winget install Gyan.FFmpeg
```

Open a new terminal afterwards so `ffmpeg` is on the PATH.

## First start of the screensaver

The settings open by themselves the first time. Pick your town (only its coordinates go to Open-Meteo
for the weather), the sources, the monitor and how long each image stays. They are kept in
`screensaver_settings.json`, which is not part of the repository. The random-words source and its scene
lists (`scene_parts.txt`, `llm_prompt.txt`) are written around Ghent, Belgium; edit them for your own
town.

## What leaves your PC

- **Weather**: your town's coordinates, to [Open-Meteo](https://open-meteo.com).
- **News**: the RSS/Atom feeds you choose are downloaded.
- **On this day**: Wikipedia's "selected anniversaries" for today.
- **Model downloads**: from Hugging Face, GitHub and Ollama, once.

Prompts, images and LLM calls stay on your machine.

## Troubleshooting

- **Generation suddenly crawls.** Windows is paging GPU memory. Check Task Manager → Performance → GPU →
  Dedicated GPU memory, close GPU-heavy programs, and don't run the app and the screensaver at the same
  time.
- **`torch.cuda.is_available()` is False.** Update the NVIDIA driver, then `uv sync` again.
- **Terminal windows flash up for every screensaver image.** That was Ollama's helper processes when the
  screensaver started Ollama without a console; it is fixed (the server is started with a hidden
  console). If you still see it, quit any `ollama serve` that an older version started (Task Manager →
  `ollama.exe`) and start the screensaver again.
- **The app crashes with heap corruption after a change to `app.pyw`.** `torch` must be imported before
  PySide6 (PySide6 ships an older MSVC runtime/OpenMP that conflicts with torch's).
- **Logs**: `app.log` and `screensaver.log` in this folder.
