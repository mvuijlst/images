# Manual

How to use the desktop app, the image screensaver, the day movie and the command line. To install, see [SETUP.md](SETUP.md).

## Desktop app

Double-click **`DALL·E Mega.lnk`** (made by `make_shortcuts.py`, see [SETUP.md](SETUP.md)) or run `uv run app.pyw`.
The last-used model loads in the background; type a prompt and press Enter.

- **Model**: DALL·E Mega or Stable Diffusion 1.5. Only one is kept in GPU memory, so switching costs a
  reload (SD ≈ 5–20 s, DALL·E Mega ≈ 25 s).
- **Negative prompt** (SD): what to steer away from, e.g. `blurry, text, watermark`.
- **Shape** (SD): 1:1 to 32:9 and down to 9:32, drawn directly (short side 512, long side a multiple
  of 64). Beyond about 2:1, SD 1.5 tends to repeat the subject (several lighthouses on one cliff).
- **Images**: 1, 4, 9 or 16 per batch (a batch is cheaper per image).
- **Size**: the short side, 256 to 2048 px; the dropdown shows the real output size (e.g. 1536 × 1024
  for SD at 3:2). Above the model's native size the image is upscaled, and the hint says how:
  up to 4× with **AI (Real-ESRGAN)**, which adds plausible detail; beyond that, plain interpolation;
  or **Smooth (Lanczos)** for interpolation only.
- **Seed**: untick *Random* to reproduce or vary a result. In an SD batch, image *n* uses seed + *n* − 1,
  so any single image can be redone on its own.
- **Advanced**, per model:
  - DALL·E Mega: guidance, temperature, top-k, **Seamless** (tileable textures; right-click → *Show tiled*);
  - SD: steps (25), guidance (7.5), sampler (DPM++ 2M Karras, Euler a, DDIM, PNDM), **Safety filter**
    (on by default: SD's NSFW checker blacks out what it flags, with occasional false positives; the
    status line says when it did).
- Live preview for both models, a time-left estimate, **Cancel** or Esc.
- Double-click an image to open it; right-click for copy / save as / show in Explorer.
- The strip at the bottom holds this session's batches; right-click one to reuse its prompt and settings.
- Everything is saved to `outputs/` as PNG, with prompt, negative prompt and settings in the file metadata.
- Errors go to `app.log`.

## Image screensaver (full screen)

Double-click **`Image Screensaver.lnk`** or run `uv run screensaver.pyw`. It fills a monitor with a
slow slideshow of new images based on real things, each one made on the spot:

- **Right now where you are**: live weather (Open-Meteo), time of day (from sunrise/sunset) and season
  shape a calm scene, e.g. "a canal with gabled houses at dusk, in light rain, late autumn, watercolor".
  Scenes come from about 95 in five kinds (town, country, water, wild, indoors with a view out), never
  the same kind twice in a row, and every scene is used once before any repeats; about half get a small
  detail (a heron, a cyclist, a sleeping cat) and a third a viewpoint. Edit `SCENES` in `realworld.py`
  to add your own.
- **Random words** (off by default; needs [Ollama](https://ollama.com)): a few words from `words.txt`
  (~2,750; 1–5 per image, default 3, set in the settings), a Ghent setting, sometimes a situation, and
  the real weather and time go to a small local LLM (default `qwen3.5:2b`), which makes one scene with
  the words as its main subjects. The program then adds the time of day and a style (first), the weather
  (as a full phrase: "in light rain, wet reflections", "starry sky, moonlight", frost below 2 °C), a
  viewpoint and sometimes a colour palette, all from `scene_parts.txt` (about 325 Ghent settings, 235
  situations, 120 viewpoints, 1,460 styles, 95 palettes). Every list hands out each item once before any
  comes back, and remembers where it was across restarts (`screensaver_bags.json`), so combinations
  practically never repeat. The caption shows the words, and under them the weather: "8 °C · clear sky ·
  night".
  - Two words or three: with `qwen3.5:2b`, two are used more reliably (all words in the first try 11/20
    vs 8/20; a word still missing after three tries 2/20 vs 5/20), and SD 1.5 fits fewer things into
    one picture better. Asking for the words as the scene's *main subjects*, first, mattered far more:
    with the earlier "use each word" wording a word went missing in 16–19 of 20 scenes.
  - The LLM's instructions are in `llm_prompt.txt`, re-read for every image, so edits show on the next
    one. Keep its placeholders: `{location}`, `{setting}`, `{situation}`, `{weather}`, `{words}` (or
    `{word1}`, `{word2}`…).
  - A small model sometimes drops a word or copies an example: each scene gets up to three tries, scenes
    that copy an example are rejected, and a word that is still missing goes in as a plain tag. The
    model's own time of day is removed, the real one goes first in the prompt (or SD 1.5 ignores it), and
    the prompt is trimmed to SD 1.5's 77 tokens, never cutting the style.
  - **Styles** are of two kinds. About 975 are artists SD 1.5 knows by name, written as "in the style of
    Kawase Hasui". The rest say what the picture is made of and how it looks ("two-colour linocut print,
    bold carved lines, flat red and black"): at night a vague style ("pavement chalk art") is lost in the
    dark redraw while a well-described one survives. The style comes right after the time of day for the
    same reason. Film looks stay "… style film still". Left out: styles that invite lettering (posters,
    postcards, covers, maps; also Aldus Manutius, Paula Scher, Paul Rand) and artists known mostly for
    nudes, which SD's safety filter would black out (Bob Mizer, Bunny Yeager, Milo Manara, Hans Bellmer,
    Nobuyoshi Araki).
  - Ollama is started (`ollama serve`) if it isn't running, and a model you name in the settings is
    pulled once. If it fails, the other sources fill in for 5 minutes. Nothing leaves the PC.
  - Try it without the screensaver: `uv run realworld.py Ghent --words`.
- **No text in the images** (all sources; on by default, needs Ollama and a model that can see images,
  like `qwen3.5:2b`): SD 1.5 adds fake watermarks, stock-photo marks, signatures and lettering that no
  negative prompt reliably stops (1 in 8 seeds for an old-photograph style in a test, with a weak and a
  strong text negative alike). So the local LLM looks at every finished image (~1 s) and, if it sees
  text, the same prompt is made again with a new seed, up to twice. In a test it caught a fake
  copyright line, a stock watermark, a magazine masthead, a woodblock cartouche and a tiny painter's
  signature; everything it flagged had text, and the 28 of 30 saved images it passed showed no obvious
  text (tiny signatures can't be ruled out). The `.json` next to each
  image records it (`text_check`). Besides that: the negative prompt names text every way (letters,
  writing, typography, caption, watermark, copyright, credit line…) minus any term that is one of the
  words; styles that invite lettering (posters, postcards, comic panels) are worded without it; and the
  LLM is told not to write signs, with quoted words ("a sign that reads 'Open'") removed.
- **The light of the hour** (weather and words; `hourlight.py`): SD 1.5 pulls every image towards
  mid-grey brightness, so "at night" in the prompt barely showed: over three days, night frames averaged
  0.43 brightness against 0.46 by day, and only ~5 % came out dark; a "daylight, sunny" negative didn't
  change that. So at night, dusk, dawn and golden hour each image is drawn twice: as usual, then again
  from a copy graded to the hour (darker and bluer at night, blue hour at dusk, warmer at golden hour),
  with SD redrawing ~60 % of it (img2img). The redraw keeps the subject and layout and turns the grade
  into real light: lit windows, street lamps, a moon. On the same prompts and seeds, night frames went
  from 0.41 to 0.11–0.16 brightness. A final check grades any frame that still comes out too bright for
  its hour, so a night frame is never shown in daylight. Costs one extra SD pass (a few seconds); the
  `.json` next to each image records it (`light`). Starting from a plain dark image instead goes
  near-black and loses the subject.
  - **No jumps between night and day**: the light follows the real sunrise and sunset and is blended
    between key moments (`MORNING` and `EVENING` in `hourlight.py`): in Ghent in November, night until
    ~6:35, dawn 7:30–8:20, day from ~9:35; golden hour from ~15:55, dusk by 17:20, night from ~18:25.
    From one minute to the next the target brightness changes by at most 0.006, so a 6:59 frame and a
    7:00 frame look alike. Close to full day the grade alone does it (no redraw).
  - Night frames keep a soft pool of light in the middle (centre ~1.4× as bright, corners ~0.65×, same
    overall darkness), where SD usually puts the subject: in plain darkness small subjects got lost.
- **The words must show** (random words; on by default, needs a vision model): SD 1.5 often draws the
  setting and drops small, odd or abstract subjects, even when the prompt names them: on 40 night
  frames the vision LLM saw only a third of the words, and every word in only 1 frame in 10. So the
  LLM looks at each first draft (still in daylight, before the light of the hour is added, ~1 s a word);
  if it sees none of the words, the draft is drawn again with a new seed, up to twice. The `.json`
  records it (`word_check`). Abstract words (yawn, late, ordinary) stay hard to show.
- **In the news**: an allegorical illustration of a current headline from RSS/Atom feeds
  (default BBC World, The Guardian World, NPR; English feeds work best).
- **On this day**: one of Wikipedia's selected anniversaries for today, painted in a style of its era
  (illuminated manuscript for 1217, art deco for the 1930s, …).

The real text is shown as a caption, with a clock. Safeguards: person and organisation names are
replaced in the image prompts by a small local name-recognition model (`dslim/distilbert-NER`), so
images don't target real people or brands; news and history use painting/illustration styles only, so
nothing looks like a photo of a real event; headlines and events with distressing words (an editable
list) are skipped; SD's safety filter is on, and blacked-out images are skipped.

The first start opens the settings: your town (only its coordinates are sent to Open-Meteo), which
sources to use, the feeds, the filter words, the **monitor** ("Identify monitors" shows their numbers),
seconds per image (default 60), captions, clock, and model (Stable Diffusion fills the screen; DALL·E
Mega's squares sit on a blurred backdrop), and saving. Settings are kept in `screensaver_settings.json`.

It runs full screen **until you close it**: clicking does nothing (the pointer shows while you move the
mouse and hides again after 3 s). Close it with Esc or Q, right-click → Exit, or Alt+F4.

| Key | |
|---|---|
| right-click | menu: next, favorite, captions, prompt, open saved images, settings, exit |
| → or Space | next image |
| S | add the current image to favorites |
| C | captions on/off |
| P | show the prompt that was used |
| O | settings |
| H | key help |
| Esc or Q | exit |

**Saving.** Every image is saved (switch off in settings) to `outputs\screensaver\YYYY-MM-DD\` as
`HHMMSS_<source>_<subject>.jpg` with a `.json` sidecar of the same name:

```json
{
  "file": "004122_news_sudanese-diaspora-and-human-rights-groups-denounce.jpg",
  "created": "2026-11-06T00:41:22+01:00",
  "source": "news",
  "caption": {"kicker": "In the news · The Guardian", "text": "Sudanese diaspora and human rights groups …"},
  "origin": {"headline": "…", "source": "The Guardian", "link": "https://www.theguardian.com/…",
             "feed": "https://www.theguardian.com/world/rss", "published": "…", "headline_in_prompt": "…"},
  "prompt": "linocut print, an allegorical scene about: …",
  "model": "Stable Diffusion 1.5",
  "generation": {"seed": 1122850964, "steps": 25, "guidance_scale": 7.5, "sampler": "DPM++ 2M Karras",
                 "negative_prompt": "…", "safety_filter": true},
  "image": {"width": 1600, "height": 960, "format": "JPEG", "quality": 92, "drawn_at": [832, 512],
            "upscaled_with": "Real-ESRGAN x4plus", "fitted": "cropped to the screen's shape"},
  "screen": {"monitor": "\\\\.\\DISPLAY2", "width": 800, "height": 480},
  "seconds_to_make": 6.9,
  "app": "Image Screensaver"
}
```

`origin` holds the real-world details: for weather the place, coordinates, temperature, conditions,
sunrise/sunset and local time; for history the year, the full event text and its Wikipedia pages.
The JPEG itself also carries the caption (Explorer shows it as Title, Subject, Tags and Comments).
Saved images are **twice the monitor's resolution** (1600 × 960 for an 800 × 480 display, ~0.3–0.5 MB
each, ≈ 0.6 GB per day of non-stop running at one image a minute), but no more than the upscaler can
genuinely add (2048 px on the short side). Favorites (S) are copied to `favorites\` with
`"favorite": true` in their sidecar. Optionally, day folders older than N days are deleted; favorites
never are.

The next image is made while the current one is showing, so the GPU works ~10 s per minute and idles
in between. While it runs, Windows won't blank the display or go to sleep. Run it *or* the main app:
both load a model into the same GPU, and together they are slow. Log: `screensaver.log`.

Command line: `uv run screensaver.pyw --settings` (open settings first), `--monitor 2` (that monitor).

**Day movie.** `uv run daymovie.py [2026-11-06]` turns a day folder (default: the latest) into an MP4
next to it (`outputs\screensaver\2026-11-06.mp4`), with ffmpeg. Every frame is drawn the way the
screensaver draws it: the same caption band, clock and crossfade, at the saved images' resolution
(1600 × 960). The clock shows when each image was made. A whole day fits in `--length` (default 2:55):
each image gets that length divided by the number of images, at most `--seconds` (2), so a full day of
~1300 images runs as a time-lapse at ~0.13 s per image, and a short day at 2 s per image. The crossfade
is at most `--fade` (0.5 s) and a quarter of an image's time; a caption identical to the previous one
stays put through the fade instead of dipping (at time-lapse speed that would flicker). Other options:
`--fps 30`, `--no-captions`, `--no-clock`, `--prompt`, `--crf 18`, `--out FILE`.
If you change how the screensaver draws its caption or clock, change `daymovie.py` the same way.

## Command line

```bash
uv run generate.py "a watercolor painting of a lighthouse at dusk" --grid 2 --size 1024
uv run generate.py "a gothic quarter at night" --model sd --aspect 16:9 --negative "blurry, text"
uv run generate.py          # interactive: the model stays loaded, one prompt per line
```

In interactive mode, options typed after a prompt apply to that prompt only
(`a mountain lake at dawn --model sd --aspect 3:2`); options on the command line become the session
defaults. Empty line = repeat the last prompt.

| Option | Default | Notes |
|---|---|---|
| `--model` | mega | `mega` (DALL·E Mega) or `sd` (Stable Diffusion 1.5) |
| `--aspect W:H` | 1:1 | SD only, e.g. `3:2`, `16:9`, `9:16` |
| `--grid N` | 1 | N×N images in one batch (Mega seamless: one texture of N×N tiles) |
| `--size PX` | native | output short side (256 Mega / 512 SD); larger is upscaled |
| `--upscaler` | ai | `ai` (Real-ESRGAN) or `lanczos` |
| `--seed` | random | same seed + prompt + options gives the same image |
| `--negative` | | SD negative prompt |
| `--steps` / `--cfg` / `--sampler` | 25 / 7.5 / dpmpp_2m_karras | SD sampling |
| `--no-safety` | | SD: turn off the NSFW safety filter |
| `--supercondition` / `--temperature` / `--top-k` | 16 / 1.0 / 256 | DALL·E Mega sampling |
| `--seamless` | off | DALL·E Mega tileable output |
| `--out DIR` | `outputs` | where images are saved |
| `--open` | off | open the result in the default image viewer |
| `--dtype` | float16 | DALL·E Mega weights; `bfloat16` / `float32` also work |

## Speed and memory (measured on an RTX A4000 16 GB, fp16)

| | time | VRAM peak |
|---|---|---|
| DALL·E Mega, 1 image / 4 images | ≈ 18 s / ≈ 22 s | ≈ 6.5 GB |
| SD 1.5, 1 image 512 × 896 (25 steps) | ≈ 7 s | |
| SD 1.5, 4 images 896 × 512 | ≈ 15 s | |
| SD 1.5, 4 images 3:2 upscaled to 1536 × 1024 | ≈ 30 s | ≈ 5.7 GB |
| AI upscaling | ≈ 0.3 s per image | small |

The GPU also drives the desktop: here the window manager, browser, Figma etc. hold ≈ 6.5 GB of the
16 GB, which is why only one model is kept loaded. If generation suddenly crawls, Windows is paging
GPU memory: check Task Manager → Performance → GPU → Dedicated GPU memory.

## Files

- `app.pyw`: desktop app (PySide6)
- `screensaver.pyw`: full-screen real-world image generator
- `realworld.py`: weather / three words / news / on-this-day sources and prompt building
- `hourlight.py`: grades images to the light of the hour (night, dusk, dawn, golden hour)
- `daymovie.py`: a movie of one day of screensaver images, frames as on screen
- `words.txt`, `scene_parts.txt`, `llm_prompt.txt`: word list, scene parts and LLM instructions for the
  three-words source (plain text, edit freely)
- `screensaver_bags.json`: which list items were already used this round (delete to start over)
- `generate.py`: command line
- `dalle_core.py`: DALL·E Mega engine, upscaling, saving, model switching (shared)
- `sd_engine.py`: Stable Diffusion 1.5 engine (diffusers); `Settings.init_image` + `strength` for img2img
- `download_weights.py`: weight downloader and verifier
- `make_shortcuts.py`: creates the two double-click shortcuts (Windows)
- `icon.ico`, `icon.png`: app icon (drawn by DALL·E Mega itself)

Note for changes to `app.pyw`: `torch` must be imported before PySide6, or generation crashes with
heap corruption (PySide6 ships an older MSVC runtime/OpenMP that conflicts with torch's).
