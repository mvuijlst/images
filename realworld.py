"""Real-world prompt sources for the screensaver.

- Weather: live conditions, time of day and season at a chosen place (Open-Meteo, no key).
- News: headlines from RSS/Atom feeds.
- On this day: Wikipedia's selected anniversaries for today's date.
- Three words: three random words from words.txt, plus the place, weather and a street setting, turned
  into a scene by a small local LLM through Ollama (template in llm_prompt.txt, re-read every image).

Each source turns something real into an Idea: a prompt for the image model, and the real text to
show as a caption. Person and organisation names are removed from news/history prompts by a small
local NER model, so images never target real people or brands; the caption keeps the original.
News and history use painting/illustration styles only, so nothing looks like a photo of a real event.
"""

import base64
import io
import json
import logging
import os
import random
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import requests

import download_weights as dw
import hourlight

HERE = Path(__file__).resolve().parent
HEADERS = {"User-Agent": "PrimitiveImageScreensaver/1.0 (local desktop app)"}
TIMEOUT = 15
log = logging.getLogger(__name__)

DEFAULT_FEEDS = [
    "https://feeds.bbci.co.uk/news/world/rss.xml",
    "https://www.theguardian.com/world/rss",
    "https://feeds.npr.org/1001/rss.xml",
]
DEFAULT_SKIP_WORDS = (
    "kill, kills, killed, killing, killings, dead, death, deaths, die, dies, died, dying, murder, murders, "
    "murdered, shooting, shootings, shot, stabbing, stabbed, rape, raped, abuse, abused, massacre, suicide, "
    "terror, terrorist, terrorism, bomb, bombs, bombing, bombed, airstrike, airstrikes, missile, missiles, "
    "war, wars, genocide, execution, executed, hostage, hostages, assault, wounded, injured, casualties, "
    "corpse, bodies, attack, attacks, attacked, explosion, derailment, shipwreck, assassin, assassinate, "
    "assassinated, assassination, slain, beheaded, torture, tortured, lynching, lynched"
)

# SD 1.5 learned lettering from captioned stock and archive photos, posters and comics: name it every way
NEG_BASE = ("text, letters, words, writing, typography, caption, title, subtitles, label, logo, watermark, "
            "signature, copyright, credit line, frame, border, blurry, lowres, deformed, ugly")
NEG_PEOPLE = "close-up face, portrait, celebrity, politician"
NEG_GRIM = "gore, blood, corpse, injury, weapon"


@dataclass
class Idea:
    prompt: str
    negative: str
    kicker: str  # small caption line, e.g. "In the news · BBC News"
    text: str  # the real thing: conditions, headline, event
    source: str  # "weather" | "news" | "history"
    origin: dict = field(default_factory=dict)  # the real-world details behind it (for metadata)


@dataclass
class Place:
    name: str
    country: str
    lat: float
    lon: float

    @property
    def label(self) -> str:
        return f"{self.name}, {self.country}" if self.country else self.name


def _get(url: str, **kw) -> requests.Response:
    r = requests.get(url, headers=HEADERS, timeout=TIMEOUT, **kw)
    r.raise_for_status()
    return r


def geocode(query: str) -> Place | None:
    """City name → Place via Open-Meteo's geocoder (sends only the query text)."""
    r = _get("https://geocoding-api.open-meteo.com/v1/search", params={"name": query, "count": 1, "language": "en"})
    results = r.json().get("results") or []
    if not results:
        return None
    g = results[0]
    return Place(g["name"], g.get("country", ""), g["latitude"], g["longitude"])


# --------------------------------------------------------------------------- names out of prompts


class NameStripper:
    """Replaces person and organisation names with neutral words, using dslim/distilbert-NER
    (local, CPU). Falls back to a capitalised-word heuristic if the model can't load."""

    TITLES = {"king", "queen", "prince", "princess", "emperor", "empress", "pope", "president", "tsar", "czar",
              "sultan", "duke", "duchess", "lord", "lady", "sir", "saint", "st.", "general", "admiral", "chancellor",
              "governor", "senator", "bishop", "archbishop", "cardinal", "pharaoh", "shah", "khan", "count", "countess"}

    def __init__(self):
        self.ner = None
        try:
            import logging
            import warnings

            # importing transformers pulls in torch.distributed, which logs a Windows-only notice
            logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
            import transformers
            from transformers import pipeline

            transformers.utils.logging.set_verbosity_error()
            transformers.utils.logging.disable_progress_bar()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.ner = pipeline("token-classification", model=str(dw.ensure_ner()),
                                    aggregation_strategy="simple", device=-1)
        except Exception:  # offline before the first download, etc.
            self.ner = None

    def __call__(self, text: str) -> str:
        if self.ner is None:
            return self._heuristic(text)
        spans = [e for e in self.ner(text) if e["entity_group"] in ("PER", "ORG") and e["score"] > 0.5]
        for e in sorted(spans, key=lambda e: -e["start"]):
            start, end = e["start"], e["end"]
            while end < len(text) and (text[end].isalnum() or text[end] in "-"):  # whole word
                end += 1
            if text[end:end + 2] in ("'s", "’s"):
                end += 2
            before = text[:start]
            if e["entity_group"] == "PER":
                title = re.search(r"(\b[A-Z][a-z]+\.?)\s+$", before)
                first = text[start:end].split()[0].lower()
                if title and title.group(1).lower() in self.TITLES:  # "King Henry III" → "a king"
                    before, word = before[: title.start()], f"a {title.group(1).lower().rstrip('.')}"
                elif first in self.TITLES and len(text[start:end].split()) > 1:  # span "Pope Francis" → "a pope"
                    word = f"a {first.rstrip('.')}"
                else:
                    word = "a person"
            elif re.search(r"\bthe\s+$", before, re.I) or re.search(r"[A-Z][\w.]*\s+$", before):
                word = "institution"  # "the Senate" → "the institution", "United States Senate" → "… institution"
            else:
                word = "an institution"
            if re.match(r"\ba ", word) and re.search(r"\b(a|an)\s+$", before):  # avoid "a a person"
                before = re.sub(r"\b(a|an)\s+$", "", before)
            text = before + word + text[end:]
        # merge runs like "a person and a person"
        text = re.sub(r"\ba person(?:,? and a person)+\b", "two people", text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _heuristic(text: str) -> str:
        words = text.split()
        kept = [w for i, w in enumerate(words) if i == 0 or not w[:1].isupper()]
        return " ".join(kept)


def _skip_pattern(words: str) -> re.Pattern | None:
    items = [w.strip().lower() for w in words.split(",") if w.strip()]
    if not items:
        return None
    return re.compile(r"\b(" + "|".join(re.escape(w) for w in items) + r")\b", re.I)


# --------------------------------------------------------------------------- weather, time, season

WMO = {  # code: (caption, prompt phrase)
    0: ("clear sky", "under a clear sky"),
    1: ("mostly clear", "under a mostly clear sky"),
    2: ("partly cloudy", "with scattered clouds"),
    3: ("overcast", "under a soft grey overcast sky"),
    45: ("fog", "in thick fog"), 48: ("freezing fog", "in freezing fog, frost on everything"),
    51: ("light drizzle", "in light drizzle"), 53: ("drizzle", "in drizzle"), 55: ("dense drizzle", "in dense drizzle"),
    56: ("freezing drizzle", "in freezing drizzle"), 57: ("freezing drizzle", "in freezing drizzle"),
    61: ("light rain", "in light rain, wet reflections"), 63: ("rain", "in the rain, wet streets and reflections"),
    65: ("heavy rain", "in heavy rain"), 66: ("freezing rain", "in freezing rain"), 67: ("freezing rain", "in freezing rain"),
    71: ("light snow", "in gently falling snow"), 73: ("snow", "in falling snow, snow-covered"),
    75: ("heavy snow", "in heavy snowfall, deep snow"), 77: ("snow grains", "with fine snow in the air"),
    80: ("rain showers", "during passing rain showers"), 81: ("rain showers", "during rain showers"),
    82: ("violent showers", "during a violent downpour"),
    85: ("snow showers", "during snow showers"), 86: ("heavy snow showers", "during heavy snow showers"),
    95: ("thunderstorm", "during a thunderstorm, lightning in the sky"),
    96: ("thunderstorm with hail", "during a thunderstorm with hail"), 99: ("thunderstorm with hail", "during a thunderstorm with hail"),
}

# Scenes for the weather source, by kind. A scene is drawn from a kind that differs from the previous
# one, every scene of a kind is used once before any comes back, and about half get a small detail
# and a third a viewpoint, so the same scene rarely looks the same twice. Keep SD 1.5's habits in mind
# when adding: coasts sprout lighthouses and rivers sprout bridges, so few scenes name either.
SCENES = {
    "town": [
        "a quiet canal lined with old gabled houses", "an old cobblestone street with a corner café",
        "a market square with a medieval belfry", "a city skyline seen from a rooftop garden",
        "a railway station platform", "a quiet béguinage courtyard with whitewashed houses",
        "a covered market hall with iron columns", "a narrow alley with steps and potted plants",
        "a bookshop window on a side street", "a small square with a fountain and plane trees",
        "rooftops and chimneys seen from an attic window", "a tram stop on a wide boulevard",
        "a riverside quay with moored barges", "a park bandstand among old trees",
        "a row of colourful terraced houses", "a corner bakery with a striped awning",
        "a university courtyard with ivy-covered walls", "an old cinema facade on a quiet street",
        "a small petrol station on the edge of town", "a Victorian glasshouse in a botanical garden",
        "a street of narrow brick houses with bicycles outside", "a church square with linden trees",
        "a fish market by the harbour", "a city park with a pond and a boathouse",
    ],
    "country": [
        "rolling farmland with a windmill", "a hillside vineyard with a farmhouse",
        "a country road lined with poplar trees", "an allotment garden with a small shed",
        "an orchard of old apple trees", "a farmyard with a red barn",
        "a village green with a small stone church", "a hay meadow with a wooden gate",
        "a walled kitchen garden with a greenhouse", "a lone oak tree in a wide field",
        "a flat polder landscape with ditches and pollard willows", "a watermill by a stream",
        "a level crossing on a country railway", "a ploughed field with a hedgerow",
        "a hop garden with tall poles", "a country bus stop with a wooden shelter",
        "a row of beehives at the edge of a field", "a winding lane between hedges",
        "a farm pond with willows", "a hamlet of thatched cottages",
    ],
    "water": [
        "a lighthouse on a rocky coast", "a small harbour with fishing boats",
        "a cliff-top path above the sea", "a beach with dunes and marram grass",
        "a wooden pier stretching into the sea", "a row of beach huts on a pebble beach",
        "a tidal flat at low tide", "a lake with a wooden jetty and a rowing boat",
        "a wide river with an old stone bridge", "a reed-fringed marsh",
        "a lock keeper's cottage on a canal", "a houseboat moored on a quiet river",
        "a small ferry crossing a wide estuary", "a seaside promenade with old hotels",
        "rock pools on a rocky shore", "a river bend with a sandy bank",
        "a quiet fishing village on a fjord", "a wooden boathouse on a lake",
    ],
    "wild": [
        "a misty mountain valley with a small village", "a forest clearing with tall trees",
        "a lakeside cabin among birch trees", "a beech forest with a winding path",
        "a heath with heather and a sandy track", "a mountain pass with a stone refuge hut",
        "a pine forest on a steep hillside", "a waterfall in a mossy gorge",
        "a boardwalk through a peat bog", "a glacier valley with a turquoise lake",
        "a ruined castle on a wooded hill", "a moorland with a winding stone wall",
        "an alpine meadow with a wooden chalet", "a hilltop with ancient standing stones",
        "a woodland pond with fallen trees", "a rocky mountain ridge above the clouds",
        "a birch wood with silver trunks", "a wide river delta seen from a hill",
    ],
    "indoor": [  # all have a view outside, so the weather and time of day still show
        "a cosy kitchen with a big window", "a reading nook by a tall window",
        "an artist's studio with a skylight and canvases", "a window seat on a train",
        "a café table by a large window", "an old library reading room with tall windows",
        "a woodworking workshop with a dusty window", "a conservatory with wicker chairs",
        "an attic bedroom with a round window", "a potting shed with clay pots and a window",
        "a music room with a piano by the window", "a quiet museum gallery with a large window",
        "the inside of an empty tram", "a bakery counter with a window onto the street",
    ],
}
KIND_WEIGHTS = {"town": 3, "country": 2, "water": 2, "wild": 2, "indoor": 1}
DETAILS = {  # optional small focus for a scene
    "town": [
        "a cyclist passing by", "a black cat on a doorstep", "pigeons on the pavement", "a yellow tram",
        "a street musician", "a red bicycle against a wall", "a person walking a dog", "a newspaper kiosk",
        "a flock of starlings overhead", "an old car parked by the kerb", "a lone walker", "a flower stall",
    ],
    "country": [
        "a flock of sheep", "a grazing horse", "crows in a field", "a tractor", "a scarecrow",
        "a hare in the grass", "hay bales", "a walker with a dog", "geese flying overhead", "a cyclist on the lane",
    ],
    "water": [
        "a heron in the shallows", "gulls overhead", "a small sailing boat", "swans on the water",
        "a fisherman on the bank", "ducks on the water", "a cormorant on a post", "a kayaker",
    ],
    "wild": [
        "a deer among the trees", "a fox", "a hiker in a red coat", "an owl on a branch", "a hawk circling",
        "a small tent", "a red squirrel", "wild ponies", "a wooden signpost",
    ],
    "indoor": [
        "a sleeping cat", "a steaming cup of tea", "a stack of old books", "a vase of flowers", "an old radio",
        "a dog on a rug", "a potted fern", "a bowl of apples",
    ],
}
VIEWS = ["wide view", "seen from above", "low angle view", "seen from a distance", "panoramic view",
         "close view"]  # outdoor scenes only
NIGHT = {  # kind → (clear night, cloudy night)
    "town": ("at night under a starry sky, moonlight", "at night, glowing windows and street lights"),
    "country": ("at night under a starry sky, moonlight", "at night, a few lit windows in the dark"),
    "water": ("at night, moonlight on the water, starry sky", "at night, distant lights reflected in dark water"),
    "wild": ("at night under a starry sky, moonlight", "at night, faint moonlight through the clouds"),
    "indoor": ("at night, warm lamplight, stars outside", "at night, warm lamplight, dark windows"),
}
AMBIENT_STYLES = [
    "impressionist oil painting", "watercolor painting", "detailed gouache illustration",
    "atmospheric cinematic photograph", "Japanese woodblock print", "soft pastel drawing",
    "tonalist landscape painting", "plein-air oil sketch", "35mm film photograph", "Studio Ghibli-like background painting",
    "pen and ink drawing with watercolor wash", "pointillist painting", "1950s travel poster illustration",
    "Hudson River School painting", "polaroid photograph", "colored pencil drawing",
]


BAGS_PATH = HERE / "screensaver_bags.json"


class ShuffleBag:
    """Hands out every item once, in random order, before any comes back (and never the same twice in a row).
    A named bag remembers what it handed out in screensaver_bags.json, so a restart doesn't start it over;
    items added to a list later join the current round, removed ones simply drop out."""

    _saved: dict | None = None  # all named bags, loaded once

    def __init__(self, items: list, name: str | None = None):
        self.items, self.name = list(dict.fromkeys(items)), name
        state = self._state().get(name) if name else None
        known = set(self.items)
        self.drawn = [i for i in (state or {}).get("drawn", []) if i in known]
        left = [i for i in self.items if i not in set(self.drawn)]
        self.bag = random.sample(left, len(left)) if state else []
        self.last = (state or {}).get("last")

    @classmethod
    def _state(cls) -> dict:
        if cls._saved is None:
            try:
                cls._saved = json.loads(BAGS_PATH.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                cls._saved = {}
        return cls._saved

    def draw(self):
        if not self.items:
            return ""
        if not self.bag:
            self.bag, self.drawn = random.sample(self.items, len(self.items)), []
            if len(self.bag) > 1 and self.bag[-1] == self.last:  # draw() pops from the end
                self.bag[0], self.bag[-1] = self.bag[-1], self.bag[0]
        self.last = self.bag.pop()
        self.drawn.append(self.last)
        if self.name:
            self._save()
        return self.last

    def _save(self) -> None:
        state = self._state()
        state[self.name] = {"drawn": self.drawn, "last": self.last}
        tmp = BAGS_PATH.with_suffix(".tmp")
        try:
            tmp.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, BAGS_PATH)
        except OSError:  # read-only folder etc.: just don't remember
            pass
SEASONS = {  # (northern-hemisphere months) → phrase
    (12, 1, 2): "in winter, bare trees",
    (3,): "in early spring, first green buds", (4,): "in spring, blossoming trees", (5,): "in late spring, lush and green",
    (6, 7): "in summer, full green trees", (8,): "in late summer, golden fields",
    (9,): "in early autumn, first yellow leaves", (10,): "in autumn, red and golden leaves",
    (11,): "in late autumn, bare branches and fallen leaves",
}


@dataclass
class Conditions:
    temp: float
    code: int
    local: datetime
    sunrise: datetime | None
    sunset: datetime | None


def time_of_day(c: Conditions, kind: str = "town") -> tuple[str, str]:
    """(caption word, prompt phrase); the night phrase suits the kind of scene (street lights only in town)."""
    now, rise, set_ = c.local, c.sunrise, c.sunset
    if rise is None or set_ is None:  # rough fallback
        rise, set_ = now.replace(hour=7, minute=0), now.replace(hour=19, minute=0)
    m = lambda minutes: timedelta(minutes=minutes)  # noqa: E731
    clear = c.code in (0, 1)
    if now < rise - m(45) or now > set_ + m(60):
        return "night", NIGHT.get(kind, NIGHT["town"])[0 if clear else 1]
    if now <= rise + m(45):
        return "dawn", "at dawn, first light on the horizon"
    if now < rise + m(180):
        return "morning", "on a fresh morning, soft light"
    if now >= set_:
        return "dusk", "at dusk in the blue hour, first lights coming on"
    if now >= set_ - m(75):
        return "golden hour", "at golden hour, long shadows and warm low sunlight"
    noon = rise + (set_ - rise) / 2
    if abs(now - noon) <= m(75):
        return "midday", "at midday"
    return "afternoon", "in the afternoon"


def season_phrase(month: int, lat: float) -> str:
    if lat < 0:  # southern hemisphere: shift by six months
        month = (month + 5) % 12 + 1
    for months, phrase in SEASONS.items():
        if month in months:
            return phrase
    return ""


def temp_phrase(t: float) -> str:
    if t < -5:
        return "bitterly cold, frost and ice"
    if t < 2:
        return "freezing cold, frost on the ground"
    if t < 8:
        return "chilly air"
    if t > 29:
        return "summer heat haze"
    if t > 23:
        return "warm air"
    return ""


class WeatherSource:
    def __init__(self, place: Place | None):
        self.place = place
        self._cache: tuple[float, Conditions] | None = None
        self._scenes = {k: ShuffleBag(v, f"weather:scenes:{k}") for k, v in SCENES.items()}
        self._details = {k: ShuffleBag(v, f"weather:details:{k}") for k, v in DETAILS.items()}
        self._styles = ShuffleBag(AMBIENT_STYLES, "weather:styles")
        self._last_kind: str | None = None

    def conditions(self) -> Conditions | None:
        if self.place is None:
            return None
        if self._cache and time.time() - self._cache[0] < 600:
            return self._cache[1]
        r = _get("https://api.open-meteo.com/v1/forecast", params={
            "latitude": self.place.lat, "longitude": self.place.lon,
            "current": "temperature_2m,weather_code", "daily": "sunrise,sunset",
            "timezone": "auto", "forecast_days": 1,
        }).json()
        cur, daily = r["current"], r.get("daily", {})
        parse = lambda s: datetime.fromisoformat(s) if s else None  # noqa: E731
        offset = timedelta(seconds=r.get("utc_offset_seconds", 0))
        c = Conditions(
            temp=cur["temperature_2m"], code=int(cur["weather_code"]),
            local=(datetime.now(timezone.utc).replace(tzinfo=None) + offset).replace(microsecond=0),
            sunrise=parse((daily.get("sunrise") or [None])[0]), sunset=parse((daily.get("sunset") or [None])[0]),
        )
        self._cache = (time.time(), c)
        return c

    def _scene(self) -> dict:
        kinds = [k for k in KIND_WEIGHTS if k != self._last_kind]
        kind = random.choices(kinds, weights=[KIND_WEIGHTS[k] for k in kinds])[0]
        self._last_kind = kind
        return {"kind": kind, "scene": self._scenes[kind].draw(),
                "detail": self._details[kind].draw() if random.random() < 0.5 else "",
                "view": random.choice(VIEWS) if kind != "indoor" and random.random() < 0.35 else ""}

    def idea(self) -> Idea:
        try:
            c = self.conditions()
        except requests.RequestException:
            c = None
        s, style = self._scene(), self._styles.draw()
        live = c is not None
        if not live:  # no location or offline: local clock and calendar only
            c = Conditions(temp=12, code=2, local=datetime.now(), sunrise=None, sunset=None)
        tod, tod_phrase = time_of_day(c, s["kind"])
        desc, weather = WMO.get(c.code, ("", "")) if live else ("", "")
        if tod == "night" and c.code in (0, 1):
            weather = ""  # the starry-sky night phrase already says it
        elif s["kind"] == "indoor" and desc:
            weather = f"{desc} outside the window"
        elif s["kind"] != "town":
            weather = weather.replace("wet streets and reflections", "wet reflections")
        season = season_phrase(c.local.month, self.place.lat if self.place else 50)
        parts = [s["scene"], s["detail"], s["view"], tod_phrase, weather, season,
                 temp_phrase(c.temp) if live else "", style, "beautiful composition, highly detailed"]
        prompt = ", ".join(p for p in parts if p)
        scene = {"scene_kind": s["kind"], "scene": s["scene"], "detail": s["detail"] or None, "view": s["view"] or None}
        negative = hourlight.add_negative(NEG_BASE, hourlight.light_at(c.local, c.sunrise, c.sunset))  # "sunny" at night
        if not live:
            origin = {"local_time": c.local.isoformat(timespec="minutes"), "time_of_day": tod, "season": season,
                      **scene, "note": "no location set or weather unavailable: clock and calendar only"}
            return Idea(prompt, negative, "Right now", f"{tod} · {c.local:%H:%M}", "weather", origin)
        text = f"{c.temp:.0f} °C · {desc} · {tod}"
        origin = {
            **scene, "place": self.place.label, "latitude": self.place.lat, "longitude": self.place.lon,
            "temperature_c": c.temp, "weather_code": c.code, "conditions": desc, "time_of_day": tod,
            "season": season, "local_time": c.local.isoformat(timespec="minutes"),
            "sunrise": c.sunrise.isoformat(timespec="minutes") if c.sunrise else None,
            "sunset": c.sunset.isoformat(timespec="minutes") if c.sunset else None,
            "provider": "Open-Meteo (open-meteo.com)",
        }
        return Idea(prompt, negative, f"Right now in {self.place.name}", text, "weather", origin)

    def describe(self) -> tuple[str, str, dict]:
        """(location, weather in words, details) for the LLM, e.g. "Cool night in late autumn, 7°C, overcast"."""
        try:
            c = self.conditions()
        except requests.RequestException:
            c = None
        live = c is not None
        if not live:
            c = Conditions(temp=12, code=2, local=datetime.now(), sunrise=None, sunset=None)
        tod, _ = time_of_day(c)
        season = season_phrase(c.local.month, self.place.lat if self.place else 50).split(",")[0].removeprefix("in ")
        when = {"golden hour": "golden-hour evening"}.get(tod, tod)
        details = {"time_of_day": tod, "season": season, "local_time": c.local.isoformat(timespec="minutes"),
                   "sunrise": c.sunrise.isoformat(timespec="minutes") if c.sunrise else None,  # for the light
                   "sunset": c.sunset.isoformat(timespec="minutes") if c.sunset else None}
        if not live:
            return "a small European town", f"{when.capitalize()} in {season}", {
                **details, "note": "no location set or weather unavailable: clock and calendar only"}
        desc = WMO.get(c.code, ("", ""))[0]
        details.update(place=self.place.label, temperature_c=c.temp, weather_code=c.code, conditions=desc,
                       provider="Open-Meteo (open-meteo.com)")
        return self.place.name, f"{feel_word(c.temp).capitalize()} {when} in {season}, {c.temp:.0f}°C, {desc}", details


def feel_word(t: float) -> str:
    for limit, word in ((-5, "bitterly cold"), (2, "freezing"), (6, "cold"), (14, "cool"), (20, "mild"), (26, "warm")):
        if t < limit:
            return word
    return "hot"


# --------------------------------------------------------------------------- three words, local LLM

DEFAULT_LLM = "qwen3.5:2b"
OLLAMA_URL = "http://127.0.0.1:11434"
WORDS_PATH = HERE / "words.txt"
LLM_PROMPT_PATH = HERE / "llm_prompt.txt"
SCENE_PARTS_PATH = HERE / "scene_parts.txt"
CLIP_LIMIT = 77  # SD 1.5's text encoder reads this many tokens (start and end included) and drops the rest
# short time-of-day tags for the end of the prompt (the LLM writes the scene, these say when)
WHEN_TAGS = {"night": "at night", "dawn": "at dawn", "morning": "morning light", "dusk": "at dusk, blue hour",
             "golden hour": "golden hour", "midday": "midday", "afternoon": "afternoon light"}
# for each image: one setting, a situation (65%), a viewpoint (80%), a style, a palette (35%)
PART_CHANCE = {"settings": 1.0, "situations": 0.65, "viewpoints": 0.8, "styles": 1.0, "palettes": 0.35}


def load_words(path: Path = WORDS_PATH) -> list[str]:
    words: dict[str, None] = {}  # ordered, no duplicates
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip().startswith("#"):
            words.update(dict.fromkeys(w.lower() for w in re.split(r"[\s,]+", line) if w))
    return list(words)


def load_parts(path: Path = SCENE_PARTS_PATH) -> dict[str, list[str]]:
    """scene_parts.txt → {"settings": [...], "situations": [...], ...}."""
    parts: dict[str, list[str]] = {}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("## "):
            current = parts.setdefault(line[3:].strip().lower(), [])
        elif line and not line.startswith("#") and current is not None:
            current.append(line)
    return parts


def clean_llm_prompt(text: str) -> list[str]:
    """The tags on the first line the model wrote, without labels, quotes or a reasoning block. Sentences
    count too (a small model sometimes writes prose): they are split at commas, full stops and semicolons."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    lines = [l.strip() for l in text.splitlines()
             if l.strip() and not re.fullmatch(r"(?i)(output|prompt|scene):?", l.strip())]
    if not lines:
        return []
    line = re.sub(r"(?i)^(output|prompt|scene)\s*:\s*", "", lines[0]).strip(" \"'`*")
    # spelled-out lettering makes SD 1.5 draw (garbled) text: "a plaque that reads 'Oysters'" → "a plaque"
    line = re.sub(r"(?i)\s*\b(?:that |which )?(?:reads?|reading|says?|saying|labell?ed|titled|spelling(?: out)?)"
                  r"\s*:?\s*(['\"“‘«]).*?(?:['\"”’»]|$)", "", line)
    line = re.sub(r"[\"“«][^\"”»]*[\"”»]", "", line)  # any other double-quoted words
    sentence_end = r"(?<!\bSt)(?<!\bMr)(?<!\bDr)(?<!\bMrs)\.\s+"  # not after "St." in "St. Catherine"
    return [t for t in (t.strip(" .\"'`*") for t in re.split(rf",|;|{sentence_end}|\.$", line)) if t]


_TIME = r"(dusk|dawn|night|nighttime|midnight|noon|midday|sunset|sunrise|twilight|evening|morning|afternoon|daytime|golden hour|blue hour)"


def strip_time(tags: list[str]) -> list[str]:
    """Drop the model's own time of day ("at dusk"): the real one is added after the scene."""
    out = []
    for t in tags:
        t = re.sub(rf"\s*\b(at|in the|during the|by|on a)\s+{_TIME}\b", "", t, flags=re.I).strip()
        if t and not re.fullmatch(rf"(?i)((at|in the|during the) )?{_TIME}( light| sky| air)?", t):
            out.append(t)
    return out


def _stem(word: str) -> str:
    word = word.lower().split("-")[0]
    return word[: max(4, len(word) - 2)]  # "secure" matches "secured", "teddy" matches "teddy bear"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zà-ÿ0-9']+", text.lower())


def copied_from_examples(scene: str, template: str, own: str, n: int = 5) -> str | None:
    """A run of n words the scene shares with the template's example scenes (not explained by its own input)."""
    examples = [l.split(":", 1)[1] for l in template.splitlines()
                if re.match(r"(?i)\s*(scene|output)\s*:\s*\S", l) and "{" not in l] or [template]
    seen = {" ".join(t[i:i + n]) for e in examples for t in [_tokens(e)] for i in range(len(t) - n + 1)}
    mine = f" {' '.join(_tokens(own))} "
    words = _tokens(scene)
    for i in range(len(words) - n + 1):
        run = " ".join(words[i:i + n])
        if run in seen and f" {run} " not in mine:
            return run
    return None


_clip_tokenizer = None


def clip_length(text: str) -> int:
    """Tokens SD 1.5's text encoder sees, using its own tokenizer when the weights are there."""
    global _clip_tokenizer
    if _clip_tokenizer is None:
        try:
            import transformers
            from transformers import CLIPTokenizer

            transformers.utils.logging.set_verbosity_error()
            _clip_tokenizer = CLIPTokenizer.from_pretrained(str(dw.SD_DIR), subfolder="tokenizer")
        except Exception:
            _clip_tokenizer = False
    if _clip_tokenizer is False:  # rough: ~1.3 tokens a word, commas count, plus start and end
        return int(len(text.split()) * 1.3 + text.count(",")) + 2
    return len(_clip_tokenizer(text).input_ids)


def fit_prompt(scene: list[str], tail: dict[str, str], keep: str = "") -> tuple[str, list[str]]:
    """When, the setting (first scene tag), the weather, the rest of the scene, then season, view, style,
    palette (time and weather come early, or SD 1.5 all but ignores them); trimmed to fit CLIP_LIMIT so the
    style at the end always survives. Trimmed in order: the palette, the season, scene details beyond three, long
    sentence-like tags down to 8 words, the viewpoint, the weather, long tags down to 3 words.
    Words containing `keep` (the city), and the "in"/"near" before them, are never cut."""
    scene, tail, dropped = list(scene), dict(tail), []
    front = ("when", "weather")
    text = lambda: ", ".join(t for t in [tail.get("when", "")] + scene[:1] + [tail.get("weather", "")]  # noqa: E731
                             + scene[1:] + [v for k, v in tail.items() if k not in front] if t)
    fits = lambda: clip_length(text()) <= CLIP_LIMIT  # noqa: E731

    def drop_tail(key: str) -> None:
        if tail.get(key):
            dropped.append(tail[key])
            tail[key] = ""

    def shorten(min_words: int) -> None:
        while not fits():
            i = max(range(len(scene)), key=lambda k: len(scene[k].split()))
            words = scene[i].split()
            protected = {k for k, w in enumerate(words) if keep and keep.lower() in w.lower()}
            protected |= {k - 1 for k in protected if k and words[k - 1].lower() in ("in", "near", "of")}
            cut = [k for k in range(len(words)) if k not in protected]
            if len(words) <= min_words or not cut:
                return
            dropped.append(words.pop(cut[-1]))
            scene[i] = " ".join(words)

    for step in ("palette", "season", "details", 8, "view", "weather", 3):
        if fits():
            break
        if step == "details":
            while len(scene) > 3 and not fits():
                dropped.append(scene.pop())
        elif isinstance(step, int):
            shorten(step)
        else:
            drop_tail(step)
    return text(), dropped


class Ollama:
    """A minimal client for the local Ollama server: starts `ollama serve` if needed and pulls the model once."""

    def __init__(self, model: str, url: str = OLLAMA_URL):
        self.model, self.url, self.ready = model, url.rstrip("/"), False

    def _up(self) -> bool:
        try:
            return requests.get(f"{self.url}/api/version", timeout=2).ok
        except requests.RequestException:
            return False

    def _start(self) -> None:
        exe = shutil.which("ollama") or str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs/Ollama/ollama.exe")
        if not Path(exe).exists():
            raise RuntimeError("Ollama is not installed (see ollama.com)")
        log.info("starting Ollama: %s serve", exe)
        # A hidden console, not none: with DETACHED_PROCESS the server has no console, so every runner it
        # launches (model loads, GPU checks) opens a visible terminal window; with CREATE_NO_WINDOW they share
        # the server's invisible one. The server still outlives the screensaver either way.
        flags = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        subprocess.Popen([exe, "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, creationflags=flags, close_fds=True)
        for _ in range(60):
            time.sleep(0.5)
            if self._up():
                return
        raise RuntimeError("Ollama did not start")

    def _prepare(self) -> None:
        if self.ready:
            return
        if not self._up():
            self._start()
        r = requests.post(f"{self.url}/api/show", json={"model": self.model}, timeout=TIMEOUT)
        if r.status_code == 404:
            log.info("pulling %s through Ollama (one time)", self.model)
            requests.post(f"{self.url}/api/pull", json={"model": self.model, "stream": False}, timeout=3600).raise_for_status()
        else:
            r.raise_for_status()
        self.ready = True

    def generate(self, prompt: str, images: list | None = None, **options) -> str:
        """One answer from the model; `images` (PIL) are shown to it when it has vision."""
        self._prepare()
        body = {"model": self.model, "prompt": prompt, "stream": False, "think": False, "keep_alive": "15m",
                "options": {"temperature": 0.8, "top_p": 0.95, "num_predict": 120, "num_ctx": 2048, **options}}
        if images:
            body["images"] = [_jpeg_b64(im) for im in images]
            body["options"]["num_ctx"] = max(body["options"]["num_ctx"], 4096)  # room for the image tokens
        r = requests.post(f"{self.url}/api/generate", json=body, timeout=120)
        if r.status_code == 400 and "think" in r.text:  # a model without a thinking switch
            body.pop("think")
            r = requests.post(f"{self.url}/api/generate", json=body, timeout=120)
        if not r.ok:
            self.ready = False
            raise RuntimeError(f"Ollama: {r.status_code} {r.text[:200]}")
        return r.json().get("response", "")


def _jpeg_b64(image, longest: int = 1024) -> str:
    im = image.convert("RGB")
    im.thumbnail((longest, longest))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii")


TEXT_QUESTION = ("Look carefully at the whole image, including the corners and edges. Is there any visible text "
                 "in it: letters, words, numbers, writing, a watermark, a logo, a caption or a signature? "
                 "Answer with one word: yes or no.")


WORD_QUESTION = ("Look at the whole image. Does it show {word} in any form: the thing itself, a toy, picture, "
                 "statue, pattern or costume of it? Answer with one word: yes or no.")


class VisionCheck:
    """A yes/no question about an image to the local vision LLM: True / False, or None when it can't tell.
    If Ollama fails, the check stays off for 5 minutes instead of stalling every frame."""

    def __init__(self, model: str = DEFAULT_LLM, url: str = OLLAMA_URL):
        self.llm, self.down_until = Ollama(model, url), 0.0

    def ask(self, image, question: str) -> bool | None:
        if time.time() < self.down_until:
            return None
        try:
            answer = self.llm.generate(question, images=[image], temperature=0, num_predict=4)
        except (requests.RequestException, RuntimeError, OSError) as e:
            log.warning("%s unavailable for 5 minutes: %s", type(self).__name__, e)
            self.down_until = time.time() + 300
            return None
        word = answer.strip().lower()
        return True if word.startswith("yes") else False if word.startswith("no") else None


class TextCheck(VisionCheck):
    """Does a finished image show text? (SD 1.5 adds fake watermarks and lettering that no negative prompt
    reliably stops.)"""

    def __init__(self, model: str = DEFAULT_LLM, url: str = OLLAMA_URL, question: str = TEXT_QUESTION):
        super().__init__(model, url)
        self.question = question

    def __call__(self, image) -> bool | None:
        return self.ask(image, self.question)


class WordCheck(VisionCheck):
    """Which of the random words an image shows. SD 1.5 often draws the setting and drops small, odd or
    abstract subjects: on 40 night frames the vision LLM saw only a third of the words."""

    def __call__(self, image, words: list[str]) -> list[str] | None:
        """The words it sees; None when the check is unavailable."""
        seen = []
        for w in words:
            answer = self.ask(image, WORD_QUESTION.format(word=w.replace("-", " ")))
            if answer is None:
                return None
            if answer:
                seen.append(w)
        return seen


class WordsSource:
    """Three random words, a Ghent setting and the real weather: a local LLM makes them one scene, then
    a viewpoint, a style and sometimes a palette from scene_parts.txt finish the prompt."""

    def __init__(self, weather: WeatherSource, model: str = DEFAULT_LLM, url: str = OLLAMA_URL, n_words: int = 3):
        self.weather, self.n_words = weather, max(1, min(5, n_words))
        self.llm = Ollama(model, url)
        self.words = ShuffleBag(load_words(), "words:words")
        self.parts = {k: ShuffleBag(v, f"words:{k}") for k, v in load_parts().items()}
        self.down_until = 0.0

    def _part(self, name: str) -> str:
        bag = self.parts.get(name)
        return bag.draw() if bag and random.random() < PART_CHANCE.get(name, 1.0) else ""

    def _scene(self, request: str, template: str, words: list[str], own: str) -> tuple[list[str], list[dict]]:
        """Up to three tries for a scene that uses all the words and copies nothing from the examples.
        A small model often drops a word: the best try is kept and missing words go in as plain tags."""
        best, best_used, tries = None, [], []
        for _ in range(3):
            raw = self.llm.generate(request)
            tags = strip_time(clean_llm_prompt(raw))
            text = ", ".join(tags).lower()
            used = [w for w in words if _stem(w) in text]
            copied = copied_from_examples(text, template, own)
            tries.append({"output": raw.strip(), "words_used": len(used), "copied": copied})
            if not copied and len(text.split()) >= 8 and (best is None or len(used) > len(best_used)):
                best, best_used = tags, used
            if best is not None and len(best_used) == len(words):
                break
        if best is None:
            return [], tries
        missing = [w.replace("-", " ") for w in words if w not in best_used]
        return best[:1] + missing + best[1:], tries  # right after the setting, so trimming can't drop them

    def idea(self) -> Idea | None:
        if time.time() < self.down_until:  # Ollama failed recently: don't stall every frame on it
            return None
        words: list[str] = []
        while len(words) < self.n_words:
            if (w := self.words.draw()) not in words:
                words.append(w)
        setting, situation = self._part("settings") or "a street", self._part("situations")
        view, style, palette = self._part("viewpoints"), self._part("styles"), self._part("palettes")
        location, weather, details = self.weather.describe()
        values = {"location": location, "weather": weather, "setting": setting, "situation": situation or "none",
                  "words": ", ".join(words), **{f"word{i + 1}": w for i, w in enumerate(words)}}
        template = LLM_PROMPT_PATH.read_text(encoding="utf-8")  # re-read each time: edits show on the next image
        request = re.sub(r"\{(\w+)\}", lambda m: values.get(m.group(1), m.group(0)), template)
        t0 = time.perf_counter()
        try:
            scene, tries = self._scene(request, template, words, f"{setting} {situation} {location}")
        except (requests.RequestException, RuntimeError, OSError) as e:
            log.warning("local LLM unavailable, other sources for 5 minutes: %s", e)
            self.down_until = time.time() + 300
            return None
        if not scene:
            log.info("LLM gave no usable scene for %s: %s", words, tries)
            return None
        key = [w for w in _tokens(setting) if len(w) > 3 and w not in ("with", "from", "into", "near", "along", "under")]
        if key and not any(w in ", ".join(scene).lower() for w in key):  # the model lost the setting: put it back
            scene.insert(1, f"{setting} in {location}")  # after the main subjects, which the scene starts with
        if location.lower() not in ", ".join(scene).lower():
            scene[0] = f"{scene[0]} in {location}"
        tod, code, temp = details.get("time_of_day", ""), details.get("weather_code"), details.get("temperature_c")
        if code is None:  # offline / no location: time of day and season only
            sky = ""
        elif tod == "night" and code in (0, 1):
            sky = "starry sky, moonlight"
        else:  # the full phrase ("in light rain, wet reflections") shows far better than the bare word
            sky = WMO.get(code, ("", ""))[1]
        if temp is not None and (temp < 2 or temp > 29):  # only temperatures you can see: frost, ice, heat haze
            sky = ", ".join(x for x in (sky, temp_phrase(temp)) if x)
        tail = {"when": WHEN_TAGS.get(tod, tod), "weather": sky, "season": details.get("season", ""),
                "view": view, "style": style, "palette": palette}
        prompt, dropped = fit_prompt(scene, tail, keep=location)
        origin = {"words": words, "setting": setting, "situation": situation or None, "viewpoint": view or None,
                  "style": style, "palette": palette or None, "location": location, "weather": weather, **details,
                  "llm": self.llm.model, "llm_scene": ", ".join(scene), "llm_tries": tries,
                  "llm_seconds": round(time.perf_counter() - t0, 1), "llm_template": LLM_PROMPT_PATH.name,
                  "trimmed_to_fit": dropped or None}
        if temp is not None:  # caption: the words, and under them the weather as the weather source shows it
            now = f"{temp:.0f} °C · {details.get('conditions', '')} · {tod}"
        else:
            now = f"{tod} · {details['local_time'][11:16]}"
        # a word can't also be negated ("letter" vs "letters", "frame"): leave those terms out for this image
        negative = ", ".join(t for t in NEG_BASE.split(", ") if not any(_stem(w) in t for w in words))
        negative = hourlight.add_negative(negative, hourlight.light_for(details), words, _stem)
        count = ["One word", "Two words", "Three words", "Four words", "Five words"][len(words) - 1]
        return Idea(prompt, negative, f"{count} · right now in {location}", f"{' · '.join(words)}\n{now}",
                    "words", origin)


# --------------------------------------------------------------------------- news


NEWS_STYLES = [
    "surreal oil painting", "editorial illustration", "symbolic watercolor painting", "woodcut print",
    "art deco poster illustration", "gouache painting", "collage-style illustration", "linocut print",
    "magic realism painting",
]


@dataclass
class Headline:
    title: str
    source: str
    published: datetime | None = None
    link: str | None = None
    feed: str | None = None


class NewsSource:
    def __init__(self, feeds: list[str], skip: re.Pattern | None, names: NameStripper):
        self.feeds = [f for f in feeds if f.strip()]
        self.skip = skip
        self.names = names
        self.items: list[Headline] = []
        self.fetched = 0.0
        self.seen: set[str] = set()

    def _refresh(self) -> None:
        if self.items and time.time() - self.fetched < 1800:
            return
        items: list[Headline] = []
        for url in self.feeds:
            try:
                root = ET.fromstring(_get(url).content)
            except (requests.RequestException, ET.ParseError):
                continue
            items += self._parse(root, url)
        if items:
            self.items, self.fetched = items, time.time()

    @staticmethod
    def _parse(root: ET.Element, url: str) -> list[Headline]:
        atom = "{http://www.w3.org/2005/Atom}"
        if root.tag == f"{atom}feed":  # Atom
            source = (root.findtext(f"{atom}title") or url).strip()
            entries = [(e.findtext(f"{atom}title"), e.findtext(f"{atom}updated"),
                        (e.find(f"{atom}link").get("href") if e.find(f"{atom}link") is not None else None))
                       for e in root.iter(f"{atom}entry")]
        else:  # RSS 2.0
            channel = root.find("channel")
            source = ((channel.findtext("title") if channel is not None else None) or url).strip()
            entries = [(i.findtext("title"), i.findtext("pubDate"), i.findtext("link")) for i in root.iter("item")]
        # "World news | The Guardian" → "The Guardian", "NPR Topics: News" → "NPR"
        generic = {"world", "world news", "news", "top stories", "topics", "home", "latest", "headlines"}
        parts = [p.strip() for p in re.split(r"\s+[|\-–]\s+|:\s+", source) if p.strip()]
        named = [re.sub(r"\s+topics$", "", p, flags=re.I) for p in parts if p.lower() not in generic]
        source = named[-1] if named else source
        out = []
        for title, stamp, link in entries[:25]:
            if not title:
                continue
            try:
                published = parsedate_to_datetime(stamp) if stamp and "," in stamp else (
                    datetime.fromisoformat(stamp.replace("Z", "+00:00")) if stamp else None)
            except (TypeError, ValueError):
                published = None
            out.append(Headline(" ".join(title.split()), source, published, (link or "").strip() or None, url))
        return out

    def idea(self) -> Idea | None:
        self._refresh()
        fresh = [h for h in self.items if h.title not in self.seen and not (self.skip and self.skip.search(h.title))]
        if not fresh:
            if self.items:  # everything shown once: start over
                self.seen.clear()
            return None
        h = random.choice(fresh[:40])
        self.seen.add(h.title)
        subject = self.names(h.title).rstrip(" .;,")
        style = random.choice(NEWS_STYLES)
        prompt = f"{style}, an allegorical scene about: {subject}, symbolic, evocative, atmospheric"
        origin = {"headline": h.title, "source": h.source, "link": h.link, "feed": h.feed,
                  "published": h.published.isoformat() if h.published else None, "headline_in_prompt": subject}
        return Idea(prompt, ", ".join((NEG_BASE, NEG_PEOPLE, NEG_GRIM)), f"In the news · {h.source}", h.title, "news",
                    origin)


# --------------------------------------------------------------------------- on this day


def era_style(year: int) -> str:
    """Painting/illustration styles only: nothing should pass for a photo of a real event."""
    if year < 500:
        return "ancient Roman mosaic"
    if year < 1400:
        return "medieval illuminated manuscript illustration"
    if year < 1600:
        return "Renaissance oil painting"
    if year < 1780:
        return "Baroque oil painting"
    if year < 1880:
        return "19th-century romantic history painting"
    if year < 1920:
        return "Art Nouveau illustration"
    if year < 1950:
        return "1930s art deco poster illustration"
    if year < 1975:
        return "1960s magazine illustration"
    if year < 2000:
        return "1980s airbrush illustration"
    return "contemporary digital painting"


class OnThisDaySource:
    def __init__(self, skip: re.Pattern | None, names: NameStripper):
        self.skip = skip
        self.names = names
        self.day: date | None = None
        self.events: list[dict] = []
        self.used: set[str] = set()

    def _refresh(self) -> None:
        today = date.today()
        if self.day == today and self.events:
            return
        url = f"https://en.wikipedia.org/api/rest_v1/feed/onthisday/selected/{today:%m}/{today:%d}"
        self.events = _get(url).json().get("selected", [])
        self.day, self.used = today, set()

    def idea(self) -> Idea | None:
        try:
            self._refresh()
        except requests.RequestException:
            return None
        fresh = [e for e in self.events if e.get("text") and e["text"] not in self.used
                 and not (self.skip and self.skip.search(e["text"]))]
        if not fresh:
            self.used.clear()
            return None
        e = random.choice(fresh)
        self.used.add(e["text"])
        year = int(e.get("year", 2000))
        # "Syrian civil war: The forces launched …" → drop the short topic prefix; and trailing punctuation
        text = e["text"].split(": ", 1)[-1] if len(e["text"].split(": ", 1)[0]) < 40 else e["text"]
        subject = self.names(text).rstrip(" .;,")
        prompt = f"{era_style(year)}, historical scene: {subject}, around the year {year}, detailed"
        when = f"{-year} BC" if year < 0 else str(year)
        pages = [{"title": p.get("titles", {}).get("normalized") or p.get("title"),
                  "url": p.get("content_urls", {}).get("desktop", {}).get("page")} for p in e.get("pages", [])]
        origin = {"date": f"{self.day:%m-%d}", "year": year, "event": e["text"], "event_in_prompt": subject,
                  "wikipedia_pages": pages, "provider": "Wikipedia: On this day (selected anniversaries)"}
        return Idea(prompt, ", ".join((NEG_BASE, NEG_PEOPLE, NEG_GRIM)), f"On this day · {when}", e["text"], "history",
                    origin)


# --------------------------------------------------------------------------- rotation


class Sources:
    """Takes turns between the enabled sources; falls back to the weather scene, which always works."""

    def __init__(self, cfg: dict):
        place = cfg.get("place")
        self.weather = WeatherSource(Place(**place) if place else None)
        skip = _skip_pattern(cfg.get("skip_words", DEFAULT_SKIP_WORDS)) if cfg.get("skip_distressing", True) else None
        self.enabled = [s for s in ("weather", "words", "news", "history")
                        if cfg.get(f"use_{s}", s != "words")] or ["weather"]  # words (needs Ollama) is opt-in
        self.names = NameStripper() if {"news", "history"} & set(self.enabled) else None
        self.words = WordsSource(self.weather, cfg.get("llm_model") or DEFAULT_LLM, cfg.get("ollama_url") or OLLAMA_URL,
                                 int(cfg.get("llm_words", 3))) \
            if "words" in self.enabled else None
        self.news = NewsSource(cfg.get("feeds", DEFAULT_FEEDS), skip, self.names) if "news" in self.enabled else None
        self.history = OnThisDaySource(skip, self.names) if "history" in self.enabled else None
        self.turn = random.randrange(len(self.enabled))

    def next(self) -> Idea:
        for _ in range(len(self.enabled)):
            name = self.enabled[self.turn % len(self.enabled)]
            self.turn += 1
            try:
                idea = {"weather": self.weather.idea, "words": lambda: self.words.idea(),
                        "news": lambda: self.news.idea(), "history": lambda: self.history.idea()}[name]()
            except Exception:
                log.exception("the %s source failed", name)
                idea = None
            if idea is not None:
                return idea
        return self.weather.idea()


if __name__ == "__main__":  # quick look at what the sources produce
    import sys

    logging.basicConfig(level=logging.INFO)
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    city = args[0] if args else None
    p = geocode(city) if city else None
    cfg = {"place": vars(p) if p else None}
    if "--words" in sys.argv:  # only the three-words source: uv run realworld.py Ghent --words
        cfg.update(use_weather=False, use_news=False, use_history=False, use_words=True)
    src = Sources(cfg)
    for _ in range(6):
        i = src.next()
        print(json.dumps(vars(i), ensure_ascii=False, indent=1))
