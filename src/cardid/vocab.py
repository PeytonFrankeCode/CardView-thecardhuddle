"""Football-card domain vocabulary.

Everything the parsers know about brands, sets, parallels, teams and graders
lives here. Each canonical value maps to the aliases and misspellings that show
up on card faces and in eBay titles ("prism" for "prizm", "ud" for "upper deck").

Override or extend any table at runtime by pointing ``CARDID_VOCAB_PATH`` at a
JSON file with the same shape; entries there are merged over these defaults, so
a new set can be added without editing code.
"""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache

# --- brands -----------------------------------------------------------------

BRANDS: dict[str, list[str]] = {
    "panini": ["panini", "pannini"],
    "topps": ["topps", "tops"],
    "upper deck": ["upper deck", "ud", "upperdeck"],
    "donruss": ["donruss", "donrus", "dunruss"],
    "leaf": ["leaf"],
    "sage": ["sage"],
    "wild card": ["wild card", "wildcard"],
    "bowman": ["bowman"],
    "fleer": ["fleer"],
    "score": ["score"],
    "pro set": ["pro set", "proset"],
    "playoff": ["playoff"],
    "press pass": ["press pass"],
}

# --- sets -------------------------------------------------------------------
# canonical set name -> (owning brand or None, aliases)

SETS: dict[str, tuple[str | None, list[str]]] = {
    # Panini flagship
    "prizm": ("panini", ["prizm", "prism", "prizms", "panini prizm"]),
    "optic": ("panini", ["optic", "donruss optic", "optics"]),
    "select": ("panini", ["select"]),
    "mosaic": ("panini", ["mosaic", "mozaic"]),
    "contenders": ("panini", ["contenders", "contender"]),
    "absolute": ("panini", ["absolute", "absolute memorabilia"]),
    "certified": ("panini", ["certified"]),
    "immaculate": ("panini", ["immaculate", "immaculate collection"]),
    "national treasures": ("panini", ["national treasures", "nt"]),
    "flawless": ("panini", ["flawless"]),
    "obsidian": ("panini", ["obsidian"]),
    "phoenix": ("panini", ["phoenix"]),
    "spectra": ("panini", ["spectra"]),
    "illusions": ("panini", ["illusions", "illusion"]),
    "chronicles": ("panini", ["chronicles"]),
    "prestige": ("panini", ["prestige"]),
    "playbook": ("panini", ["playbook"]),
    "origins": ("panini", ["origins"]),
    "impeccable": ("panini", ["impeccable"]),
    "limited": ("panini", ["limited"]),
    "rookies and stars": ("panini", ["rookies and stars", "rookies & stars", "r&s"]),
    "legacy": ("panini", ["legacy"]),
    "xr": ("panini", ["xr"]),
    "zenith": ("panini", ["zenith"]),
    "unparalleled": ("panini", ["unparalleled"]),
    "gold standard": ("panini", ["gold standard"]),
    "encased": ("panini", ["encased"]),
    "noir": ("panini", ["noir"]),
    "one": ("panini", ["panini one"]),
    "black": ("panini", ["panini black"]),
    "clearly donruss": ("panini", ["clearly donruss", "clearly"]),
    "luminance": ("panini", ["luminance"]),
    "classics": ("panini", ["classics"]),
    "crown royale": ("panini", ["crown royale"]),
    "elite extra edition": ("panini", ["elite extra edition", "eee"]),
    "vertex": ("panini", ["vertex"]),
    "titan": ("panini", ["titan"]),
    "elite": ("panini", ["elite", "donruss elite"]),
    "instant": ("panini", ["instant", "panini instant"]),
    "score": ("panini", ["score"]),
    "donruss": ("panini", ["donruss"]),
    # Topps
    "chrome": ("topps", ["chrome", "topps chrome"]),
    "finest": ("topps", ["finest"]),
    "fire": ("topps", ["fire"]),
    "gallery": ("topps", ["gallery"]),
    "heritage": ("topps", ["heritage"]),
    "stadium club": ("topps", ["stadium club"]),
    "museum collection": ("topps", ["museum collection", "museum"]),
    "allen and ginter": ("topps", ["allen and ginter", "allen & ginter", "a&g"]),
    "bowman chrome": ("bowman", ["bowman chrome"]),
    "inception": ("topps", ["inception"]),
    "dynasty": ("topps", ["dynasty"]),
    # Upper Deck
    "sp authentic": ("upper deck", ["sp authentic", "spa"]),
    "spx": ("upper deck", ["spx"]),
    "exquisite": ("upper deck", ["exquisite", "exquisite collection"]),
    "ultimate collection": ("upper deck", ["ultimate collection"]),
}

# --- parallels --------------------------------------------------------------

PARALLELS: dict[str, list[str]] = {
    "base": ["base"],
    "silver": ["silver", "silver prizm", "silver prism"],
    "gold": ["gold"],
    "green": ["green"],
    "blue": ["blue"],
    "red": ["red"],
    "orange": ["orange"],
    "purple": ["purple"],
    "pink": ["pink"],
    "black": ["black"],
    "white sparkle": ["white sparkle", "sparkle"],
    "camo": ["camo", "camouflage"],
    "disco": ["disco"],
    "hyper": ["hyper"],
    "ice": ["ice"],
    "mojo": ["mojo"],
    "pulsar": ["pulsar"],
    "shimmer": ["shimmer"],
    "wave": ["wave"],
    "cracked ice": ["cracked ice"],
    "fast break": ["fast break", "fastbreak"],
    "scope": ["scope"],
    "tie dye": ["tie dye", "tie-dye", "tiedye"],
    "snakeskin": ["snakeskin", "snake skin"],
    "choice": ["choice"],
    "no huddle": ["no huddle"],
    "refractor": ["refractor", "refactor", "reffy"],
    "x-fractor": ["x-fractor", "xfractor", "x fractor"],
    "superfractor": ["superfractor", "super fractor"],
    "atomic": ["atomic"],
    "speckle": ["speckle"],
    "lava": ["lava"],
    "nebula": ["nebula"],
    "galactic": ["galactic"],
    "holo": ["holo", "hologram", "holographic"],
    "prizm": ["prizm parallel"],
}

# Insert/subset names that are NOT parallels but are often adjacent to them.
SUBSETS: dict[str, list[str]] = {
    "rated rookie": ["rated rookie", "rated rookies", "rr"],
    "rookie ticket": ["rookie ticket", "rookie ticket auto", "ticket"],
    "downtown": ["downtown", "downtown!"],
    "kaboom": ["kaboom", "kabooms"],
    "color blast": ["color blast", "colorblast"],
    "hobby": ["hobby"],
    "night moves": ["night moves"],
    "stained glass": ["stained glass"],
    "zoom": ["zoom"],
    "genesis": ["genesis"],
}

GRADERS: dict[str, list[str]] = {
    "psa": ["psa"],
    "bgs": ["bgs", "beckett", "bgs beckett"],
    "sgc": ["sgc"],
    "cgc": ["cgc"],
    "csg": ["csg"],
    "hga": ["hga"],
}

TEAMS: dict[str, list[str]] = {
    "arizona cardinals": ["arizona cardinals", "cardinals", "ari"],
    "atlanta falcons": ["atlanta falcons", "falcons", "atl"],
    "baltimore ravens": ["baltimore ravens", "ravens", "bal"],
    "buffalo bills": ["buffalo bills", "bills", "buf"],
    "carolina panthers": ["carolina panthers", "panthers", "car"],
    "chicago bears": ["chicago bears", "bears", "chi"],
    "cincinnati bengals": ["cincinnati bengals", "bengals", "cin"],
    "cleveland browns": ["cleveland browns", "browns", "cle"],
    "dallas cowboys": ["dallas cowboys", "cowboys", "dal"],
    "denver broncos": ["denver broncos", "broncos", "den"],
    "detroit lions": ["detroit lions", "lions", "det"],
    "green bay packers": ["green bay packers", "packers", "gb"],
    "houston texans": ["houston texans", "texans", "hou"],
    "indianapolis colts": ["indianapolis colts", "colts", "ind"],
    "jacksonville jaguars": ["jacksonville jaguars", "jaguars", "jags", "jax"],
    "kansas city chiefs": ["kansas city chiefs", "chiefs", "kc"],
    "las vegas raiders": ["las vegas raiders", "oakland raiders", "raiders", "lv"],
    "los angeles chargers": ["los angeles chargers", "chargers", "lac"],
    "los angeles rams": ["los angeles rams", "rams", "lar"],
    "miami dolphins": ["miami dolphins", "dolphins", "mia"],
    "minnesota vikings": ["minnesota vikings", "vikings", "min"],
    "new england patriots": ["new england patriots", "patriots", "ne"],
    "new orleans saints": ["new orleans saints", "saints", "no"],
    "new york giants": ["new york giants", "giants", "nyg"],
    "new york jets": ["new york jets", "jets", "nyj"],
    "philadelphia eagles": ["philadelphia eagles", "eagles", "phi"],
    "pittsburgh steelers": ["pittsburgh steelers", "steelers", "pit"],
    "san francisco 49ers": ["san francisco 49ers", "49ers", "niners", "sf"],
    "seattle seahawks": ["seattle seahawks", "seahawks", "sea"],
    "tampa bay buccaneers": ["tampa bay buccaneers", "buccaneers", "bucs", "tb"],
    "tennessee titans": ["tennessee titans", "titans", "ten"],
    "washington commanders": [
        "washington commanders",
        "commanders",
        "washington football team",
        "redskins",
        "was",
    ],
}

# Marketing filler that carries no identifying signal. Stripped before the
# leftover text is treated as a player name.
NOISE_TOKENS: set[str] = {
    "mint", "nm", "near", "gem", "gem mint", "sharp", "clean", "centered",
    "hot", "invest", "investment", "rare", "scarce", "look", "wow", "nice",
    "beautiful", "stunning", "gorgeous", "fire", "steal", "deal", "sale",
    "free", "shipping", "ship", "fast", "lot", "read", "l@@k", "pack",
    "fresh", "pulled", "pull", "case", "hit", "sp", "ssp", "hof", "mvp",
    "nfl", "football", "card", "cards", "trading", "collectible", "psa",
    "graded", "ungraded", "raw", "slab", "slabbed", "pop", "low", "qty",
    "the", "and", "with", "for", "from", "his", "her", "new", "used",
    "combined", "bundle", "sports", "collection",
}

# Tokens that mark the card as a rookie / autograph / relic.
ROOKIE_TOKENS = {"rc", "rookie", "rookies", "yg", "young guns", "1st", "first"}
AUTO_TOKENS = {"auto", "autograph", "autographed", "signed", "signature", "sig", "on card"}
PATCH_TOKENS = {"patch", "relic", "jersey", "mem", "memorabilia", "swatch", "rpa", "gu"}

_VOCAB_TABLES = {
    "BRANDS": BRANDS,
    "SETS": SETS,
    "PARALLELS": PARALLELS,
    "SUBSETS": SUBSETS,
    "GRADERS": GRADERS,
    "TEAMS": TEAMS,
}


def _load_overrides() -> None:
    """Merge user-supplied vocabulary from ``CARDID_VOCAB_PATH`` if present."""
    path = os.environ.get("CARDID_VOCAB_PATH")
    if not path or not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    for table_name, entries in data.items():
        table = _VOCAB_TABLES.get(table_name.upper())
        if table is None or not isinstance(entries, dict):
            continue
        for canonical, value in entries.items():
            if table_name.upper() == "SETS" and isinstance(value, list):
                # Allow the simpler ["alias", ...] form for sets.
                table[canonical] = (None, value)
            else:
                table[canonical] = value


_load_overrides()


def _alias_index(table: dict[str, list[str]]) -> list[tuple[str, str]]:
    """Flatten a table into (alias, canonical) pairs, longest alias first.

    Longest-first ordering is what makes "bowman chrome" win over "bowman" and
    "cracked ice" win over "ice" during greedy scanning.
    """
    pairs = [
        (alias.lower(), canonical)
        for canonical, aliases in table.items()
        for alias in aliases
    ]
    pairs.sort(key=lambda pair: (-len(pair[0]), pair[0]))
    return pairs


@lru_cache(maxsize=1)
def brand_index() -> list[tuple[str, str]]:
    return _alias_index(BRANDS)


@lru_cache(maxsize=1)
def set_index() -> list[tuple[str, str]]:
    return _alias_index({k: v[1] for k, v in SETS.items()})


@lru_cache(maxsize=1)
def parallel_index() -> list[tuple[str, str]]:
    return _alias_index(PARALLELS)


@lru_cache(maxsize=1)
def subset_index() -> list[tuple[str, str]]:
    return _alias_index(SUBSETS)


@lru_cache(maxsize=1)
def grader_index() -> list[tuple[str, str]]:
    return _alias_index(GRADERS)


@lru_cache(maxsize=1)
def team_index() -> list[tuple[str, str]]:
    return _alias_index(TEAMS)


def brand_for_set(set_name: str | None) -> str | None:
    """The brand that owns a set, used to fill in an unstated brand."""
    if not set_name:
        return None
    entry = SETS.get(set_name)
    return entry[0] if entry else None


_WS = re.compile(r"\s+")
_PUNCT = re.compile("[^-\\w\\s#/&\x00]+")


_DECIMAL = re.compile(r"(?<=\d)\.(?=\d)")
_DECIMAL_SENTINEL = "\x00"


def normalize_text(text: str) -> str:
    """Lowercase, strip punctuation noise, and collapse whitespace.

    Decimal points inside numbers are protected so a "BGS 9.5" grade survives
    punctuation stripping, while sentence punctuation is still removed.
    """
    if not text:
        return ""
    text = text.lower().replace("&amp;", "&")
    text = text.replace("_", " ")
    text = _DECIMAL.sub(_DECIMAL_SENTINEL, text)
    text = _PUNCT.sub(" ", text)
    text = text.replace(_DECIMAL_SENTINEL, ".")
    return _WS.sub(" ", text).strip()
