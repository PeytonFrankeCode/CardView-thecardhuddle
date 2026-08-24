"""Turn free text into :class:`CardAttributes`.

The same parser handles both an eBay listing title and the text OCR pulls off a
card face, because after normalization they contain the same kinds of tokens in
a different order. Parsing is span-consuming: once "bowman chrome" is claimed as
the set, those characters cannot be re-read as a player name.
"""

from __future__ import annotations

import re

from ..models import CardAttributes, Extraction, Source
from ..vocab import (
    AUTO_TOKENS,
    NOISE_TOKENS,
    PATCH_TOKENS,
    ROOKIE_TOKENS,
    brand_for_set,
    brand_index,
    grader_index,
    normalize_text,
    parallel_index,
    set_index,
    subset_index,
    team_index,
)

Span = tuple[int, int]

# 1948-2049, optionally a season range like "2017-18" or "2017/18".
_YEAR_RE = re.compile(r"(?<!\d)(19[4-9]\d|20[0-4]\d)(?:\s*[-/]\s*\d{2,4})?(?!\d)")
# "#269", "# 269", "no. 269", "card 269"; number may be alphanumeric (#BS-PM).
_NUMBER_RE = re.compile(
    r"(?:#\s*|\bno\.?\s*|\bcard\s+(?:no\.?\s*)?)([a-z]{0,4}-?\d{1,4}[a-z]?|[a-z]{2,5}-[a-z0-9]{1,5})(?!\d)"
)
# "12/99", "/99", "1/1"
_SERIAL_RE = re.compile(r"(?<![\d/])(\d{1,5})?\s*/\s*(\d{1,5})(?![\d/])")
_ONE_OF_ONE_RE = re.compile(r"\b(?:1\s*of\s*1|one\s*of\s*one|1/1)\b")


def _boundary_re(alias: str) -> re.Pattern[str]:
    return re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")


_BOUNDARY_CACHE: dict[str, re.Pattern[str]] = {}


def _pattern(alias: str) -> re.Pattern[str]:
    pattern = _BOUNDARY_CACHE.get(alias)
    if pattern is None:
        pattern = _boundary_re(alias)
        _BOUNDARY_CACHE[alias] = pattern
    return pattern


def _overlaps(span: Span, consumed: list[Span]) -> bool:
    start, end = span
    return any(start < c_end and end > c_start for c_start, c_end in consumed)


def _scan(
    text: str, index: list[tuple[str, str]], consumed: list[Span]
) -> tuple[str | None, str | None]:
    """Find the first non-overlapping alias from ``index``.

    ``index`` is ordered longest-alias-first, so the most specific name wins.
    Returns ``(canonical, matched_alias)`` and records the consumed span.
    """
    for alias, canonical in index:
        for match in _pattern(alias).finditer(text):
            span = (match.start(), match.end())
            if _overlaps(span, consumed):
                continue
            consumed.append(span)
            return canonical, alias
    return None, None


def _strip_known_vocab(text: str) -> str:
    """Remove every remaining vocabulary alias so residue is name-like."""
    for index in (
        set_index(),
        brand_index(),
        parallel_index(),
        subset_index(),
        team_index(),
        grader_index(),
    ):
        for alias, _ in index:
            text = _pattern(alias).sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}

# Flag words describe the card, never the player, so they must not survive into
# the name residue.
_FLAG_TOKEN_RE = re.compile(
    r"(?<!\w)(?:"
    + "|".join(
        re.escape(token)
        for token in sorted(ROOKIE_TOKENS | AUTO_TOKENS | PATCH_TOKENS, key=len, reverse=True)
    )
    + r")(?!\w)"
)


def _extract_player(residue: str, player_index: list[tuple[str, str]] | None) -> str | None:
    """Pull a player name out of what is left after every known token is gone.

    When a catalog-derived ``player_index`` is supplied we scan for known names,
    which is far more reliable than trusting leftovers. Otherwise we fall back to
    the longest run of consecutive non-noise word tokens.
    """
    if player_index:
        for alias, canonical in player_index:
            if _pattern(alias).search(residue):
                return canonical

    best: list[str] = []
    current: list[str] = []
    for token in residue.split():
        is_wordy = token.isalpha() and len(token) > 1
        if is_wordy and token not in NOISE_TOKENS:
            current.append(token)
            continue
        if token in _NAME_SUFFIXES and current:
            current.append(token)
            continue
        if len(current) > len(best):
            best = current
        current = []
    if len(current) > len(best):
        best = current

    # A single leftover word is usually noise, not a name; two+ reads as a name.
    if len(best) >= 2:
        return " ".join(best)
    return None


def canonical_player(name: str | None) -> str:
    """Reduce a player name to a form that groups spellings of one person.

    Generational suffixes are dropped because "Patrick Mahomes" and "Patrick
    Mahomes II" are the same player, and a catalog that holds both splits every
    match between two rows and manufactures near-ties.
    """
    if not name:
        return ""
    words = [word for word in normalize_text(name).replace(".", " ").split()]
    while words and words[-1] in _NAME_SUFFIXES:
        words.pop()
    return " ".join(words)


def parse_text(
    text: str,
    source: Source = Source.TITLE,
    player_index: list[tuple[str, str]] | None = None,
) -> Extraction:
    """Parse one blob of text into an :class:`Extraction`."""
    raw = text or ""
    norm = normalize_text(raw)
    attrs = CardAttributes()
    consumed: list[Span] = []
    used: list[str] = []

    # --- grade: consumed before card numbers so "psa 10" is not read as #10 ---
    grader, grader_alias = _scan(norm, grader_index(), consumed)
    if grader:
        attrs.grader = grader
        used.append(grader_alias or grader)
        tail = norm[consumed[-1][1] : consumed[-1][1] + 8]
        grade_match = re.match(r"\s*(\d{1,2}(?:\.\d)?)", tail)
        if grade_match:
            value = float(grade_match.group(1))
            if 1.0 <= value <= 10.0:
                attrs.grade = value
                start = consumed[-1][1]
                consumed.append((start, start + grade_match.end()))

    # --- one-of-one and serial numbering ---
    if _ONE_OF_ONE_RE.search(norm):
        attrs.is_one_of_one = True
        attrs.print_run = 1
        match = _ONE_OF_ONE_RE.search(norm)
        if match:
            consumed.append((match.start(), match.end()))
    for match in _SERIAL_RE.finditer(norm):
        span = (match.start(), match.end())
        if _overlaps(span, consumed):
            continue
        numerator, denominator = match.group(1), match.group(2)
        run = int(denominator)
        if run < 1 or run > 100000:
            continue
        attrs.print_run = run
        attrs.serial = f"{numerator}/{denominator}" if numerator else f"/{denominator}"
        if run == 1:
            attrs.is_one_of_one = True
        consumed.append(span)
        break

    # --- year ---
    year_match = _YEAR_RE.search(norm)
    if year_match:
        attrs.year = int(year_match.group(1))
        consumed.append((year_match.start(), year_match.end()))

    # --- card number ---
    for match in _NUMBER_RE.finditer(norm):
        span = (match.start(), match.end())
        if _overlaps(span, consumed):
            continue
        attrs.card_number = match.group(1).upper().strip("-")
        consumed.append(span)
        break

    # --- vocabulary scans, most specific category first ---
    set_name, set_alias = _scan(norm, set_index(), consumed)
    if set_name:
        attrs.set_name = set_name
        used.append(set_alias or set_name)

    brand, brand_alias = _scan(norm, brand_index(), consumed)
    if brand:
        attrs.brand = brand
        used.append(brand_alias or brand)
    elif set_name:
        attrs.brand = brand_for_set(set_name)

    subset, subset_alias = _scan(norm, subset_index(), consumed)
    if subset:
        attrs.subset = subset
        used.append(subset_alias or subset)

    parallel, parallel_alias = _scan(norm, parallel_index(), consumed)
    if parallel and parallel != "base":
        attrs.parallel = parallel
        used.append(parallel_alias or parallel)

    team, team_alias = _scan(norm, team_index(), consumed)
    if team:
        attrs.team = team
        used.append(team_alias or team)

    # --- boolean flags ---
    tokens = set(norm.split())
    attrs.is_rookie = bool(tokens & ROOKIE_TOKENS)
    attrs.is_autograph = bool(tokens & AUTO_TOKENS) or "on card auto" in norm
    attrs.is_patch = bool(tokens & PATCH_TOKENS)

    # --- player name from the residue ---
    residue_chars = list(norm)
    for start, end in consumed:
        for i in range(start, min(end, len(residue_chars))):
            residue_chars[i] = " "
    residue = _strip_known_vocab("".join(residue_chars))
    residue = _FLAG_TOKEN_RE.sub(" ", residue)
    residue = re.sub(r"[\d#/.]+", " ", residue)
    attrs.player = _extract_player(residue, player_index)

    return Extraction(
        source=source,
        attributes=attrs,
        raw_text=raw,
        tokens_used=used,
        confidence=_extraction_confidence(attrs),
    )


def _extraction_confidence(attrs: CardAttributes) -> float:
    """How much of the identifying signal this extraction actually recovered."""
    weights = {
        "player": 0.35,
        "year": 0.15,
        "set_name": 0.20,
        "card_number": 0.20,
        "brand": 0.10,
    }
    return round(
        sum(weight for field, weight in weights.items() if getattr(attrs, field)), 4
    )
