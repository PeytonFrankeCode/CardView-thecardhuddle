# Football Card Identification

Identify a football card from a photo, an eBay listing title, or both, and get
back the card's full structured data — year, brand, set, player, card number,
parallel — plus a confidence score and a clear decision about whether that
answer is safe to use automatically.

Built for the job described: take unsorted sold-listing data, attach the correct
card to each row, at roughly 20,000 photos a day, callable from a website.

---

## Read this first: about "99% accuracy"

A single accuracy number is the wrong target, and chasing it directly will cost
you money. Here is the honest version.

**Top-1 accuracy from a raw eBay photo alone is realistically 85–93%**, not 99%.
Photos are angled, glare-covered, half-cropped, and often show a card sealed in
a slab. No system reads them perfectly.

**But that is not the number you actually need.** What matters for a sold-data
pipeline is: *of the rows we auto-filled, how many are right* — and that you can
push as high as you want, by declining to guess when the evidence is thin.

So this system reports one of three decisions:

| Decision | Meaning | What your site does |
|---|---|---|
| `auto_accept` | Confident. Two independent signals agree and one card fits clearly better than the rest. | Write it to the database. |
| `review` | Plausible but not certain — usually two parallels of the same card fit equally well. | Queue for a human; the API does this for you. |
| `reject` | Nothing usable was extracted. | Leave the row unsorted. |

That turns "99% accuracy" into something you can actually verify and tune:

> **99% precision on the auto-accepted slice, covering as much of the feed as
> that precision allows.**

On a feed that includes listing titles, expect roughly **90–95% of rows
auto-accepted at ~99% precision**, with the remainder going to review. Photos
with no title attached land lower — parallels are often genuinely
indistinguishable without the seller's own description.

`cardid calibrate` measures this on *your* data and tells you the exact
threshold to set. Do not take the numbers above on faith — measure them.

### The single biggest accuracy lever

**Feed the listing title in alongside the photo.** You already have it; you said
so. A title is usually a *stronger* signal than the image, and the two together
are far stronger than either alone, because they fail in uncorrelated ways —
OCR misreads foil, and sellers mistype. When they agree, confidence is very
high; when they disagree, that is exactly the row a human should see.

Every endpoint accepts `title` for this reason. Use it.

---

## Quick start

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .

# 1. Load a catalog — this is what identification maps onto.
cardid --catalog data/catalog.db import-catalog data/sample_catalog.csv

# 2. Identify something.
cardid --catalog data/catalog.db identify \
  --title "2017 Panini Prizm Patrick Mahomes II #269 RC Silver Prizm PSA 10"

# 3. Serve the API.
cardid --catalog data/catalog.db serve --port 8000
```

The ambiguous case behaves the way it should:

```
$ cardid identify --title "2017 Prizm Mahomes #269 RC"
decision   : review
confidence : 0.6
card       : 2017 Panini Prizm Patrick Mahomes II #269 RC
candidates:
   1.000  2017 Panini Prizm Patrick Mahomes II #269 RC
   0.957  2017 Panini Prizm Silver Patrick Mahomes II #269 RC
   0.957  2017 Panini Prizm Gold Patrick Mahomes II #269 RC /10
reasons:
  - near-tie: margin 0.043 below floor 0.080 — capped for review
```

The title never said which parallel it was, so three cards fit almost equally.
Guessing here is how a tracking database silently fills with wrong data.

---

## Calling it from your website

`POST /v1/identify` — one card, by URL and/or title:

```bash
curl -X POST https://your-host/v1/identify \
  -H 'Content-Type: application/json' -H 'X-API-Key: your-key' \
  -d '{"image_url": "https://i.ebayimg.com/images/g/abc/s-l1600.jpg",
       "title": "2017 Panini Prizm Patrick Mahomes II #269 RC Silver"}'
```

```json
{
  "decision": "auto_accept",
  "confidence": 0.98,
  "card_id": "prizm_2017_269_silver",
  "display_name": "2017 Panini Prizm Silver Patrick Mahomes II #269 RC",
  "attributes": {"year": 2017, "set_name": "prizm", "player": "patrick mahomes ii",
                 "card_number": "269", "parallel": "silver"},
  "candidates": [...]
}
```

Your site only needs to branch on `decision`.

| Endpoint | Use |
|---|---|
| `POST /v1/identify` | One card, by image URL and/or title |
| `POST /v1/identify/upload` | One card, direct file upload (multipart) |
| `POST /v1/identify/batch` | Up to 500 at a time — use this for backfills |
| `GET /v1/review` | Items awaiting a human, most-confident first |
| `POST /v1/review/{id}/resolve` | Record the right answer; becomes a training label |
| `GET /healthz` | Liveness, catalog size, active OCR backend |

Set `CARDID_API_KEY` to require an `X-API-Key` header. `/healthz` stays open so
load balancers can probe it. Interactive docs at `/docs`.

For a bulk backfill, prefer the CLI — it streams, resumes, and does not time out:

```bash
cardid batch sold_listings.csv identified.csv --concurrency 8
```

Re-running skips rows already in the output, so an interrupted job picks up
where it stopped.

---

## Capacity

20,000 photos/day is **~0.25 per second** averaged out. That is a small
workload; a single 4-core box handles it with room to spare. The design targets
robustness over raw throughput:

- **Result caching** by content hash and perceptual hash. eBay reuses photos
  heavily, so a large share of a real feed never runs OCR at all.
- **Bounded concurrency** on image fetching, so a batch cannot exhaust sockets.
- **Per-item failure isolation** — one dead URL never fails a 20,000-row job.
- **CPU work off the event loop**, so downloads and matching overlap.

Scale by running more `uvicorn` workers; the SQLite catalog is read-mostly and
safe to share. If the catalog passes a few million rows or you need multi-node
writes, move it to Postgres — `CatalogStore` is the only class that would change.

---

## How it works

```
photo ──► detect & rectify ──► slab label ──┐
                └──► card face ──► OCR ─────┤
                                             ├──► parse ──► fuse ──► catalog
listing title ───────────────────────────────┘                 │      match
                                                               ▼
                                              confidence gate ──► decision
```

1. **Detect and rectify.** Find the card's corners, perspective-correct it.
   Angled photos otherwise OCR as garbage.
2. **Split slabs.** A graded slab's label already spells out year, set, player
   and number in clean printed text — the most reliable text in the whole image.
   It is detected and read as its own region.
3. **OCR** the label and the face separately.
4. **Parse** every text source — including the listing title — through one
   parser that knows football-card vocabulary: 56 sets, 36 parallels, brand
   aliases, all 32 teams, and the misspellings that show up in real listings
   (`prism` for `prizm`, `donrus` for `donruss`).
5. **Fuse** the sources in trust order: slab label, then title, then card face.
6. **Match** against the catalog: a structured lookup first, then progressively
   looser full-text queries, then weighted scoring with explicit penalties for
   contradictions.
7. **Gate.** Score, margin over the runner-up, and cross-source agreement decide
   whether to accept, review, or reject.

### Why the margin matters more than the score

The dangerous failure is not a low score — it is a *high score shared by two
rows*. A base Prizm and a Silver Prizm are the same card number, same player,
same year; the only difference is a finish that a compressed eBay thumbnail may
not resolve at all. So a top candidate that does not clearly beat the runner-up
is capped and sent to review no matter how well it scores.

---

## Setting up for production

### 1. Install a real OCR backend

Out of the box there is no OCR engine, and the service runs title-only. Install
one on the box that does inference:

```bash
pip install paddlepaddle paddleocr   # recommended
# or: apt install tesseract-ocr && pip install pytesseract
```

PaddleOCR is markedly better on stylized, foil, low-contrast card fonts, which
is where most identification failures come from. It is picked up automatically.

### 2. Get a catalog

Identification maps a photo onto a catalog row, so **catalog quality sets your
accuracy ceiling.** Two options:

```bash
# You have a structured card database (best):
cardid import-catalog cards.csv

# You don't — build one from your own sold titles:
cardid bootstrap sold_titles.txt --min-occurrences 3
```

Bootstrapping parses your titles, collapses the ones describing the same card,
and emits one row each. It is not as clean as a real card database, but it is
derived from exactly the population you need to match, and it works on day one.
`--min-occurrences 3` drops one-off typos.

### 3. Calibrate the threshold

This is the step that makes the precision target real. Label a few hundred rows
you are sure about, then:

```bash
cardid calibrate labels.csv --target 0.99
```

```
recommended threshold: 0.87
  precision at that threshold: 0.9912
  coverage  at that threshold: 0.9240 (92.4% auto-accepted, the rest go to review)

Set it with: export CARDID_AUTO_ACCEPT_THRESHOLD=0.87
```

Use at least ~300 labeled examples — the sample data in `data/` is far too small
to calibrate against and will just recommend `0.0`. Re-run whenever the catalog
or the OCR backend changes.

### 4. Close the loop

Every review resolution is stored as a label. Those labels are both future
training data and the ground truth for your next calibration, so accuracy
improves with use rather than drifting.

---

## Configuration

Environment variables, all prefixed `CARDID_`:

| Variable | Default | Purpose |
|---|---|---|
| `CARDID_CATALOG_PATH` | `data/catalog.db` | Catalog database |
| `CARDID_AUTO_ACCEPT_THRESHOLD` | `0.90` | Set this from `calibrate` |
| `CARDID_REVIEW_THRESHOLD` | `0.45` | Below this, reject instead of queueing |
| `CARDID_MIN_MARGIN` | `0.08` | Near-tie floor; raise it to be stricter about parallels |
| `CARDID_OCR_BACKEND` | `auto` | `paddle`, `tesseract`, `null`, or `auto` |
| `CARDID_API_KEY` | unset | Require `X-API-Key` when set |
| `CARDID_MAX_IMAGE_BYTES` | `12582912` | Upload/fetch size cap |

Extend the card vocabulary without touching code by pointing
`CARDID_VOCAB_PATH` at a JSON file of the same shape as `src/cardid/vocab.py`.

---

## Security

The service fetches caller-supplied URLs, which is a server-side request forgery
risk. Every destination is resolved and rejected if it points at private,
loopback, or link-local space — including cloud metadata endpoints — before any
request is made. Responses are size-capped while streaming, and only `http` and
`https` are allowed.

---

## Development

```bash
.venv/bin/python -m pytest        # 135 tests
```

| Path | Contents |
|---|---|
| `src/cardid/vocab.py` | Card vocabulary — sets, parallels, brands, teams, aliases |
| `src/cardid/pipeline/parse.py` | Text → structured attributes |
| `src/cardid/pipeline/match.py` | Candidate scoring, weights and penalties |
| `src/cardid/pipeline/confidence.py` | Confidence and the decision gate |
| `src/cardid/pipeline/image.py` | Detection, rectification, slab split, hashing |
| `src/cardid/catalog/` | Catalog storage and ingestion |
| `src/cardid/api/app.py` | HTTP API |
| `src/cardid/work/batch.py` | Bulk backfill worker |
| `src/cardid/calibrate.py` | Threshold calibration |

### Where to improve accuracy first

1. **Catalog coverage** — a card absent from the catalog can never be matched.
2. **A real OCR backend** — the single largest jump from the default install.
3. **Vocabulary** — add the sets and parallels your inventory actually contains.
4. **Visual embeddings** — the remaining ceiling. Text alone cannot separate two
   parallels that differ only in finish. `src/cardid/pipeline/embed/` is the
   stub for a CLIP or DINOv2 nearest-neighbour reranker over reference images;
   that is the work that would push the auto-accepted share meaningfully higher.
