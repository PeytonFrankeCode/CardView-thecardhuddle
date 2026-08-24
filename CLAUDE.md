# CardView — football card identification

Identifies a football card from a photo, an eBay listing title, or both, and
returns structured card data with a confidence score and an explicit
`auto_accept` / `review` / `reject` decision. Serves a card-tracking website
that needs unsorted sold-listing rows matched to real cards (~20k photos/day).

## Branch workflow — follow this every time

`main` is the trunk and must always hold the latest working state. The owner
keeps it that way so work stays clean and confined across separate chats.

1. Develop on `claude/football-card-identification-jn9xos`.
2. When an update is complete and tests pass, **merge it into `main`**.
3. Push **both** branches.

```bash
git checkout claude/football-card-identification-jn9xos
# ... work, commit ...
python -m pytest                      # must be green before merging
git push -u origin claude/football-card-identification-jn9xos

git checkout main
git merge --no-ff claude/football-card-identification-jn9xos
git push -u origin main
git checkout claude/football-card-identification-jn9xos
```

Never leave an update sitting on the feature branch only. Do not open a pull
request unless asked — the merge is the delivery.

## Setup and commands

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt && .venv/bin/pip install -e .
.venv/bin/python -m pytest            # 142 tests, ~2s

cardid import-catalog data/sample_catalog.csv   # load a catalog
cardid identify --title "2017 Panini Prizm Patrick Mahomes II #269 Silver"
cardid serve --port 8000                        # HTTP API, docs at /docs
cardid batch in.csv out.csv                     # resumable bulk backfill
cardid calibrate labels.csv --target 0.99       # tune the accept threshold
```

No OCR engine ships by default — the service runs title-only until
`paddleocr` or `pytesseract` is installed, then picks it up automatically.

## Layout

| Path | Contents |
|---|---|
| `src/cardid/vocab.py` | Card vocabulary: sets, parallels, brands, teams, aliases |
| `src/cardid/pipeline/parse.py` | Text → structured attributes (titles and OCR both) |
| `src/cardid/pipeline/match.py` | Candidate scoring: weights and contradiction penalties |
| `src/cardid/pipeline/confidence.py` | Confidence and the accept/review/reject gate |
| `src/cardid/pipeline/image.py` | Detection, rectification, slab split, perceptual hash |
| `src/cardid/pipeline/identify.py` | Orchestrator |
| `src/cardid/catalog/` | Catalog storage (SQLite + FTS5) and ingestion |
| `src/cardid/api/app.py` | HTTP API |
| `src/cardid/work/batch.py` | Bulk backfill worker |
| `src/cardid/calibrate.py` | Threshold calibration |

## Design invariants — do not break these

These encode the accuracy strategy. Changing them silently degrades results.

- **Near-ties never auto-accept.** A base and a silver parallel share year, set,
  player and number; the dangerous failure is a high score shared by two rows,
  not a low score. A top candidate that does not clear `min_margin` over the
  runner-up is capped below auto-accept regardless of how well it scores.
- **The listing title is a first-class input, not a hint.** It is often stronger
  than the image and fails in ways uncorrelated with OCR. Never drop it from a
  code path that has one.
- **Contradictions cost more than agreement earns.** The penalty weights in
  `match.py` are deliberately above 1.0; that is what keeps wrong-number and
  wrong-year rows out of the top slot.
- **Caller-supplied URLs are validated before fetch.** `fetch.py` rejects
  private, loopback and link-local addresses (including cloud metadata) and
  caps response size. Do not add a fetch path that bypasses `validate_url`.
- **Thresholds are calibrated, not guessed.** Defaults are a starting point.
  Re-run `cardid calibrate` after changing the catalog, parser, or OCR backend.

## Accuracy framing

Top-1 accuracy on raw eBay photos is realistically 85–93%. The target that
matters is *precision on the auto-accepted slice*, with the uncertain remainder
routed to the review queue. Review resolutions are stored as labels and become
the ground truth for the next calibration. See README.md for the full picture.
