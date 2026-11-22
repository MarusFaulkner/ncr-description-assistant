---
title: NCR Description Assistant
emoji: 🔍
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: mit
short_description: Score nonconformance defect descriptions for clarity
---

# NCR Description Assistant

A small tool that checks a nonconformance (NCR) defect description before it is
submitted, and suggests a defect category.

**The principle it enforces:** the person who finds a defect records what they
*observed* (what, where, how much) and leaves root cause to the
investigation. Vague descriptions ("paint bad") and guesses at cause ("probably
from shipping") slow down disposition and corrupt defect data.

| Input | Score | Why |
|---|---|---|
| `bushing leaking` | ❌ Rewrite | No location, no extent |
| `leak probably from bad gasket` | ❌ Rewrite | Guesses the cause |
| `Oil weep at HV bushing #2 flange gasket, approx 1 drop per 10 min` | ✅ Actionable | Condition, location, rate |

## How it works

**1. Rule-based description check** (`ncr_assistant/rules.py`): transparent and
explainable by design, so an inspector can see exactly why a description scored
the way it did.

| Rule | Points |
|---|---|
| States an observable condition (leak, crack, missing, out of tolerance…) | +30 |
| States where (feature, side, phase, position/item number) | +25 |
| States how much (value with units, or a count) | +25 |
| Reasonable length (4–30 words) | +20 |
| Vague wording with no observable condition | −20 |
| Speculates about cause (because, probably, due to…) | −25 |

Grades: **Actionable** ≥ 75 · **Needs detail** 50–74 · **Rewrite** < 50

**2. Defect category suggestion** (`ncr_assistant/classifier.py`)
- *Keyword* (default): offline, instant, no dependencies.
- *Zero-shot* (optional checkbox): [`facebook/bart-large-mnli`](https://huggingface.co/facebook/bart-large-mnli)
  classifies against the category names with no training data.

## Results, reported honestly

| Dataset | Rows | Category accuracy (keyword) | Quality agreement |
|---|---|---|---|
| Tuning set (`synthetic_ncr_descriptions.csv`) | 38 | 100% | 100% |
| **Held-out set** (`holdout_ncr_descriptions.csv`) | 12 | **50%** | **67%** |

The tuning-set score is inflated by construction: the keyword lists and rules
were adjusted against it. The **held-out set was written separately and scored
once, without tuning**. It's the real number. Keyword matching breaks on
wording it hasn't seen ("oil film" instead of "leak", "primer showing through"
instead of "scratch"). That gap is the reason for the zero-shot option, and the
next step is to measure it on the same held-out set.

Two bugs the evaluation caught and fixed (both have regression tests):
- Units like `mm` voted for *Dimensional* on every measured defect, so a 120 mm
  scratch was classified as a dimensional issue.
- Bare `short` voted for *Electrical*, so "qty 1 short" was read as a short circuit.

## Run it

```bash
pip install -r requirements.txt
python app.py                                         # UI at http://localhost:7860
python -m pytest -q                                   # 16 tests, incl. Python/JS parity (needs Node)
python evaluate.py                                    # tuning set
python evaluate.py --data data/holdout_ncr_descriptions.csv   # honest number
pip install transformers torch && python evaluate.py --model --data data/holdout_ncr_descriptions.csv
```

## Deploy free (Hugging Face Static Space)

Server-backed Gradio Spaces need a paid plan, so the public version is a
single self-contained web page with **no external dependencies**:

```bash
python build_static.py    # writes space/index.html and space/README.md
```

Create a Space with the **Static** SDK and upload the two files from `space/`.

How it stays a single source of truth: `build_static.py` exports the
vocabulary and regex patterns from `ncr_assistant/` into the page as JSON, and
`web/engine.js` mirrors only the scoring logic. `tests/test_web_parity.py`
runs every dataset row and edge case through both engines and fails on any
difference in score, grade, flags, feedback, or category ranking.

*Why not Gradio-Lite?* It installs Python packages in the browser at load
time, and an upstream `huggingface-hub` release broke that install. A page
with no dependencies can't be broken that way, and it loads instantly.

The web build uses the keyword classifier only; the zero-shot model needs a
server. Regenerate `space/` after any change to `ncr_assistant/`.

## Adapting it

The category scheme and vocabulary are generic and illustrative. To fit a
specific product line, edit `CATEGORIES` in `classifier.py` and the term lists
in `rules.py`, add examples to the tuning CSV, and **keep a held-out set you
never tune against**.

## Data

All descriptions in `data/` are **synthetic**, written for this project to
resemble general manufacturing and power-transformer defects. No company
records, procedures, or proprietary defect codes are included. Don't paste
proprietary or customer information into a public deployment.

## Roadmap

- [ ] Measure the zero-shot classifier on the held-out set
- [ ] Grow the held-out set to 50+ rows for a stable estimate
- [ ] Fine-tune a small model (e.g., DistilBERT) once there's enough labeled data
- [ ] "Suggest a rewrite" for descriptions graded Rewrite

## License

MIT, see `LICENSE`.

---
Built by Marcus Faulkner, Quality Inspector.
