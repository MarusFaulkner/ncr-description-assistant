"""Prove the browser engine (web/engine.js) matches the Python engine.

Runs every row of both datasets, plus edge cases, through Node and compares
score, grade, flags, feedback, and category ranking with the Python result.
Skipped automatically if Node isn't installed.
"""

import csv
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from build_static import engine_config
from ncr_assistant import check_description, keyword_classify

ROOT = Path(__file__).resolve().parent.parent
EDGE_CASES = [
    "", "   ", "bad", "probably maybe because due to",
    "x3 bolts loose at pos. #4, " + "word " * 40,
    "Oil weep, 2.5 mm, qty 3, item 12, h1 terminal",
    "Scratch on cover, 120 mm long",
    "Valve not installed, qty 1 short",
]

pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _texts() -> list[str]:
    texts = list(EDGE_CASES)
    for name in ("synthetic_ncr_descriptions.csv", "holdout_ncr_descriptions.csv"):
        with (ROOT / "data" / name).open(encoding="utf-8") as f:
            texts += [r["description"] for r in csv.DictReader(f)]
    return texts


def _run_js(texts: list[str]) -> list[dict]:
    script = (
        "const {makeEngine}=require(process.argv[1]);"
        "const inp=JSON.parse(require('fs').readFileSync(0,'utf8'));"
        "const e=makeEngine(inp.cfg);"
        "console.log(JSON.stringify(inp.texts.map(t=>({r:e.checkDescription(t),c:e.keywordClassify(t)}))));"
    )
    proc = subprocess.run(
        ["node", "-e", script, str(ROOT / "web" / "engine.js")],
        input=json.dumps({"cfg": engine_config(), "texts": texts}),
        capture_output=True, text=True, check=True,
    )
    return json.loads(proc.stdout)


def test_js_matches_python_on_every_row():
    texts = _texts()
    js_results = _run_js(texts)
    assert len(js_results) == len(texts)
    for text, js in zip(texts, js_results):
        py = check_description(text)
        r = js["r"]
        assert r["score"] == py.score, text
        assert r["grade"] == py.grade, text
        assert r["has_condition"] == py.has_condition, text
        assert r["has_location"] == py.has_location, text
        assert r["has_measurement"] == py.has_measurement, text
        assert r["cause_speculation"] == py.cause_speculation, text
        assert r["feedback"] == py.feedback, text

        py_cats = list(keyword_classify(text).items())
        js_cats = [tuple(x) for x in js["c"]]
        assert [c for c, _ in js_cats] == [c for c, _ in py_cats], text
        for (_, a), (_, b) in zip(js_cats, py_cats):
            assert a == pytest.approx(b), text
