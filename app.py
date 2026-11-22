"""Gradio front end for the NCR Description Assistant.

Run locally:   python app.py
On Hugging Face Spaces this file is the entry point (see README front matter).
"""

from __future__ import annotations

import csv
from pathlib import Path

import gradio as gr

from ncr_assistant import check_description, classify

DATA = Path(__file__).parent / "data" / "synthetic_ncr_descriptions.csv"

GRADE_ICON = {"Actionable": "✅", "Needs detail": "⚠️", "Rewrite": "❌"}


def load_examples(n: int = 8) -> list[list[str]]:
    """Alternate good/poor examples from the synthetic set for the UI."""
    with DATA.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    good = [r["description"] for r in rows if r["quality"] == "good"]
    poor = [r["description"] for r in rows if r["quality"] == "poor"]
    mixed = [x for pair in zip(poor, good) for x in pair]
    return [[t] for t in mixed[:n]]


def analyze(text: str, use_model: bool):
    if not (text or "").strip():
        return "Enter a defect description.", "", {}

    report = check_description(text)
    scores, method = classify(text, use_model=use_model)

    def mark(ok: bool) -> str:
        return "✅" if ok else "—"

    summary = (
        f"## {GRADE_ICON[report.grade]} {report.grade} — {report.score}/100\n\n"
        f"| Check | Result |\n|---|---|\n"
        f"| Observable condition | {mark(report.has_condition)} |\n"
        f"| Location | {mark(report.has_location)} |\n"
        f"| Quantity / measurement | {mark(report.has_measurement)} |\n"
        f"| Cause speculation | {'⚠️ ' + ', '.join(report.cause_speculation) if report.cause_speculation else 'none'} |"
    )
    feedback = "### Feedback\n" + "\n".join(f"- {line}" for line in report.feedback)
    feedback += f"\n\n<sub>Category method: {method}</sub>"
    return summary, feedback, scores


with gr.Blocks(title="NCR Description Assistant") as demo:
    gr.Markdown(
        "# NCR Description Assistant\n"
        "Write what you **observed** — what, where, how much — and leave root "
        "cause to the investigation. Paste a defect description to score it.\n\n"
        "*All examples are synthetic. Do not paste proprietary or customer data "
        "into a public Space.*"
    )
    with gr.Row():
        with gr.Column():
            text_in = gr.Textbox(
                label="Defect description",
                placeholder="e.g., Oil weep at HV bushing #2 flange gasket",
                lines=3,
            )
            use_model = gr.Checkbox(
                label="Use zero-shot model for category (slower, better on new wording)",
                value=False,
            )
            btn = gr.Button("Check description", variant="primary")
            gr.Examples(load_examples(), inputs=text_in)
        with gr.Column():
            summary_out = gr.Markdown()
            feedback_out = gr.Markdown()
            category_out = gr.Label(label="Suggested category", num_top_classes=3)

    btn.click(analyze, [text_in, use_model], [summary_out, feedback_out, category_out])
    text_in.submit(analyze, [text_in, use_model], [summary_out, feedback_out, category_out])


if __name__ == "__main__":
    demo.launch()
