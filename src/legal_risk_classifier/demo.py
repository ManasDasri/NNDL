"""CLI: a browser demo that flags risk clauses in pasted contract text.

Paste a contract, get the six labels scored, the flagged passages highlighted
in place, and the phrases that triggered each flag. Runs anywhere Python does,
including a Colab cell with `--share` for a public link.

It deliberately reuses the evaluation runtime rather than reimplementing
inference. A demo that chunks or pools differently from the evaluation would
show numbers that do not match the reported ones, which is worse than having
no demo.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .attribution import explain
from .labels import LABEL_NAMES, LABEL_TO_INDEX
from .runtime import CNNRuntime

DESCRIPTION = """
Flags six risk-relevant clause types in commercial contracts, trained on
[CUAD v1](https://www.atticusprojectai.org/cuad).

Scores are pooled across overlapping windows of the document: a clause found
in any window flags the contract. Thresholds are tuned per label on the
validation split, so they are not all 0.5.
"""

# A held-out test contract from CUAD (CC BY 4.0), annotated with Cap on
# Liability, Insurance and License Grant. Chosen because the model gets all
# three right, so it shows a working case, not a typical one.
TEST_CONTRACT = (Path(__file__).parent / "example_contract.txt").read_text()

# A short made-up excerpt with four of the six clause types. Far shorter than
# the contracts the model trained on, and it misses most of them, including the
# "in no event shall ... exceed" liability cap (see handbook chapter 10).
EXCERPT = (
    "Supplier shall maintain, at its own expense, commercial general liability insurance "
    "with limits of not less than $2,000,000 per occurrence, and shall provide a certificate "
    "of insurance naming Customer as an additional insured. In no event shall either party's "
    "aggregate liability under this Agreement exceed the fees paid in the twelve months "
    "preceding the claim. Customer may terminate this Agreement for convenience at any time "
    "upon thirty (30) days' prior written notice. Supplier grants Customer a non-exclusive, "
    "non-transferable license to use the Software during the Term."
)


def analyse(runtime: CNNRuntime, text: str, label: str) -> tuple[dict[str, float], str, str]:
    """Score text and describe one label's evidence. The demo's whole logic."""
    if not text.strip():
        return {}, "Paste some contract text first.", ""

    prediction = runtime.predict_document(text)
    # Each label carries its own validation-tuned threshold, so a lower score
    # can be a flag while a higher one is not. Showing the threshold keeps that
    # legible instead of looking like a bug.
    scores = {
        f"{name} {'✓' if name in prediction.predicted else '·'} thr {runtime.threshold_for(name):.2f}":
            prediction.document_scores[name]
        for name in LABEL_NAMES
    }

    threshold = runtime.window_threshold_for(label)
    hits = prediction.hits(label, threshold=threshold)
    if hits:
        passages = "\n\n---\n\n".join(
            f"**Window {i + 1} — confidence {hit.score:.2f}** "
            f"(characters {hit.char_start:,}–{hit.char_end:,})\n\n> {hit.text.strip()}"
            for i, hit in enumerate(hits[:5])
        )
    else:
        passages = (
            f"No passage reached the {label} threshold of {threshold:.2f}. "
            f"The strongest window scored {prediction.chunk_probs[:, LABEL_TO_INDEX[label]].max():.2f}."
        )

    phrases = explain(runtime.model, runtime.vocab, hits[0].text if hits else text, label, top_n=6)
    triggers = (
        "\n".join(f"- `{phrase.text}` — {phrase.contribution:.2f}" for phrase in phrases)
        if phrases
        else "_No phrase contributed positively._"
    )
    return scores, passages, triggers


def build_interface(runtime: CNNRuntime):
    import gradio as gr

    def on_submit(text: str, label: str):
        return analyse(runtime, text, label)

    with gr.Blocks(title="Contract clause risk detection") as interface:
        gr.Markdown("# Contract clause risk detection")
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Column(scale=3):
                text = gr.Textbox(
                    label="Contract text",
                    lines=18,
                    placeholder="Paste a commercial contract, or load an example below.",
                )
                label = gr.Dropdown(
                    choices=list(LABEL_NAMES),
                    value=LABEL_NAMES[0],
                    label="Inspect passages for",
                )
                run = gr.Button("Analyse", variant="primary")
                gr.Examples(
                    [[TEST_CONTRACT, "Insurance"], [EXCERPT, "Cap on Liability"]],
                    inputs=[text, label],
                    example_labels=["Test-set contract (sponsorship agreement)",
                                    "Short made-up excerpt (shows its limits)"],
                )
            with gr.Column(scale=2):
                scores = gr.Label(label="Document-level scores", num_top_classes=6)

        with gr.Row():
            passages = gr.Markdown(label="Flagged passages")
        with gr.Accordion("What triggered this label", open=False):
            triggers = gr.Markdown()

        run.click(on_submit, inputs=[text, label], outputs=[scores, passages, triggers])
        label.change(on_submit, inputs=[text, label], outputs=[scores, passages, triggers])

    return interface


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_dir", type=Path, default=Path("outputs/cnn"))
    parser.add_argument("--share", action="store_true", help="public link, for Colab")
    parser.add_argument("--port", type=int, default=7860)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = CNNRuntime(args.run_dir)
    print(f"loaded {args.run_dir}  (chunk macro F1 "
          f"{runtime.metrics['test_at_tuned_thresholds']['macro_f1']:.3f})")
    if runtime.evaluation is None:
        # Without legal-risk-evaluate the contract verdicts use window thresholds
        # and will not match the reported numbers or the browser demo.
        print("WARNING: no evaluation_*.json in run_dir; run legal-risk-evaluate first, "
              "or load the published model (notebook 5).")
    build_interface(runtime).launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
