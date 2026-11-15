"""CLI: export a trained CNN run for the in-browser demo on the results site.

The TextCNN is small enough to run in a visitor's browser, which needs no
server and so costs nothing to keep online. This writes what `site/demo/model.js`
reads:

  model.bin    every weight as little-endian float16, in the order listed in
               config.json (float16 halves the download; scores move by
               under 0.001 and no test-set verdict changes)
  config.json  shapes, chunking, pooling rule and both sets of thresholds
  vocab.json   the vocabulary, index = token id
  example_contract.txt  the held-out test contract the demo offers as an example

Run `legal-risk-evaluate` on the run first, so the contract-level thresholds
and pooling rule are the ones the reported results used.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from .labels import LABEL_NAMES
from .runtime import CNNRuntime

# The browser implements these two rules only.
SUPPORTED_POOLING = ("noisy_or", "max")


def weight_tensors(runtime: CNNRuntime) -> list[tuple[str, np.ndarray]]:
    """(name, array) in the order model.bin stores them."""
    model = runtime.model
    tensors = [("embedding", model.embedding.weight)]
    for i, convolution in enumerate(model.convolutions):
        tensors += [(f"conv{i}.weight", convolution.weight), (f"conv{i}.bias", convolution.bias)]
    tensors += [("classifier.weight", model.classifier.weight),
                ("classifier.bias", model.classifier.bias)]
    return [(name, t.detach().cpu().numpy()) for name, t in tensors]


def export(run_dir: Path, out_dir: Path) -> dict:
    runtime = CNNRuntime(run_dir)
    if runtime.pooling not in SUPPORTED_POOLING:
        raise ValueError(f"the browser demo supports {SUPPORTED_POOLING}, not {runtime.pooling!r}")
    if not any((run_dir / f"evaluation_{s}.json").exists() for s in ("test", "val")):
        raise FileNotFoundError(f"run legal-risk-evaluate on {run_dir} first")

    out_dir.mkdir(parents=True, exist_ok=True)
    tensors = weight_tensors(runtime)
    with open(out_dir / "model.bin", "wb") as f:
        for _, array in tensors:
            f.write(array.astype("<f2").tobytes())

    evaluation = json.loads((run_dir / "evaluation_test.json").read_text(encoding="utf-8")) \
        if (run_dir / "evaluation_test.json").exists() else None
    model = runtime.model
    config = {
        "labels": list(LABEL_NAMES),
        "vocab_size": len(runtime.vocab),
        "embedding_dim": model.embedding.embedding_dim,
        "num_filters": model.convolutions[0].out_channels,
        "kernel_sizes": [c.kernel_size[0] for c in model.convolutions],
        "tensors": [{"name": name, "shape": list(array.shape)} for name, array in tensors],
        "chunking": runtime.chunking,
        "pooling": runtime.pooling,
        "window_thresholds": [float(t) for t in runtime.window_thresholds],
        "contract_thresholds": [float(t) for t in runtime.thresholds],
        "test_scores": evaluation and {
            "window_macro_f1": evaluation["chunk_level"]["macro_f1"],
            "contract_macro_f1": evaluation["document_level"]["macro_f1"],
            "contract_macro_ap": evaluation["document_level"]["macro_average_precision"],
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
    shutil.copyfile(run_dir / "vocab.json", out_dir / "vocab.json")
    # The demo's example, shared with the Gradio demo so the two cannot differ.
    shutil.copyfile(Path(__file__).parent / "example_contract.txt", out_dir / "example_contract.txt")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run_dir", type=Path, default=Path("outputs/cnn"))
    parser.add_argument("--out_dir", type=Path, default=Path("site/demo"))
    args = parser.parse_args()
    config = export(args.run_dir, args.out_dir)
    size = (args.out_dir / "model.bin").stat().st_size / 1e6
    print(f"exported {args.run_dir} -> {args.out_dir}  (model.bin {size:.1f} MB, "
          f"pooling {config['pooling']})")


if __name__ == "__main__":
    main()
