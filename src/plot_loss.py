"""Plot all recorded losses without smoothing; export the source points as CSV."""

import argparse
import csv
from importlib.metadata import version
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from checkpoint import write_json
from config import Config
from prepare_data import sha256_file


def plot_losses(run_dir: Path, output_dir: Path | None = None) -> None:
    output_dir = output_dir or run_dir / "report"
    output_dir.mkdir(parents=True, exist_ok=True)
    source = run_dir / "results.json"
    results = json.loads(source.read_text())
    train, validation = results["training_history"], results["validation_history"]
    with (output_dir / "loss_points.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["split", "tokens_seen", "cross_entropy_nats_per_token"])
        for split, records in (("training", train), ("validation", validation)):
            for record in records:
                writer.writerow([split, record["tokens_seen"], record["loss"]])
    with plt.rc_context({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none", "figure.facecolor": "white", "axes.facecolor": "white"}):
        fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
        ax.plot([r["tokens_seen"] / 1e6 for r in train], [r["loss"] for r in train],
                color="#0072B2", linewidth=0.75, label="Training (each optimizer step)")
        ax.plot([r["tokens_seen"] / 1e6 for r in validation], [r["loss"] for r in validation],
                color="#A34A00", linestyle="--", marker="o", markersize=4,
                linewidth=1.6, label=f"Validation (fixed {results['validation_target_count']:,} targets)")
        ax.set(xlabel="Processed target tokens (millions)", ylabel="Cross-entropy (nats/token)",
               title="TinyGPT learning curves", xlim=(0, results["tokens_seen"] / 1e6), ylim=(0, None))
        ax.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, loc="upper right")
        for suffix in ("png", "svg"):
            fig.savefig(output_dir / f"loss_curve.{suffix}", dpi=200, facecolor="white")
        plt.close(fig)
    write_json(output_dir / "figure_manifest.json", {
        "source": str(source), "source_sha256": sha256_file(source),
        "transformations": ["x divided by 1,000,000 for millions of processed tokens"],
        "smoothing": "none", "exclusions": "none", "uncertainty": "single training run; no error bars",
        "training_observations": len(train), "validation_observations": len(validation),
        "size_inches": [8, 4.8], "png_dpi": 200, "matplotlib_version": version("matplotlib"),
        "purpose": "coursework results; no publisher-specific formatting claims",
        "alt_text": f"Training and validation loss decrease over {results['tokens_seen'] / 1e6:.2f} million "
                    f"target tokens. Validation loss falls from {validation[0]['loss']:.3f} "
                    f"to {validation[-1]['loss']:.3f}. After early training, raw training points fluctuate "
                    "above the validation curve.",
    })
    print(f"Saved {output_dir / 'loss_curve.png'} and loss_curve.svg; source points in loss_points.csv")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Config().output_dir)
    plot_losses(parser.parse_args().run_dir)


if __name__ == "__main__":
    main()
