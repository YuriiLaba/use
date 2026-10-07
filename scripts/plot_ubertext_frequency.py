"""Render the paper frequency figure from audited UberText summary counts.

Run from the repository root:
    python -m scripts.plot_ubertext_frequency
The summary contains aggregates only, not corpus sentences.
"""

import argparse
import json
import os
from pathlib import Path


def plot_frequency(
    statistics_file: Path,
    output_dir: Path,
    top_n: int = 15,
    pdf_output_dir: Path = Path("output/pdf"),
) -> tuple[Path, Path]:
    with statistics_file.open(encoding="utf-8") as handle:
        report = json.load(handle)
    rows = report["corpus"]["top_20"]
    if not 1 <= top_n <= len(rows):
        raise ValueError(f"top_n must be between 1 and {len(rows)}")
    rows = rows[:top_n]
    if any(row["count"] <= 0 for row in rows):
        raise ValueError("Figure counts must be positive")
    if any(a["count"] < b["count"] for a, b in zip(rows, rows[1:])):
        raise ValueError("Summary frequencies must be sorted in descending order")

    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_output_dir.mkdir(parents=True, exist_ok=True)
    # Keep generated caches inside the project instead of a user's home directory.
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator, StrMethodFormatter

    with plt.rc_context({"font.family": "DejaVu Serif", "font.size": 12, "pdf.fonttype": 42}):
        figure, axis = plt.subplots(figsize=(8.5, max(3.5, top_n * 0.31 + 1.2)))
        labels = [row["lemma"] for row in rows]
        values = [row["count"] / 1_000_000 for row in rows]
        bars = axis.barh(labels, values, height=0.72, color="#b8b4f5", edgecolor="#635bff", linewidth=0.8)
        axis.invert_yaxis()
        axis.set_xlim(0, max(values) * 1.14)
        axis.set_xlabel("Number of sentences per lemma (millions)", labelpad=10)
        axis.xaxis.set_major_locator(MultipleLocator(0.1))
        axis.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        axis.tick_params(axis="both", direction="in", color="#a0a0a0")
        axis.tick_params(axis="y", pad=9)
        axis.set_axisbelow(True)
        axis.grid(axis="x", color="#e7e7e7", linewidth=0.55)
        for spine in axis.spines.values():
            spine.set_color("#777777")
        for bar, value in zip(bars, values):
            axis.text(value + max(values) * 0.012, bar.get_y() + bar.get_height() / 2,
                      f"{value:.2f}", va="center", fontsize=9, color="#5147ff")
        figure.tight_layout(pad=1.3)
        png = output_dir / "ubertext_frequency.png"
        pdf = pdf_output_dir / "ubertext_frequency.pdf"
        figure.savefig(png, dpi=300, facecolor="white")
        figure.savefig(pdf, facecolor="white")
        plt.close(figure)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--statistics", type=Path, default=Path("results/ubertext_coverage/statistics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/ubertext_coverage"))
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--pdf-output-dir", type=Path, default=Path("output/pdf"))
    args = parser.parse_args()
    png, pdf = plot_frequency(args.statistics, args.output_dir, args.top_n, args.pdf_output_dir)
    print(f"Saved {png}")
    print(f"Saved {pdf}")


if __name__ == "__main__":
    main()
