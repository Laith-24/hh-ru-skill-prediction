"""Analysis half of the market-signal check (see market_signal_extract.py).
Pure pandas/scipy, no Spark.

For each job title with enough test-set postings, compares the model's
mean-predicted-probability ranking against the real skill-frequency
ranking (model_corr), and against the same real ranking vs. the global,
title-blind average (baseline_corr) -- if model_corr isn't clearly
higher, the model isn't adding title-specific signal, just recovering
the population base rate.

Usage: python src/market_signal_analyze.py [min_group_size]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "results" / "_market_signal_raw.parquet"

MIN_GROUP_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def main():
    df = pd.read_parquet(RAW_PATH)
    skills = [c[len("label_"):] for c in df.columns if c.startswith("label_")]
    label_cols = [f"label_{s}" for s in skills]
    pred_cols = [f"pred_{s}" for s in skills]

    group_sizes = df.groupby("name").size().sort_values(ascending=False)
    print(f"Total test rows: {len(df)}, distinct titles: {len(group_sizes)}")
    print(f"Largest groups:\n{group_sizes.head(15)}\n")

    qualifying = group_sizes[group_sizes >= MIN_GROUP_SIZE]
    print(f"Titles with >= {MIN_GROUP_SIZE} test rows: {len(qualifying)}")
    if len(qualifying) == 0:
        print("No qualifying groups -- rerun with a smaller min_group_size.")
        return

    global_actual = df[label_cols].mean()
    global_actual.index = skills

    model_corrs, baseline_corrs, rows_out = [], [], []
    for title, n in qualifying.items():
        sub = df[df["name"] == title]
        actual = sub[label_cols].mean()
        actual.index = skills
        predicted = sub[pred_cols].mean()
        predicted.index = skills

        model_corr, _ = spearmanr(actual, predicted)
        baseline_corr, _ = spearmanr(actual, global_actual)
        model_corrs.append(model_corr)
        baseline_corrs.append(baseline_corr)
        rows_out.append({"title": title, "n": n, "model_corr": model_corr, "baseline_corr": baseline_corr})

    summary = pd.DataFrame(rows_out).sort_values("n", ascending=False)
    pd.set_option("display.width", 120)
    print("\n" + summary.to_string(index=False))
    print(f"\nMean model-vs-actual Spearman corr:    {np.nanmean(model_corrs):.3f}")
    print(f"Mean global-baseline-vs-actual Spearman: {np.nanmean(baseline_corrs):.3f}")
    print(f"Model beats baseline in {sum(m > b for m, b in zip(model_corrs, baseline_corrs))}/{len(model_corrs)} groups")

    for title, n in qualifying.head(3).items():
        sub = df[df["name"] == title]
        actual = sub[label_cols].mean()
        actual.index = skills
        predicted = sub[pred_cols].mean()
        predicted.index = skills
        detail = pd.DataFrame({"actual_freq": actual, "predicted_mean_prob": predicted})
        detail = detail.sort_values("actual_freq", ascending=False)
        print(f"\n=== {title!r} (n={n}) ===")
        print(detail.to_string())

    out_path = ROOT / "results" / "market_signal_check.csv"
    summary.to_csv(out_path, index=False)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
