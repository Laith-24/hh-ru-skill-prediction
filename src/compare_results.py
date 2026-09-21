"""Merge Spark-native and non-Spark model results into one comparison table."""

import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", nargs="+",
        default=["results/spark_results.csv", "results/nonspark_results.csv"],
    )
    parser.add_argument("--output", default="results/combined_results.csv")
    args = parser.parse_args()

    frames = [pd.read_csv(path) for path in args.results]
    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(args.output, index=False)

    print("\n=== Model comparison (mean across skills) ===")
    print(combined.groupby("model")[["pr_auc", "roc_auc"]].mean().sort_values("pr_auc", ascending=False))

    print("\n=== Best model per skill (by PR-AUC) ===")
    best = combined.loc[combined.groupby("skill")["pr_auc"].idxmax()]
    print(best[["skill", "model", "pr_auc", "roc_auc"]].to_string(index=False))


if __name__ == "__main__":
    main()
