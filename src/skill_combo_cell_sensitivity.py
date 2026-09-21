"""
Sensitivity of the stratified premiums to the minimum cell size.

skill_combo_value.py skips an (experience x city x ceiling) cell unless it holds at
least 5 postings with the itemset and 5 without. This script recomputes every premium
of a finished run with 3/3, 5/5 and 10/10 and reports how much of each itemset's "with"
postings sit in the cells used, how far the premiums move, and how stable the top-10
combinations are. The itemsets and saved premiums are read from
results/skill_combo_value{suffix}.csv and skill_single_value{suffix}.csv; the 5/5
premiums must reproduce the saved ones exactly. Nothing existing is modified.

Usage: python src/skill_combo_cell_sensitivity.py --data PATH --suffix _v3 --no-require-ceiling
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from skill_combo_value import DATA_PATH, MIN_STRATA_USED, MIN_STRATUM_N, load_usable_postings

BASE = MIN_STRATUM_N
THRESHOLDS = (3, BASE, 10)
TOP_N = 10
CELL_COLS = ["experience_bucket", "city_bucket", "has_ceiling"]


def load_itemsets(suffix):
    combos = pd.read_csv(ROOT / "results" / f"skill_combo_value{suffix}.csv")
    singles = pd.read_csv(ROOT / "results" / f"skill_single_value{suffix}.csv")
    rows = [(name, "combo", frozenset(x.strip() for x in name.split(" + ")), n, p, s)
            for name, n, p, s in zip(combos["combination"], combos["n_with"], combos["adjusted_premium"], combos["strata_used"])]
    rows += [(name, "single", frozenset([name]), n, p, None)
             for name, n, p in zip(singles["skill"], singles["n_with"], singles["adjusted_premium"])]
    return rows


def cell_table(codes, valid, salary, n_cells, has):
    """Per cell: postings with / without the itemset, and the difference of mean salary."""
    w, o = valid & has, valid & ~has
    n_with = np.bincount(codes[w], minlength=n_cells)
    n_out = np.bincount(codes[o], minlength=n_cells)
    s_with = np.bincount(codes[w], weights=salary[w], minlength=n_cells)
    s_out = np.bincount(codes[o], weights=salary[o], minlength=n_cells)
    both = (n_with > 0) & (n_out > 0)
    diff = np.zeros(n_cells)
    diff[both] = s_with[both] / n_with[both] - s_out[both] / n_out[both]
    return n_with, n_out, diff


def premium_at(n_with, n_out, diff, k, total_with):
    """Same estimator as skill_combo_value.compute_premium with the cell threshold k/k."""
    used = (n_with >= k) & (n_out >= k)
    weight = int(n_with[used].sum())
    cells = int(used.sum())
    premium = float((diff[used] * n_with[used]).sum() / weight) if cells >= MIN_STRATA_USED and weight > 0 else np.nan
    return cells, weight / total_with, premium


def summarize(res):
    rows = []
    for pop, sub in (("all", res), ("single", res[res["kind"] == "single"]), ("combo", res[res["kind"] == "combo"])):
        for k in THRESHOLDS:
            priced = sub[sub[f"premium_{k}"].notna()]
            row = {"population": pop, "threshold": f"{k}/{k}", "itemsets": len(sub), "priced": len(priced),
                   "cells_used_median": priced[f"cells_{k}"].median(), "cells_used_min": priced[f"cells_{k}"].min(),
                   "cells_used_max": priced[f"cells_{k}"].max(),
                   "coverage_median": priced[f"coverage_{k}"].median(), "coverage_min": priced[f"coverage_{k}"].min(),
                   "coverage_p10": priced[f"coverage_{k}"].quantile(0.10)}
            if k != BASE:
                both = sub[[f"premium_{BASE}", f"premium_{k}"]].dropna()
                gap = (both[f"premium_{k}"] - both[f"premium_{BASE}"]).abs()
                row.update(pairs=len(both), pearson_vs_5=both.corr().iloc[0, 1],
                           spearman_vs_5=both.corr(method="spearman").iloc[0, 1],
                           median_abs_diff=gap.median(), p90_abs_diff=gap.quantile(0.9), max_abs_diff=gap.max())
            rows.append(row)
    return pd.DataFrame(rows)


def top_table(res):
    combos = res[res["kind"] == "combo"].copy()
    for k in THRESHOLDS:
        combos[f"rank_{k}"] = combos[f"premium_{k}"].rank(ascending=False, method="min")
    base_top = combos[combos[f"rank_{BASE}"] <= TOP_N]
    overlap = {k: int((base_top[f"rank_{k}"] <= TOP_N).sum()) for k in THRESHOLDS}
    cols = ["itemset", "n_with"] + [f"{p}_{k}" for k in THRESHOLDS for p in ("premium", "rank")] + [f"cells_{BASE}", f"coverage_{BASE}"]
    return combos[combos[f"rank_{BASE}"] <= TOP_N + 5].sort_values(f"rank_{BASE}")[cols], overlap


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--suffix", default="")
    parser.add_argument("--require-ceiling", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    usable = load_usable_postings(args.data, args.require_ceiling)
    keys = usable[CELL_COLS]
    valid = keys.notna().all(axis=1).to_numpy()     # skill_combo_value's groupby drops rows with a missing cell key
    codes = np.full(len(usable), -1)
    codes[valid] = keys[valid].groupby(CELL_COLS, sort=True).ngroup().to_numpy()
    n_cells = int(codes.max()) + 1
    salary = usable["salary_net"].to_numpy(dtype=float)
    skill_sets = usable["skills_set"].tolist()
    print(f"Usable postings: {len(usable)}; cells: {n_cells} non-empty (experience x city x ceiling); rows without a cell key: {int((~valid).sum())}")

    rows, wrong = [], []
    for name, kind, itemset, saved_n, saved_premium, saved_cells in load_itemsets(args.suffix):
        has = np.fromiter((itemset <= s for s in skill_sets), dtype=bool, count=len(skill_sets))
        total_with = int(has.sum())
        if total_with != saved_n:
            wrong.append((name, total_with, saved_n))
            continue
        n_with, n_out, diff = cell_table(codes, valid, salary, n_cells, has)
        row = {"itemset": name, "kind": kind, "size": len(itemset), "n_with": total_with, "cells_with_any": int((n_with > 0).sum())}
        for k in THRESHOLDS:
            row[f"cells_{k}"], row[f"coverage_{k}"], row[f"premium_{k}"] = premium_at(n_with, n_out, diff, k, total_with)
        row["saved_premium"], row["saved_cells"] = saved_premium, saved_cells
        rows.append(row)
    if wrong:
        raise SystemExit(f"{len(wrong)} itemsets do not have the saved number of postings (wrong --data or --suffix?): {wrong[:3]}")
    res = pd.DataFrame(rows)

    reproduced = (res[f"premium_{BASE}"].round(0) == res["saved_premium"]).all()
    cells_ok = (res.loc[res["kind"] == "combo", f"cells_{BASE}"] == res.loc[res["kind"] == "combo", "saved_cells"]).all()
    print(f"Itemsets: {len(res)} ({(res['kind'] == 'combo').sum()} combinations + {(res['kind'] == 'single').sum()} single skills)")
    print(f"{BASE}/{BASE} premiums reproduce the saved ones exactly: {reproduced}; cells used equal the saved strata_used for every combination: {cells_ok}")
    if not (reproduced and cells_ok):
        raise SystemExit("baseline does not reproduce the saved results; the comparison would not be valid")

    summary = summarize(res)
    top, overlap = top_table(res)
    saved_top = set(pd.read_csv(ROOT / "results" / f"skill_combo_value{args.suffix}.csv").nlargest(TOP_N, "adjusted_premium")["combination"])
    print(f"Top {TOP_N} combinations by the {BASE}/{BASE} premium equal the saved top {TOP_N}: {set(top['itemset'].head(TOP_N)) == saved_top}")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.float_format", lambda v: f"{v:,.3f}")
    print("\n=== Cells, coverage and agreement with 5/5, by threshold ===")
    print(summary.to_string(index=False))
    print(f"\n=== Top {TOP_N} combinations by the 5/5 premium (and the next 5), with their rank under each threshold ===")
    print(top.to_string(index=False))
    print("\nOf the 5/5 top 10, still in the top 10 under: " + ", ".join(f"{k}/{k}: {overlap[k]}" for k in THRESHOLDS if k != BASE))

    res.drop(columns=["saved_premium", "saved_cells"]).to_csv(ROOT / "results" / f"skill_combo_cell_sensitivity{args.suffix}.csv", index=False)
    summary.to_csv(ROOT / "results" / f"skill_combo_cell_sensitivity_summary{args.suffix}.csv", index=False)
    top.to_csv(ROOT / "results" / f"skill_combo_cell_sensitivity_top{args.suffix}.csv", index=False)
    print(f"\nSaved skill_combo_cell_sensitivity{args.suffix}.csv, skill_combo_cell_sensitivity_summary{args.suffix}.csv and "
          f"skill_combo_cell_sensitivity_top{args.suffix}.csv to results/")


if __name__ == "__main__":
    main()
