"""
Bootstrap confidence intervals for skill_combo_value.py's stratified
salary premiums -- each is a single point estimate from a modest sample,
with no answer to "would this hold up elsewhere, or is it noise?"
Resamples the usable population WITH REPLACEMENT (default 1000x),
recomputes every premium the same way, and reports the 2.5th/97.5th
percentile spread as a 95% CI -- a CI excluding zero is unlikely to be
noise; one straddling zero shouldn't be presented as confirmed.

Precomputes a boolean (rows x skills) matrix and resamples row indices
rather than recomputing set-membership per iteration -- fast enough for
1000 resamples x ~340 itemsets.

Usage: python src/skill_combo_bootstrap_ci.py [--n-bootstrap 1000] [--data PATH] [--suffix _v2]
--suffix must match whatever skill_combo_value.py was run with.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from skill_combo_value import DATA_PATH, load_usable_postings

MIN_STRATUM_N = 5
MIN_STRATA_USED = 2
RANDOM_SEED = 42


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--suffix", default="")
    parser.add_argument("--require-ceiling", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    n_bootstrap = args.n_bootstrap

    usable = load_usable_postings(args.data, args.require_ceiling)
    n = len(usable)
    print(f"Usable postings: {n}")

    combos_df = pd.read_csv(ROOT / "results" / f"skill_combo_value{args.suffix}.csv")
    singles_df = pd.read_csv(ROOT / "results" / f"skill_single_value{args.suffix}.csv")
    combos_df["skill_list"] = combos_df["combination"].apply(lambda s: [x.strip() for x in s.split(" + ")])

    itemsets = {row["combination"]: row["skill_list"] for _, row in combos_df.iterrows()}
    itemsets.update({s: [s] for s in singles_df["skill"]})
    print(f"Itemsets to bootstrap: {len(itemsets)} ({len(combos_df)} combos + {len(singles_df)} singles)")

    all_skills = sorted({s for skills in itemsets.values() for s in skills})
    skill_index = {s: i for i, s in enumerate(all_skills)}

    skill_bool = np.zeros((n, len(all_skills)), dtype=bool)
    for row_i, skills in enumerate(usable["skills"]):
        for s in skills:
            if s in skill_index:
                skill_bool[row_i, skill_index[s]] = True

    strata_labels = (
        usable["experience_bucket"].astype(str) + "|" + usable["city_bucket"].astype(str)
        + "|" + usable["has_ceiling"].astype(str)
    ).values
    _, strata_codes = np.unique(strata_labels, return_inverse=True)
    salary = usable["salary_net"].values

    itemset_indices = {name: [skill_index[s] for s in skills] for name, skills in itemsets.items()}

    rng = np.random.default_rng(RANDOM_SEED)
    boot_results = {name: [] for name in itemsets}

    print(f"Running {n_bootstrap} bootstrap resamples...")
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        b_skill_bool = skill_bool[idx]
        b_strata = strata_codes[idx]
        b_salary = salary[idx]

        stratum_values, stratum_counts = np.unique(b_strata, return_counts=True)
        usable_strata = stratum_values[stratum_counts >= 2 * MIN_STRATUM_N]
        stratum_masks = {sv: (b_strata == sv) for sv in usable_strata}

        for name, indices in itemset_indices.items():
            has_full = b_skill_bool[:, indices].all(axis=1)
            weighted_sum, weight_total, strata_used = 0.0, 0, 0
            for sv in usable_strata:
                s_mask = stratum_masks[sv]
                s_has = has_full[s_mask]
                n_has = int(s_has.sum())
                n_not = len(s_has) - n_has
                if n_has >= MIN_STRATUM_N and n_not >= MIN_STRATUM_N:
                    s_sal = b_salary[s_mask]
                    diff = s_sal[s_has].mean() - s_sal[~s_has].mean()
                    weighted_sum += diff * n_has
                    weight_total += n_has
                    strata_used += 1
            if strata_used >= MIN_STRATA_USED and weight_total > 0:
                boot_results[name].append(weighted_sum / weight_total)

        if (b + 1) % 100 == 0:
            print(f"  {b + 1}/{n_bootstrap} resamples done")

    rows = []
    for name, vals in boot_results.items():
        if len(vals) < n_bootstrap * 0.5:
            continue
        vals = np.array(vals)
        combo_row = combos_df[combos_df["combination"] == name]
        if len(combo_row):
            point_estimate = combo_row["adjusted_premium"].iloc[0]
        else:
            point_estimate = singles_df.loc[singles_df["skill"] == name, "adjusted_premium"].iloc[0]
        ci_low, ci_high = np.percentile(vals, [2.5, 97.5])
        rows.append({
            "name": name,
            "point_estimate": round(point_estimate, 0),
            "boot_mean": round(vals.mean(), 0),
            "ci_low_95": round(ci_low, 0),
            "ci_high_95": round(ci_high, 0),
            "excludes_zero": not (ci_low <= 0 <= ci_high),
            "n_valid_resamples": len(vals),
        })

    result_df = pd.DataFrame(rows)
    result_df["abs_point"] = result_df["point_estimate"].abs()
    result_df = result_df.sort_values("abs_point", ascending=False).drop(columns="abs_point")

    pd.set_option("display.width", 140)
    print(f"\n=== Bootstrap 95% CIs, {n_bootstrap} resamples ===")
    print("excludes_zero=True means the 95% CI does not contain zero -- unlikely to be noise")
    print(result_df.to_string(index=False))

    n_sig = int(result_df["excludes_zero"].sum())
    print(f"\n{n_sig}/{len(result_df)} premiums have a 95% CI excluding zero (survive the noise check)")

    out_path = ROOT / "results" / f"skill_combo_bootstrap_ci{args.suffix}.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
