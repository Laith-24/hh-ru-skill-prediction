"""
Stress test for skill_value_app.py's recommendation logic -- runs the exact
same find_recommendations() and rules-filtering code (duplicated here since
app.py has top-level Streamlit calls that assume a `streamlit run` context,
same reason interface/test_predict.py exists for the other app) against
many real skill combinations, not just a couple of hand-picked ones.

Specifically targets the class of bug already found once by hand (a pandas
.apply()-on-empty-frame quirk that dropped the "lift" column and crashed
sort_values when a skill's rules ended up empty): every one of the 170
single skills is tried alone, which exercises that exact empty-rules path
101 times (the number of skills with zero frequent pairs, checked directly)
-- far more thorough than clicking through a handful of cases by hand.

Usage: python interface/test_multi_skill_stress.py [--suffix _v2]
"""
import argparse
import ast
import itertools
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results"


def load_data(suffix=""):
    combos = pd.read_csv(RESULTS_DIR / f"skill_combo_value{suffix}.csv")
    singles = pd.read_csv(RESULTS_DIR / f"skill_single_value{suffix}.csv")
    rules = pd.read_csv(RESULTS_DIR / f"skill_combo_rules{suffix}.csv")

    combos = combos.dropna(subset=["adjusted_premium"]).copy()
    singles = singles.dropna(subset=["adjusted_premium"]).copy()
    singles = singles.sort_values("adjusted_premium", ascending=False).reset_index(drop=True)

    combos["skills"] = combos["combination"].apply(lambda s: [x.strip() for x in s.split(" + ")])
    rules["antecedent_list"] = rules["antecedent"].apply(ast.literal_eval)
    rules["consequent_list"] = rules["consequent"].apply(ast.literal_eval)
    return combos, singles, rules


def find_recommendations(known_skills, combos):
    known_set = set(known_skills)
    n_known = len(known_set)

    exact_mask = combos["skills"].apply(lambda s: len(s) == n_known + 1 and known_set.issubset(set(s)))
    exact = combos[exact_mask].copy()
    if not exact.empty:
        exact["suggested_skill"] = exact["skills"].apply(lambda s: [x for x in s if x not in known_set][0])
        return exact.sort_values("adjusted_premium", ascending=False), "exact"

    pairwise_rows = []
    for k in known_skills:
        k_mask = combos["skills"].apply(lambda s: k in s and len(s) == 2)
        for _, row in combos[k_mask].iterrows():
            other = [x for x in row["skills"] if x != k][0]
            if other in known_set:
                continue
            pairwise_rows.append({
                "known_skill": k, "suggested_skill": other,
                "adjusted_premium": row["adjusted_premium"], "n_with": row["n_with"],
            })

    if not pairwise_rows:
        return pd.DataFrame(), "none"

    pairwise_df = pd.DataFrame(pairwise_rows)
    agg_rows = []
    for candidate, group in pairwise_df.groupby("suggested_skill"):
        agg_rows.append({
            "suggested_skill": candidate,
            "adjusted_premium": np.average(group["adjusted_premium"], weights=group["n_with"]),
            "n_with": int(group["n_with"].sum()),
            "based_on": ", ".join(sorted(group["known_skill"].unique())),
        })
    return pd.DataFrame(agg_rows).sort_values("adjusted_premium", ascending=False), "pairwise"


def filter_rules(known_skills, rules):
    known_set = set(known_skills)
    rule_mask = rules["antecedent_list"].apply(lambda a: set(a).issubset(known_set))
    skill_rules = rules[rule_mask].copy()
    if not skill_rules.empty:
        consequent_mask = skill_rules["consequent_list"].apply(lambda c: not set(c).issubset(known_set))
        skill_rules = skill_rules[consequent_mask]
    if not skill_rules.empty:
        skill_rules = skill_rules.sort_values("lift", ascending=False)
    return skill_rules


def run_trial(known_skills, combos, singles, rules):
    recs, mode = find_recommendations(known_skills, combos)
    skill_rules = filter_rules(known_skills, rules)
    # sanity checks on shape/content, not just "did it crash"
    assert mode in ("exact", "pairwise", "none")
    if mode == "none":
        assert recs.empty
    else:
        assert not recs.empty
        assert recs["adjusted_premium"].notna().all()
        assert not set(recs["suggested_skill"]).intersection(known_skills), "recommended a skill the user already knows"
    if not skill_rules.empty:
        assert "lift" in skill_rules.columns
        for c in skill_rules["consequent_list"]:
            assert not set(c).issubset(set(known_skills)), "rule consequent is already known"
    return mode, len(recs), len(skill_rules)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suffix", default="")
    args = parser.parse_args()

    combos, singles, rules = load_data(args.suffix)
    all_skills = singles["skill"].tolist()
    random.seed(42)

    failures = []
    mode_counts = {"exact": 0, "pairwise": 0, "none": 0}

    print(f"=== Trial batch 1: every single skill alone ({len(all_skills)} trials) ===")
    for skill in all_skills:
        try:
            mode, n_recs, n_rules = run_trial([skill], combos, singles, rules)
            mode_counts[mode] += 1
        except Exception as e:
            failures.append((f"[{skill}]", e))
            print(f"  FAIL on {skill!r}: {e}")
    print(f"Done. Mode counts so far: {mode_counts}")

    print(f"\n=== Trial batch 2: 150 random skill pairs ===")
    all_pairs = list(itertools.combinations(all_skills, 2))
    sample_pairs = random.sample(all_pairs, 150)
    for pair in sample_pairs:
        try:
            mode, n_recs, n_rules = run_trial(list(pair), combos, singles, rules)
            mode_counts[mode] += 1
        except Exception as e:
            failures.append((pair, e))
            print(f"  FAIL on {pair}: {e}")
    print(f"Done. Mode counts so far: {mode_counts}")

    print(f"\n=== Trial batch 3: 50 random skill triples ===")
    triple_count = 0
    while triple_count < 50:
        triple = tuple(random.sample(all_skills, 3))
        try:
            mode, n_recs, n_rules = run_trial(list(triple), combos, singles, rules)
            mode_counts[mode] += 1
            triple_count += 1
        except Exception as e:
            failures.append((triple, e))
            print(f"  FAIL on {triple}: {e}")
            triple_count += 1
    print(f"Done. Mode counts so far: {mode_counts}")

    print(f"\n=== Trial batch 4: 20 random 4-5 skill combos (should always be pairwise or none) ===")
    for _ in range(20):
        k = random.choice([4, 5])
        combo = random.sample(all_skills, k)
        try:
            mode, n_recs, n_rules = run_trial(combo, combos, singles, rules)
            mode_counts[mode] += 1
            assert mode != "exact", f"unexpected exact match for a {k}-skill combo (max itemset size is 3)"
        except Exception as e:
            failures.append((combo, e))
            print(f"  FAIL on {combo}: {e}")

    total_trials = sum(mode_counts.values()) + len(failures)
    print(f"\n=== SUMMARY ===")
    print(f"Total trials: {total_trials}")
    print(f"Mode distribution: {mode_counts}")
    print(f"Failures: {len(failures)}")
    if failures:
        print("\nFailure details:")
        for inp, exc in failures[:10]:
            print(f"  Input: {inp}")
            print(f"  {type(exc).__name__}: {exc}")
            traceback.print_exception(type(exc), exc, exc.__traceback__, limit=3)
            print()
    else:
        print("All trials passed with no exceptions and no logical inconsistencies.")


if __name__ == "__main__":
    main()
