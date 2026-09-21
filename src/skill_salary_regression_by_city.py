"""
City-sensitive extension of skill_salary_regression.py: fits a SEPARATE
Ridge regression per city bucket instead of one pooled model, so each
city gets its own skill coefficients (the pooled model only lets city
shift the baseline additively).

Scoped to the top 15 skills, shared across all three city models (not a
different top-15 per city) -- Saint Petersburg's ~455 training rows can't
support the pooled model's full 173-skill vocabulary without mostly
noise, and using the same skills everywhere keeps the comparison fair.

Reuses skill_salary_regression.py's data loading, split, and evaluate().

Usage: python src/skill_salary_regression_by_city.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from skill_combo_value import load_usable_postings
from skill_salary_regression import id_hash_split, evaluate

TOP_N_SKILLS = 15


def build_city_features(usable, skill_vocab):
    exp_dummies = pd.get_dummies(usable["experience_bucket"], prefix="exp", dtype=int)
    skill_matrix = pd.DataFrame(
        {f"skill_{s}": usable["skills_set"].apply(lambda ss: int(s in ss)) for s in skill_vocab},
        index=usable.index,
    )
    baseline_X = exp_dummies
    full_X = pd.concat([exp_dummies, skill_matrix], axis=1)
    return baseline_X, full_X


def main():
    usable = load_usable_postings()
    train_mask = id_hash_split(usable["id"])

    skill_counts = pd.Series([s for skills in usable["skills"] for s in skills]).value_counts()
    skill_vocab = skill_counts.head(TOP_N_SKILLS).index.tolist()
    print(f"Shared skill vocabulary across all three city models (top {TOP_N_SKILLS} most frequent overall):")
    print(", ".join(skill_vocab))

    coef_rows = {}
    for city in ["Москва", "Санкт-Петербург", "Other"]:
        city_mask = usable["city_bucket"] == city
        city_usable = usable[city_mask]
        city_train = train_mask[city_mask]

        baseline_X, full_X = build_city_features(city_usable, skill_vocab)
        y = city_usable["salary_net"]

        n_train, n_test = int(city_train.sum()), int((~city_train).sum())
        print(f"\n=== {city} (n_train={n_train}, n_test={n_test}) ===")
        if n_train < 100 or n_test < 20:
            print("Too few rows to fit/evaluate reliably here -- skipping.")
            continue

        _, base_r2, base_rmse, _ = evaluate(
            baseline_X[city_train], y[city_train], baseline_X[~city_train], y[~city_train],
            "  Baseline (experience only)",
        )
        full_model, full_r2, full_rmse, _ = evaluate(
            full_X[city_train], y[city_train], full_X[~city_train], y[~city_train],
            "  Full (+ skills)",
        )
        print(f"  Skills add {full_r2 - base_r2:+.3f} R2 in {city} specifically")

        coefs = pd.Series(full_model.coef_, index=full_X.columns)
        coef_rows[city] = coefs[[f"skill_{s}" for s in skill_vocab]]

    comparison = pd.DataFrame(coef_rows)
    comparison.index = [i.replace("skill_", "") for i in comparison.index]
    comparison["spread"] = comparison.max(axis=1) - comparison.min(axis=1)
    comparison = comparison.sort_values("spread", ascending=False)

    pd.set_option("display.width", 120)
    print("\n=== Same skill, different city -- premium comparison (RUB/mo, regression-adjusted for experience) ===")
    print(comparison.round(0).to_string())

    out_path = ROOT / "results" / "skill_salary_by_city.csv"
    comparison.to_csv(out_path)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
