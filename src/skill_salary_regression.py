"""
Held-out evaluation metric for the Skill Combination Value Calculator.
skill_combo_value.py's premiums are descriptive (mined and priced on the
same rows, no train/test split) -- this adds that: fits a salary Ridge
regression on TRAIN, evaluates a baseline (experience+city) against the
full model (+skills) on TEST, neither has seen. The R2 gap between them
is the headline: how much skills explain beyond seniority/city alone,
measured on unseen data.

Reuses skill_combo_value.load_usable_postings() for the same usable
population, and this project's id-hash split (crc32(id) % 100).

Usage: python src/skill_salary_regression.py [--data PATH] [--suffix _v2]
"""
import argparse
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from skill_combo_value import DATA_PATH, load_usable_postings

TEST_FRACTION = 0.2
MIN_SKILL_COUNT = 45  # same frequency floor as FPGrowth's minSupport=0.01 on ~4,500 rows


def id_hash_split(ids, test_fraction=TEST_FRACTION):
    h = ids.apply(lambda s: zlib.crc32(str(s).encode()) % 100)
    cutoff = int((1 - test_fraction) * 100)
    return h < cutoff  # True = train


def build_features(usable, skill_vocab):
    city_dummies = pd.get_dummies(usable["city_bucket"], prefix="city", dtype=int)
    exp_dummies = pd.get_dummies(usable["experience_bucket"], prefix="exp", dtype=int)
    ceiling_dummy = usable[["has_ceiling"]].astype(int)
    skill_matrix = pd.DataFrame(
        {f"skill_{s}": usable["skills_set"].apply(lambda ss: int(s in ss)) for s in skill_vocab},
        index=usable.index,
    )
    baseline_X = pd.concat([city_dummies, exp_dummies, ceiling_dummy], axis=1)
    full_X = pd.concat([baseline_X, skill_matrix], axis=1)
    return baseline_X, full_X


def evaluate(X_train, y_train, X_test, y_test, label):
    model = RidgeCV(alphas=np.logspace(-2, 4, 30))
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    print(f"{label}: R2={r2:.3f}  RMSE={rmse:,.0f} RUB/mo  MAE={mae:,.0f} RUB/mo  (alpha={model.alpha_:.2f})")
    return model, r2, rmse, mae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--suffix", default="")
    parser.add_argument("--require-ceiling", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    usable = load_usable_postings(args.data, args.require_ceiling)
    print(f"Usable postings: {len(usable)}")

    train_mask = id_hash_split(usable["id"])
    print(f"Train: {int(train_mask.sum())}  Test: {int((~train_mask).sum())}")

    skill_counts = pd.Series([s for skills in usable["skills"] for s in skills]).value_counts()
    skill_vocab = skill_counts[skill_counts >= MIN_SKILL_COUNT].index.tolist()
    print(f"Skill features included (>= {MIN_SKILL_COUNT} occurrences in the full usable set): {len(skill_vocab)}")

    baseline_X, full_X = build_features(usable, skill_vocab)
    y = usable["salary_net"]

    X_train_base, X_test_base = baseline_X[train_mask], baseline_X[~train_mask]
    X_train_full, X_test_full = full_X[train_mask], full_X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    print("\n=== Held-out evaluation (test rows the model never saw while fitting) ===")
    _, base_r2, base_rmse, base_mae = evaluate(
        X_train_base, y_train, X_test_base, y_test, "Baseline (experience + city only)"
    )
    full_model, full_r2, full_rmse, full_mae = evaluate(
        X_train_full, y_train, X_test_full, y_test, "Full model (+ skills)"
    )

    print(f"\nSkills add {full_r2 - base_r2:+.3f} R2 on held-out data vs. the seniority/city-only baseline")
    print(f"RMSE improvement: {base_rmse - full_rmse:+,.0f} RUB/mo ({(base_rmse - full_rmse) / base_rmse:.1%} lower error)")

    coefs = pd.Series(full_model.coef_, index=full_X.columns)
    skill_coefs = coefs[[c for c in coefs.index if c.startswith("skill_")]].sort_values(ascending=False)
    print("\nTop 10 skill coefficients (regression-adjusted, controlling for experience + city simultaneously):")
    print(skill_coefs.head(10).to_string())
    print("\nBottom 10:")
    print(skill_coefs.tail(10).to_string())

    results_dir = ROOT / "results"
    pd.DataFrame({
        "model": ["baseline_experience_city", "full_with_skills"],
        "r2": [base_r2, full_r2],
        "rmse": [base_rmse, full_rmse],
        "mae": [base_mae, full_mae],
        "n_train": [int(train_mask.sum())] * 2,
        "n_test": [int((~train_mask).sum())] * 2,
    }).to_csv(results_dir / f"skill_salary_regression_eval{args.suffix}.csv", index=False)
    skill_coefs.to_csv(results_dir / f"skill_salary_regression_coefs{args.suffix}.csv", header=["coef"])
    print(f"\nSaved to results/skill_salary_regression_eval{args.suffix}.csv, skill_salary_regression_coefs{args.suffix}.csv")


if __name__ == "__main__":
    main()
