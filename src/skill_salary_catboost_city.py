"""
City-sensitive salary model: one CatBoostRegressor on the FULL 173-skill
vocabulary, with city_bucket/experience_bucket as CatBoost's native
categorical features (no one-hot, no manual interaction terms) -- tree
splits condition skill effects on city wherever the data supports it.
Supersedes skill_salary_regression_by_city.py's per-city Ridge approach,
which had to cap every city to the same shared 15 skills since Saint
Petersburg's ~455 training rows can't support a full 173-skill x 3-city
linear interaction model.

Same id-hash train/val/test split as the rest of this project, so the
TEST split here is identical to skill_salary_regression.py's -- the two
models are compared on the exact same held-out rows.

City-sensitivity is read off via CatBoost's native SHAP
(get_feature_importance(type="ShapValues")), computed over the full
usable population (explaining an already-evaluated model, not tuning
it -- more data here specifically helps Saint Petersburg's thin sample).

Usage: python src/skill_salary_catboost_city.py [--data PATH] [--suffix _v2]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from skill_combo_value import DATA_PATH, load_usable_postings

TEST_FRACTION = 0.2
VAL_FRACTION = 0.15
MIN_SKILL_COUNT = 45


def three_way_split(ids, test_fraction=TEST_FRACTION, val_fraction=VAL_FRACTION):
    import zlib
    h = ids.apply(lambda s: zlib.crc32(str(s).encode()) % 100)
    train_cutoff = int((1 - test_fraction - val_fraction) * 100)
    test_cutoff = int((1 - test_fraction) * 100)
    return pd.cut(h, bins=[-1, train_cutoff - 1, test_cutoff - 1, 99], labels=["train", "val", "test"])


def build_features(usable, skill_vocab):
    skill_matrix = pd.DataFrame(
        {f"skill_{s}": usable["skills_set"].apply(lambda ss: int(s in ss)) for s in skill_vocab},
        index=usable.index,
    )
    meta = usable[["city_bucket", "experience_bucket"]].copy()
    meta["has_ceiling"] = usable["has_ceiling"].astype(str)
    X = pd.concat([meta, skill_matrix], axis=1)
    return X


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--suffix", default="")
    parser.add_argument("--require-ceiling", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    usable = load_usable_postings(args.data, args.require_ceiling)
    print(f"Usable postings: {len(usable)}")

    split = three_way_split(usable["id"])
    print(split.value_counts())

    skill_counts = pd.Series([s for skills in usable["skills"] for s in skills]).value_counts()
    skill_vocab = skill_counts[skill_counts >= MIN_SKILL_COUNT].index.tolist()
    print(f"Skill features (full vocabulary, >= {MIN_SKILL_COUNT} occurrences): {len(skill_vocab)}")

    X = build_features(usable, skill_vocab)
    y = usable["salary_net"]
    cat_features = ["city_bucket", "experience_bucket", "has_ceiling"]

    train_mask, val_mask, test_mask = split == "train", split == "val", split == "test"
    train_pool = Pool(X[train_mask], y[train_mask], cat_features=cat_features)
    val_pool = Pool(X[val_mask], y[val_mask], cat_features=cat_features)
    test_pool = Pool(X[test_mask], y[test_mask], cat_features=cat_features)

    model = CatBoostRegressor(
        iterations=1000, learning_rate=0.03, depth=6,
        loss_function="RMSE", eval_metric="RMSE",
        early_stopping_rounds=50, random_seed=42, verbose=False,
    )
    model.fit(train_pool, eval_set=val_pool)
    print(f"Best iteration: {model.get_best_iteration()}")

    preds = model.predict(test_pool)
    y_test = y[test_mask]
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    print(f"\n=== Held-out test evaluation (same 847-ish test rows as the Ridge model) ===")
    print(f"CatBoost (full 173 skills + native city/experience categoricals): R2={r2:.3f}  RMSE={rmse:,.0f} RUB/mo  MAE={mae:,.0f} RUB/mo")
    print("Compare to skill_salary_regression.py's Ridge full model: R2=0.369  RMSE=69,386 RUB/mo")

    # SHAP-based city sensitivity, computed over the FULL usable population (post-hoc explanation,
    # not model selection -- more data here specifically helps the thin Saint Petersburg sample)
    full_pool = Pool(X, y, cat_features=cat_features)
    shap_raw = model.get_feature_importance(full_pool, type="ShapValues")
    shap_df = pd.DataFrame(shap_raw[:, :-1], columns=X.columns, index=X.index)

    rows = []
    for s in skill_vocab:
        col = f"skill_{s}"
        has_skill = X[col] == 1
        if has_skill.sum() < 15:
            continue
        by_city = shap_df.loc[has_skill, col].groupby(usable.loc[has_skill, "city_bucket"]).agg(["mean", "count"])
        row = {"skill": s}
        for city in ["Москва", "Санкт-Петербург", "Other"]:
            if city in by_city.index and by_city.loc[city, "count"] >= 10:
                row[f"{city}_shap"] = round(by_city.loc[city, "mean"], 0)
                row[f"{city}_n"] = int(by_city.loc[city, "count"])
            else:
                row[f"{city}_shap"] = None
                row[f"{city}_n"] = int(by_city.loc[city, "count"]) if city in by_city.index else 0
        rows.append(row)

    city_shap_df = pd.DataFrame(rows)
    shap_cols = ["Москва_shap", "Санкт-Петербург_shap", "Other_shap"]
    city_shap_df["spread"] = city_shap_df[shap_cols].max(axis=1) - city_shap_df[shap_cols].min(axis=1)
    city_shap_df = city_shap_df.dropna(subset=shap_cols).sort_values("spread", ascending=False)

    pd.set_option("display.width", 140)
    print(f"\n=== Skill x city sensitivity via SHAP (mean SHAP contribution among postings that HAVE the skill) ===")
    print(f"({len(city_shap_df)}/{len(skill_vocab)} skills had enough rows in all 3 cities to compare)")
    print(city_shap_df.head(20).to_string(index=False))

    # Same idea, one dimension: does a skill's value differ between floor-only and
    # full-range postings? (only meaningful in v3 mode -- in v1/v2, has_ceiling is
    # constant so every row falls in one bucket and this section is skipped)
    ceiling_rows = []
    if usable["has_ceiling"].nunique() > 1:
        for s in skill_vocab:
            col = f"skill_{s}"
            has_skill = X[col] == 1
            if has_skill.sum() < 15:
                continue
            by_ceiling = shap_df.loc[has_skill, col].groupby(usable.loc[has_skill, "has_ceiling"]).agg(["mean", "count"])
            if True not in by_ceiling.index or False not in by_ceiling.index:
                continue
            if by_ceiling.loc[True, "count"] < 10 or by_ceiling.loc[False, "count"] < 10:
                continue
            ceiling_rows.append({
                "skill": s,
                "floor_only_shap": round(by_ceiling.loc[False, "mean"], 0),
                "floor_only_n": int(by_ceiling.loc[False, "count"]),
                "has_ceiling_shap": round(by_ceiling.loc[True, "mean"], 0),
                "has_ceiling_n": int(by_ceiling.loc[True, "count"]),
            })
    if ceiling_rows:
        ceiling_shap_df = pd.DataFrame(ceiling_rows)
        ceiling_shap_df["spread"] = (ceiling_shap_df["floor_only_shap"] - ceiling_shap_df["has_ceiling_shap"]).abs()
        ceiling_shap_df = ceiling_shap_df.sort_values("spread", ascending=False)
        print(f"\n=== Skill x has_ceiling sensitivity via SHAP ({len(ceiling_shap_df)}/{len(skill_vocab)} skills comparable) ===")
        print("Does a skill's value differ between floor-only postings and full-range postings?")
        print(ceiling_shap_df.head(15).to_string(index=False))
        ceiling_shap_df.to_csv(ROOT / "results" / f"skill_salary_catboost_shap_by_ceiling{args.suffix}.csv", index=False)

    results_dir = ROOT / "results"
    pd.DataFrame({
        "model": ["catboost_full_vocab"], "r2": [r2], "rmse": [rmse], "mae": [mae],
        "n_train": [int(train_mask.sum())], "n_val": [int(val_mask.sum())], "n_test": [int(test_mask.sum())],
    }).to_csv(results_dir / f"skill_salary_catboost_eval{args.suffix}.csv", index=False)
    city_shap_df.to_csv(results_dir / f"skill_salary_catboost_shap_by_city{args.suffix}.csv", index=False)
    print(f"\nSaved to results/skill_salary_catboost_eval{args.suffix}.csv, skill_salary_catboost_shap_by_city{args.suffix}.csv")


if __name__ == "__main__":
    main()
