"""
Cross-checks CatBoost's salary-regression result against XGBoost and
LightGBM on the same held-out split, same hyperparameter budget (1000
rounds, lr=0.03, depth 6, early stop 50), so no library wins on capacity
alone. XGBoost's categoricals are one-hot encoded rather than native --
this environment's XGBoost 3.1.1 doesn't recognize pandas `category`
dtype (confirmed via an isolated repro, not a usage mistake); LightGBM
gets genuine native categorical treatment like CatBoost.

Performance cross-check only, not a full SHAP redo for all three --
CatBoost stays the interpretability model regardless of result (see
skill_salary_catboost_city.py).

Usage: python src/skill_salary_gbm_comparison.py [--data PATH] [--suffix _v3]
    [--no-require-ceiling]
"""
import argparse
import sys
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from skill_combo_value import DATA_PATH, load_usable_postings
from skill_salary_catboost_city import MIN_SKILL_COUNT, three_way_split

CAT_COLS = ["city_bucket", "experience_bucket", "has_ceiling"]
N_ROUNDS = 1000
LEARNING_RATE = 0.03
MAX_DEPTH = 6
EARLY_STOPPING = 50


def build_features(usable, skill_vocab):
    # Index-based names (skill_0, ...), not raw skill names -- LightGBM
    # rejects some real skill names as feature names (special characters).
    skill_matrix = pd.DataFrame(
        {f"skill_{i}": usable["skills_set"].apply(lambda ss: int(s in ss)) for i, s in enumerate(skill_vocab)},
        index=usable.index,
    )
    meta = usable[["city_bucket", "experience_bucket"]].copy()
    meta["has_ceiling"] = usable["has_ceiling"].astype(str)
    return meta, skill_matrix


def evaluate_catboost(meta, skill_matrix, y, train_mask, val_mask, test_mask):
    X = pd.concat([meta, skill_matrix], axis=1)
    train_pool = Pool(X[train_mask], y[train_mask], cat_features=CAT_COLS)
    val_pool = Pool(X[val_mask], y[val_mask], cat_features=CAT_COLS)
    test_pool = Pool(X[test_mask], y[test_mask], cat_features=CAT_COLS)
    model = CatBoostRegressor(
        iterations=N_ROUNDS, learning_rate=LEARNING_RATE, depth=MAX_DEPTH,
        loss_function="RMSE", eval_metric="RMSE",
        early_stopping_rounds=EARLY_STOPPING, random_seed=42, verbose=False,
    )
    model.fit(train_pool, eval_set=val_pool)
    return model.predict(test_pool), model.get_best_iteration()


def _categorical_X(meta, skill_matrix):
    meta_cat = meta.copy()
    for c in CAT_COLS:
        meta_cat[c] = meta_cat[c].astype("category")
    return pd.concat([meta_cat, skill_matrix], axis=1)


def evaluate_xgboost(meta, skill_matrix, y, train_mask, val_mask, test_mask):
    # One-hot, not native -- see module docstring (XGBoost 3.1.1 doesn't
    # recognize category dtype here). Only 9 dummy columns, cheap.
    onehot = pd.get_dummies(meta, columns=CAT_COLS, dtype=float)
    X = pd.concat([onehot, skill_matrix], axis=1)
    model = XGBRegressor(
        n_estimators=N_ROUNDS, learning_rate=LEARNING_RATE, max_depth=MAX_DEPTH,
        tree_method="hist",
        early_stopping_rounds=EARLY_STOPPING, eval_metric="rmse", random_state=42,
    )
    model.fit(X[train_mask], y[train_mask], eval_set=[(X[val_mask], y[val_mask])], verbose=False)
    return model.predict(X[test_mask]), model.best_iteration


def evaluate_lightgbm(meta, skill_matrix, y, train_mask, val_mask, test_mask):
    X = _categorical_X(meta, skill_matrix)
    model = LGBMRegressor(
        n_estimators=N_ROUNDS, learning_rate=LEARNING_RATE, max_depth=MAX_DEPTH,
        random_state=42, verbosity=-1,
    )
    model.fit(
        X[train_mask], y[train_mask],
        eval_set=[(X[val_mask], y[val_mask])],
        categorical_feature=CAT_COLS,
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )
    return model.predict(X[test_mask]), model.best_iteration_


def report(name, y_test, preds, best_iteration):
    r2 = r2_score(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    mae = mean_absolute_error(y_test, preds)
    print(f"{name}: R2={r2:.3f}  RMSE={rmse:,.0f} RUB/mo  MAE={mae:,.0f} RUB/mo  (best_iteration={best_iteration})")
    return {"model": name, "r2": r2, "rmse": rmse, "mae": mae, "best_iteration": best_iteration}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--suffix", default="")
    parser.add_argument("--require-ceiling", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    usable = load_usable_postings(args.data, args.require_ceiling)
    print(f"Usable postings: {len(usable)}")

    split = three_way_split(usable["id"])
    train_mask, val_mask, test_mask = split == "train", split == "val", split == "test"
    print(split.value_counts())

    skill_counts = pd.Series([s for skills in usable["skills"] for s in skills]).value_counts()
    skill_vocab = skill_counts[skill_counts >= MIN_SKILL_COUNT].index.tolist()
    print(f"Skill features: {len(skill_vocab)}")

    meta, skill_matrix = build_features(usable, skill_vocab)
    y = usable["salary_net"]
    y_test = y[test_mask]

    print("\n=== Held-out test evaluation, same split, same native-categorical treatment, same hyperparameter budget ===")
    results = []
    preds, it = evaluate_catboost(meta, skill_matrix, y, train_mask, val_mask, test_mask)
    results.append(report("CatBoost", y_test, preds, it))

    preds, it = evaluate_xgboost(meta, skill_matrix, y, train_mask, val_mask, test_mask)
    results.append(report("XGBoost", y_test, preds, it))

    preds, it = evaluate_lightgbm(meta, skill_matrix, y, train_mask, val_mask, test_mask)
    results.append(report("LightGBM", y_test, preds, it))

    result_df = pd.DataFrame(results).sort_values("r2", ascending=False)
    print("\n=== Ranked ===")
    print(result_df.to_string(index=False))

    out_path = ROOT / "results" / f"skill_salary_gbm_comparison{args.suffix}.csv"
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
