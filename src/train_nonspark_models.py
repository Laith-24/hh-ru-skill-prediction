"""
Non-Spark model training path: every model in NONSPARK_MODEL_FACTORIES
(CatBoost, XGBoost, LightGBM), same id-hash train/val/test split as
train_spark_native.py. Saves each fitted model so retrain_on_new_data.py
can fine-tune from it later.

Threshold picked per skill/model on held-out "val" (maximizes F1), not
hardcoded at 0.5 -- with imbalanced labels a fixed threshold collapses F1
regardless of how good the underlying probabilities are. Also trains
with balanced sample_weight, a separate, complementary mechanism (changes
what the model learns, not just where to cut its output).

Usage:
    python3 src/train_nonspark_models.py --data data/vacancies.parquet
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.ml.functions import vector_to_array
from pyspark.sql import SparkSession, functions as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_sample_weight

from data_pipeline import SKILL_LIST_PATH, TITLE_FEATURE_COLS, load_and_engineer, safe_skill_name, best_f1_threshold
from models_config import NONSPARK_MODEL_FACTORIES, NONSPARK_MODEL_EXTENSIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_tuned_params(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def save_model(model, model_name, skill, models_dir):
    ext = NONSPARK_MODEL_EXTENSIONS[model_name]
    path = Path(models_dir) / model_name / f"{safe_skill_name(skill)}.{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    if model_name == "lightgbm":
        model.booster_.save_model(str(path))
    else:
        model.save_model(str(path))  # catboost, xgboost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the raw hh.ru vacancy parquet file")
    parser.add_argument("--top-n-skills", type=int, default=20)
    parser.add_argument("--skill-list-path", default=SKILL_LIST_PATH,
                         help="Cached skill list. Use a separate file for a different --top-n-skills, "
                              "otherwise the existing cached list is reused whatever its size")
    parser.add_argument("--features", choices=["full", "title"], default="full",
                         help="'title' trains on the title's TF-IDF and Word2Vec vectors plus its seniority flags only, "
                              "i.e. on exactly what the title-only demo can supply. Give it its own --output, "
                              "--models-dir and --pipeline-path so the full-feature files are not overwritten")
    parser.add_argument("--output", default="results/nonspark_results.csv")
    parser.add_argument("--models-dir", default="models/nonspark")
    parser.add_argument("--tuned-params", default="configs/tuned_params.json")
    parser.add_argument("--pipeline-path", default="models/feature_pipeline",
                         help="Where to save the fitted feature pipeline, for retrain_on_new_data.py to reuse")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("hh-ru-skill-prediction-nonspark").getOrCreate()

    df_features, top_skills, pipeline_model = load_and_engineer(
        spark, args.data, args.top_n_skills, skill_list_path=args.skill_list_path
    )
    pipeline_model.write().overwrite().save(args.pipeline_path)
    logger.info(f"Feature pipeline saved to {args.pipeline_path}")

    label_cols = [f"label_{skill}" for skill in top_skills]
    if args.features == "title":
        feature_col = F.concat(*[vector_to_array(F.col(c)) for c in TITLE_FEATURE_COLS]).alias("features_array")
    else:
        feature_col = F.col("features_array")
    pdf = df_features.select("split", feature_col, *label_cols).toPandas()
    spark.stop()  # done with Spark -- everything below is plain pandas/numpy

    X = np.stack(pdf["features_array"].values)
    train_mask = (pdf["split"] == "train").to_numpy()
    val_mask = (pdf["split"] == "val").to_numpy()
    test_mask = (pdf["split"] == "test").to_numpy()
    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]

    tuned = load_tuned_params(args.tuned_params)
    results = []

    for skill in top_skills:
        label_col = f"label_{skill}"
        y_train = pdf.loc[train_mask, label_col]
        y_val = pdf.loc[val_mask, label_col]
        y_test = pdf.loc[test_mask, label_col]

        if y_train.nunique() < 2:
            logger.warning(f"Skipping '{skill}': only one class present in training split")
            continue

        sample_weight = compute_sample_weight("balanced", y_train)

        for model_name, factory in NONSPARK_MODEL_FACTORIES.items():
            model = factory(tuned.get(model_name))
            model.fit(X_train, y_train, sample_weight=sample_weight)
            val_probs = model.predict_proba(X_val)[:, 1]
            threshold = best_f1_threshold(y_val.to_numpy(), val_probs)

            probs = model.predict_proba(X_test)[:, 1]
            preds = (probs >= threshold).astype(int)

            save_model(model, model_name, skill, args.models_dir)

            row = {
                "skill": skill,
                "model": model_name,
                "pr_auc": average_precision_score(y_test, probs),
                "roc_auc": roc_auc_score(y_test, probs) if y_test.nunique() > 1 else None,
                "f1": f1_score(y_test, preds, zero_division=0),
                "precision": precision_score(y_test, preds, zero_division=0),
                "recall": recall_score(y_test, preds, zero_division=0),
                "threshold": round(threshold, 4),
            }
            results.append(row)
            logger.info(f"{skill} / {model_name}: PR-AUC={row['pr_auc']:.3f}  F1={row['f1']:.3f}  thr={threshold:.3f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(args.output, index=False)
    logger.info(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
