"""Computes the full metric set (accuracy, precision, recall, F1) per
skill/model on the held-out test set for all 6 models, merging in the
already-computed PR-AUC/ROC-AUC into one combined table. Reports the
majority-class baseline too, since accuracy alone is misleading on
imbalanced labels.

Uses each model's saved tuned threshold (results/*_results.csv), never a
Spark model's own transform() prediction column -- that applies Spark's
internal 0.5-on-probability rule, not the tuned one, and gives a
different, wrong-looking answer for the same model.

Always reads the live results/*.csv files, never a "_final"-suffixed
snapshot, so a stale threshold never gets paired with freshly retrained
models. Uses the Column API (F.col/F.avg), never raw SQL text, since
skill names with spaces/Cyrillic break unquoted SQL identifiers.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.classification import (
    RandomForestClassificationModel, GBTClassificationModel, LogisticRegressionModel,
)
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import Booster

from data_pipeline import load_and_engineer, safe_skill_name

SPARK_MODEL_CLASSES = {
    "spark_random_forest": RandomForestClassificationModel,
    "spark_gbt": GBTClassificationModel,
    "spark_logistic_regression": LogisticRegressionModel,
}

spark = SparkSession.builder.appName("compute-accuracy").getOrCreate()
# Reuse the exact pipeline the final models were trained with, rather than
# fitting a fresh one here -- guarantees the same feature space instead of
# relying on "should be deterministic" reasoning (see load_and_engineer's
# pipeline_model docstring for why independent fits are risky in general).
# Falls back to fitting fresh if no saved pipeline exists yet (e.g. models
# trained before train_*.py started saving one) -- still correct by
# determinism for same-file calls, just without the extra guarantee.
try:
    _saved_pipeline = PipelineModel.load("models/feature_pipeline")
except Exception:
    logging.warning("No saved models/feature_pipeline found -- fitting fresh (fine for same-file calls).")
    _saved_pipeline = None
df_features, top_skills, _ = load_and_engineer(spark, "data/vacancies.parquet", pipeline_model=_saved_pipeline)
df_features.cache()
test_df = df_features.filter("split = 'test'")

nonspark_thresholds = pd.read_csv("results/nonspark_results.csv").set_index(["skill", "model"])["threshold"]
spark_thresholds = pd.read_csv("results/spark_results.csv").set_index(["skill", "model"])["threshold"]

rows = []

# --- Spark-native: apply the SAME tuned threshold train_spark_native.py used ---
for skill in top_skills:
    label_col = f"label_{skill}"
    positive_rate = test_df.select(F.avg(F.col(label_col)).alias("r")).collect()[0]["r"] or 0.0
    majority_baseline = max(positive_rate, 1 - positive_rate)

    for model_name, cls in SPARK_MODEL_CLASSES.items():
        model_dir = Path("models/spark") / model_name / safe_skill_name(skill)
        if not model_dir.exists():
            continue
        thr = spark_thresholds.get((skill, model_name), 0.5)
        model = cls.load(str(model_dir))
        preds = model.transform(test_df).select(
            F.col(label_col).cast("double").alias("y"),
            vector_to_array("probability")[1].alias("prob"),
        )
        counts = preds.select(
            F.sum(((F.col("y") == 1) & (F.col("prob") >= thr)).cast("double")).alias("tp"),
            F.sum(((F.col("y") == 0) & (F.col("prob") >= thr)).cast("double")).alias("fp"),
            F.sum(((F.col("y") == 1) & (F.col("prob") < thr)).cast("double")).alias("fn"),
            F.sum(((F.col("y") == 0) & (F.col("prob") < thr)).cast("double")).alias("tn"),
        ).collect()[0]
        tp, fp, fn, tn = counts["tp"], counts["fp"], counts["fn"], counts["tn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        rows.append({
            "skill": skill, "model": model_name, "accuracy": accuracy,
            "precision": precision, "recall": recall, "f1": f1,
            "majority_baseline_acc": majority_baseline, "positive_rate": positive_rate,
        })

# --- Non-Spark: bring test features to pandas once, reuse saved threshold ---
label_cols = [f"label_{skill}" for skill in top_skills]
pdf = test_df.select(["features_array"] + label_cols).toPandas()
X_test = np.stack(pdf["features_array"].values)


def _eval_nonspark(model_name, y_test, preds, positive_rate, majority_baseline):
    rows.append({
        "skill": skill, "model": model_name,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "majority_baseline_acc": majority_baseline, "positive_rate": positive_rate,
    })


for skill in top_skills:
    y_test = pdf[f"label_{skill}"]
    positive_rate = y_test.mean()
    majority_baseline = max(positive_rate, 1 - positive_rate)
    safe = safe_skill_name(skill)

    cb_path = f"models/nonspark/catboost/{safe}.cbm"
    if Path(cb_path).exists():
        cb = CatBoostClassifier()
        cb.load_model(cb_path)
        thr = nonspark_thresholds.get((skill, "catboost"), 0.5)
        preds = (cb.predict_proba(X_test)[:, 1] >= thr).astype(int)
        _eval_nonspark("catboost", y_test, preds, positive_rate, majority_baseline)

    xgb_path = f"models/nonspark/xgboost/{safe}.json"
    if Path(xgb_path).exists():
        xgb = XGBClassifier()
        xgb.load_model(xgb_path)
        thr = nonspark_thresholds.get((skill, "xgboost"), 0.5)
        preds = (xgb.predict_proba(X_test)[:, 1] >= thr).astype(int)
        _eval_nonspark("xgboost", y_test, preds, positive_rate, majority_baseline)

    lgb_path = f"models/nonspark/lightgbm/{safe}.txt"
    if Path(lgb_path).exists():
        booster = Booster(model_file=lgb_path)
        thr = nonspark_thresholds.get((skill, "lightgbm"), 0.5)
        preds = (booster.predict(X_test) >= thr).astype(int)
        _eval_nonspark("lightgbm", y_test, preds, positive_rate, majority_baseline)

spark.stop()

out = pd.DataFrame(rows)

# Merge in PR-AUC/ROC-AUC already computed by the training scripts
spark_metrics = pd.read_csv("results/spark_results.csv")[["skill", "model", "pr_auc", "roc_auc"]]
nonspark_metrics = pd.read_csv("results/nonspark_results.csv")[["skill", "model", "pr_auc", "roc_auc"]]
auc = pd.concat([spark_metrics, nonspark_metrics], ignore_index=True)
out = out.merge(auc, on=["skill", "model"], how="left")

out.to_csv("results/full_metrics_report.csv", index=False)

summary = out.groupby("model")[["pr_auc", "roc_auc", "precision", "recall", "f1", "accuracy", "majority_baseline_acc"]].mean()
summary = summary.round(4).sort_values("pr_auc", ascending=False)
print("=== Full metric summary, mean across 20 skills ===")
print(summary.to_string())
