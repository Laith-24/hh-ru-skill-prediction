"""For the incremental-retraining demo only: evaluates whichever models
exist in models/nonspark_demo and models/spark_demo against the held-out
data/demo/demo_eval.parquet. Run once before and once after
retrain_on_new_data.py's update to show whether predictions on unseen
data changed. Uses PR-AUC/ROC-AUC only (threshold-independent, so a
different picked threshold each run doesn't confound the comparison).

Usage: python3 src/demo_evaluate.py before|after [--k8s] [--tag pod]
--tag appends a suffix to the output filename so a pod run doesn't
overwrite the local full-6-model results of the same name.
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.classification import (
    RandomForestClassificationModel, GBTClassificationModel, LogisticRegressionModel,
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import average_precision_score, roc_auc_score

from data_pipeline import load_and_engineer, safe_skill_name, spark_save_path, spark_load_with_retry

SPARK_MODEL_CLASSES = {
    "spark_random_forest": RandomForestClassificationModel,
    "spark_gbt": GBTClassificationModel,
    "spark_logistic_regression": LogisticRegressionModel,
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=["before", "after", "eval"], nargs="?", default="eval")
parser.add_argument("--k8s", action="store_true")
parser.add_argument("--tag", default="")
args = parser.parse_args()
stage = args.stage

if args.k8s:
    from spark_session import build_k8s_spark_session
    spark = build_k8s_spark_session("demo-evaluate")
else:
    spark = SparkSession.builder.appName("demo-evaluate").getOrCreate()
# Reuse the SAME pipeline the demo models were trained/fine-tuned with --
# fitting fresh on demo_eval.parquet would produce an incompatible feature
# space (different vocab/indices/scaler stats), the exact bug this whole
# fix was for.
demo_pipeline = spark_load_with_retry(
    lambda: PipelineModel.load(spark_save_path("models/feature_pipeline_demo", args.k8s)),
    "models/feature_pipeline_demo",
)
df_features, top_skills, _ = load_and_engineer(spark, "data/demo/demo_eval.parquet", pipeline_model=demo_pipeline)
df_features.cache()

rows = []
for skill in top_skills:
    label_col = f"label_{skill}"
    class_counts = df_features.groupBy(label_col).count().collect()
    if len(class_counts) < 2:
        continue
    for model_name, cls in SPARK_MODEL_CLASSES.items():
        model_dir = Path("models/spark_demo") / model_name / safe_skill_name(skill)
        if not model_dir.exists():
            continue
        model = spark_load_with_retry(lambda: cls.load(spark_save_path(str(model_dir), args.k8s)), str(model_dir))
        predictions = model.transform(df_features)
        pr_auc = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderPR").evaluate(predictions)
        roc_auc = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderROC").evaluate(predictions)
        rows.append({"skill": skill, "model": model_name, "pr_auc": pr_auc, "roc_auc": roc_auc})

label_cols = [f"label_{skill}" for skill in top_skills]
pdf = df_features.select(["features_array"] + label_cols).toPandas()
X = np.stack(pdf["features_array"].values)
spark.stop()

for skill in top_skills:
    y = pdf[f"label_{skill}"]
    if y.nunique() < 2:
        continue
    safe = safe_skill_name(skill)

    # Deferred imports: these three aren't installed on the k8s pod (no
    # outbound internet there) and aren't needed for a
    # --spark-only run -- importing them only when a matching model file
    # actually exists keeps this script runnable there.
    cb_path = f"models/nonspark_demo/catboost/{safe}.cbm"
    if Path(cb_path).exists():
        from catboost import CatBoostClassifier
        cb = CatBoostClassifier()
        cb.load_model(cb_path)
        probs = cb.predict_proba(X)[:, 1]
        rows.append({"skill": skill, "model": "catboost", "pr_auc": average_precision_score(y, probs),
                     "roc_auc": roc_auc_score(y, probs)})

    xgb_path = f"models/nonspark_demo/xgboost/{safe}.json"
    if Path(xgb_path).exists():
        from xgboost import XGBClassifier
        xgb = XGBClassifier()
        xgb.load_model(xgb_path)
        probs = xgb.predict_proba(X)[:, 1]
        rows.append({"skill": skill, "model": "xgboost", "pr_auc": average_precision_score(y, probs),
                     "roc_auc": roc_auc_score(y, probs)})

    lgb_path = f"models/nonspark_demo/lightgbm/{safe}.txt"
    if Path(lgb_path).exists():
        from lightgbm import Booster
        booster = Booster(model_file=lgb_path)
        probs = booster.predict(X)
        rows.append({"skill": skill, "model": "lightgbm", "pr_auc": average_precision_score(y, probs),
                     "roc_auc": roc_auc_score(y, probs)})

out = pd.DataFrame(rows)
out_name = f"demo_eval_{stage}_{args.tag}.csv" if args.tag else f"demo_eval_{stage}.csv"
out.to_csv(f"results/{out_name}", index=False)
print(f"=== Demo eval ({stage}), mean across skills with >=2 classes in this 600-row eval set ===")
print(out.groupby("model")[["pr_auc", "roc_auc"]].mean().round(4))
print(f"(skills with usable signal: {out['skill'].nunique()} / {len(top_skills)})")
