"""
Pod version of demo_finetune_single_process.py, scoped to the 3 Spark-
native models only (catboost/xgboost/lightgbm aren't installed on the
pod). Same root problem as the local version
(PipelineModel.load() across separate processes is unreliable), different
cause: confirmed directly on the pod that a saved pipeline/model's
metadata/ directory holds only an empty _SUCCESS marker, no content file
-- not a slow-to-appear write, a structurally missing one, because
executor pods don't share the driver's NFS mount, so the metadata RDD's
single partition can be written to an executor's own ephemeral local
disk instead of shared storage. No retry fixes that. Sidesteps it the
same way the local version does: never calls .load() on anything saved
mid-run, keeps every model as an in-memory object across all four demo
steps in one Spark session/process.

Usage (via kubectl exec, see deploy/run_finetune_demo_on_pod.sh):
    python3 src/demo_finetune_single_process_pod.py
"""
import json
import logging
from pathlib import Path

import pandas as pd
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from data_pipeline import load_and_engineer, add_class_weight_column
from models_config import SPARK_MODEL_FACTORIES
from spark_session import build_k8s_spark_session

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OLD_DATA = "data/demo/demo_old.parquet"
NEW_DATA = "data/demo/demo_new.parquet"
EVAL_DATA = "data/demo/demo_eval.parquet"
TUNED_PARAMS_PATH = "configs/tuned_params.json"
TOP_N_SKILLS = 20


def load_tuned_params():
    p = Path(TUNED_PARAMS_PATH)
    return json.loads(p.read_text()) if p.exists() else {}


def evaluable_skills(pdf, top_skills, label_cols):
    """Skills where this particular slice of data has both classes present
    -- PR-AUC/ROC-AUC are undefined otherwise."""
    return [s for s, col in zip(top_skills, label_cols) if pdf[col].nunique() >= 2]


def evaluate(spark_models, df_features, skills):
    rows = []
    for skill in skills:
        label_col = f"label_{skill}"
        for model_name, model in spark_models.get(skill, {}).items():
            predictions = model.transform(df_features)
            pr_auc = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderPR").evaluate(predictions)
            roc_auc = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderROC").evaluate(predictions)
            rows.append({"skill": skill, "model": model_name, "pr_auc": pr_auc, "roc_auc": roc_auc})
    return rows


def fit_all(features_df, skills, tuned):
    models = {}
    for skill in skills:
        label_col = f"label_{skill}"
        weighted_df = add_class_weight_column(features_df, label_col)
        models[skill] = {
            name: factory("scaled_features", label_col, tuned.get(name)).fit(weighted_df)
            for name, factory in SPARK_MODEL_FACTORIES.items()
        }
    return models


def main():
    tuned = load_tuned_params()
    spark = build_k8s_spark_session("hh-ru-demo-finetune-pod")

    logger.info("Step A: fitting feature pipeline + baseline Spark models on demo_old ...")
    old_features, top_skills, pipeline_model = load_and_engineer(spark, OLD_DATA, TOP_N_SKILLS)
    old_features.cache()
    label_cols = [f"label_{s}" for s in top_skills]
    old_pdf = old_features.select(label_cols).toPandas()
    trainable_skills = evaluable_skills(old_pdf, top_skills, label_cols)
    logger.info(f"{len(trainable_skills)}/{len(top_skills)} skills have both classes in demo_old")

    spark_models = fit_all(old_features, trainable_skills, tuned)
    logger.info("Baseline Spark models trained (in memory).")

    logger.info("Step B: evaluating BEFORE fine-tuning on demo_eval ...")
    eval_features, _, _ = load_and_engineer(spark, EVAL_DATA, TOP_N_SKILLS, pipeline_model=pipeline_model)
    eval_features.cache()
    eval_pdf = eval_features.select(label_cols).toPandas()
    eval_skills = [s for s in trainable_skills if s in evaluable_skills(eval_pdf, top_skills, label_cols)]

    before_rows = evaluate(spark_models, eval_features, eval_skills)
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(before_rows).to_csv("results/demo_eval_before_pod.csv", index=False)
    logger.info(f"Wrote results/demo_eval_before_pod.csv ({len(before_rows)} rows)")

    logger.info("Step C: retraining from scratch on demo_old + demo_new (Spark ML has no incremental fit) ...")
    new_features, _, _ = load_and_engineer(spark, NEW_DATA, TOP_N_SKILLS, pipeline_model=pipeline_model)
    combined_features = old_features.unionByName(new_features, allowMissingColumns=True)
    spark_models = fit_all(combined_features, trainable_skills, tuned)
    logger.info("Retraining complete.")

    logger.info("Step D: evaluating AFTER retraining on the SAME demo_eval ...")
    after_rows = evaluate(spark_models, eval_features, eval_skills)
    pd.DataFrame(after_rows).to_csv("results/demo_eval_after_pod.csv", index=False)
    logger.info(f"Wrote results/demo_eval_after_pod.csv ({len(after_rows)} rows)")

    spark.stop()
    logger.info("Demo complete. Compare results/demo_eval_before_pod.csv vs results/demo_eval_after_pod.csv")


if __name__ == "__main__":
    main()
