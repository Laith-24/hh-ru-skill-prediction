"""
Retrain / fine-tune all models on new data.

- Non-Spark (CatBoost/XGBoost/LightGBM): TRUE incremental fine-tuning --
  continues training from the saved model using only the new rows
  (init_model=/xgb_model=). Doesn't need the old data file around.
- Spark (RandomForest/GBT/LogisticRegression): no incremental fit in
  Spark ML, so retrained from scratch. Pass --old-data to retrain on
  old+new combined (recommended).
- --full-retrain forces every model, including non-Spark, to retrain
  from scratch on old+new instead of fine-tuning (requires --old-data).
- --k8s runs on the real cluster; pair with --spark-only since the pod
  has no catboost/xgboost/lightgbm installed.

Always reuses the SAME saved feature pipeline for both old_data and
new_data rather than fitting a fresh one per file -- two independently-
fit pipelines produce "scaled_features" columns with the same name but
incompatible meaning (different vocab/indices/scaler stats), so unioning
or fine-tuning across them would silently train on garbage.

Usage:
    python3 src/retrain_on_new_data.py --new-data data/new_vacancies.parquet \\
        --old-data data/vacancies.parquet
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from sklearn.utils.class_weight import compute_sample_weight

from data_pipeline import load_and_engineer, safe_skill_name, add_class_weight_column, spark_save_path, spark_load_with_retry
from models_config import SPARK_MODEL_FACTORIES, NONSPARK_MODEL_FACTORIES, NONSPARK_MODEL_EXTENSIONS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_tuned_params(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


def retrain_spark_models(spark, pipeline_model, new_data_path, old_data_path, top_n_skills, models_dir, tuned, k8s):
    df, top_skills, _ = load_and_engineer(spark, new_data_path, top_n_skills, pipeline_model=pipeline_model)
    if old_data_path:
        old_df, _, _ = load_and_engineer(spark, old_data_path, top_n_skills, pipeline_model=pipeline_model)
        df = df.unionByName(old_df, allowMissingColumns=True)

    for skill in top_skills:
        label_col = f"label_{skill}"
        weighted_df = add_class_weight_column(df, label_col)
        for model_name, factory in SPARK_MODEL_FACTORIES.items():
            classifier = factory("scaled_features", label_col, tuned.get(model_name))
            model = classifier.fit(weighted_df)
            model_dir = spark_save_path(str(Path(models_dir) / model_name / safe_skill_name(skill)), k8s)
            model.write().overwrite().save(model_dir)
            logger.info(f"Retrained {model_name} / {skill} on {df.count()} rows")


def retrain_nonspark_models(spark, pipeline_model, new_data_path, old_data_path, top_n_skills, models_dir, tuned, full_retrain):
    df, top_skills, _ = load_and_engineer(spark, new_data_path, top_n_skills, pipeline_model=pipeline_model)
    label_cols = [f"label_{s}" for s in top_skills]
    new_pdf = df.select(["features_array"] + label_cols).toPandas()

    if old_data_path and full_retrain:
        old_df, _, _ = load_and_engineer(spark, old_data_path, top_n_skills, pipeline_model=pipeline_model)
        old_pdf = old_df.select(["features_array"] + label_cols).toPandas()
        pdf = pd.concat([old_pdf, new_pdf], ignore_index=True)
    else:
        pdf = new_pdf

    X = np.stack(pdf["features_array"].values)

    for skill in top_skills:
        label_col = f"label_{skill}"
        y = pdf[label_col]
        if y.nunique() < 2:
            logger.warning(f"Skipping '{skill}': only one class present in this batch")
            continue

        sample_weight = compute_sample_weight("balanced", y)

        for model_name, factory in NONSPARK_MODEL_FACTORIES.items():
            ext = NONSPARK_MODEL_EXTENSIONS[model_name]
            model_path = Path(models_dir) / model_name / f"{safe_skill_name(skill)}.{ext}"
            model = factory(tuned.get(model_name))

            if full_retrain or not model_path.exists():
                model.fit(X, y, sample_weight=sample_weight)
                logger.info(f"Retrained {model_name} / {skill} from scratch ({len(y)} rows)")
            else:
                if model_name == "catboost":
                    model.fit(X, y, sample_weight=sample_weight, init_model=str(model_path))
                elif model_name == "xgboost":
                    model.fit(X, y, sample_weight=sample_weight, xgb_model=str(model_path))
                elif model_name == "lightgbm":
                    model.fit(X, y, sample_weight=sample_weight, init_model=str(model_path))
                logger.info(f"Fine-tuned {model_name} / {skill} on {len(y)} new rows only")

            model_path.parent.mkdir(parents=True, exist_ok=True)
            if model_name == "lightgbm":
                model.booster_.save_model(str(model_path))
            else:
                model.save_model(str(model_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--new-data", required=True)
    parser.add_argument("--old-data", default=None, help="Required for Spark models, and for --full-retrain")
    parser.add_argument("--top-n-skills", type=int, default=20)
    parser.add_argument("--spark-models-dir", default="models/spark")
    parser.add_argument("--nonspark-models-dir", default="models/nonspark")
    parser.add_argument("--tuned-params", default="configs/tuned_params.json")
    parser.add_argument("--pipeline-path", default="models/feature_pipeline",
                         help="Saved feature pipeline to reuse (from train_spark_native.py / train_nonspark_models.py)")
    parser.add_argument("--full-retrain", action="store_true",
                         help="Retrain everything from scratch on old+new data instead of fine-tuning")
    parser.add_argument("--k8s", action="store_true",
                         help="Build the SparkSession against the course's real k8s:// cluster "
                              "(see spark_session.py) instead of the local/spark-submit default. "
                              "Use this when running inside the course's Jupyter driver pod.")
    parser.add_argument("--spark-only", action="store_true",
                         help="Skip retrain_nonspark_models() entirely -- never imports catboost/xgboost/"
                              "lightgbm. Required on the k8s pod, which has no outbound internet to pip "
                              "install those (see CLAUDE.md) and doesn't need them: they're non-distributed "
                              "and driver-only, with solid results already from local runs.")
    args = parser.parse_args()

    if args.full_retrain and not args.old_data:
        parser.error("--full-retrain requires --old-data")
    if not args.old_data:
        logger.warning("No --old-data given: Spark models will train on the new batch alone.")

    tuned = load_tuned_params(args.tuned_params)
    if args.k8s:
        from spark_session import build_k8s_spark_session
        spark = build_k8s_spark_session("hh-ru-skill-prediction-retrain")
    else:
        spark = SparkSession.builder.appName("hh-ru-skill-prediction-retrain").getOrCreate()

    pipeline_model = spark_load_with_retry(
        lambda: PipelineModel.load(spark_save_path(args.pipeline_path, args.k8s)), args.pipeline_path
    )
    logger.info(f"Loaded feature pipeline from {args.pipeline_path}")

    retrain_spark_models(spark, pipeline_model, args.new_data, args.old_data, args.top_n_skills,
                          args.spark_models_dir, tuned, args.k8s)
    if args.spark_only:
        logger.info("--spark-only set: skipping non-Spark (catboost/xgboost/lightgbm) retraining.")
    else:
        retrain_nonspark_models(spark, pipeline_model, args.new_data, args.old_data, args.top_n_skills,
                                 args.nonspark_models_dir, tuned, args.full_retrain)

    spark.stop()
    logger.info("Retraining complete.")


if __name__ == "__main__":
    main()
