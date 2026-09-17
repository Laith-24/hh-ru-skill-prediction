"""
Single-process incremental-retraining demo: fits baseline models on
demo_old, evaluates on demo_eval (before), fine-tunes on demo_new, then
evaluates on demo_eval again (after) -- all within ONE Spark session and
ONE Python process, keeping every model as an in-memory object rather than
round-tripping through disk via PipelineModel.load()/model.load().

Needed because the original multi-process demo (separate spark-submit
invocations per step) crashes locally on Windows the moment a script
calls PipelineModel.load() -- isolated to PySpark's own model-
deserialization path on this Windows/Python 3.12 setup, not Spark
training in general. This sidesteps it entirely: nothing is ever loaded
from disk mid-run.

Non-Spark fine-tuning (catboost/xgboost/lightgbm) supports this directly
-- their init_model=/xgb_model= arguments accept the fitted model/Booster
object itself, not just a file path. Spark models have no incremental
fit in Spark ML anyway, so they're just refit on old+new combined.

Usage:
    spark-submit --master local[2] --driver-memory 4g \\
        --conf spark.sql.parquet.columnarReaderBatchSize=512 \\
        --conf spark.sql.shuffle.partitions=8 --conf spark.default.parallelism=8 \\
        --conf spark.local.dir=<repo>/spark-tmp \\
        src/demo_finetune_single_process.py
"""
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from data_pipeline import load_and_engineer, add_class_weight_column
from models_config import SPARK_MODEL_FACTORIES, NONSPARK_MODEL_FACTORIES

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


def evaluate(spark_models, nonspark_models, df_features, pdf, X, skills):
    rows = []
    for skill in skills:
        label_col = f"label_{skill}"
        for model_name, model in spark_models.get(skill, {}).items():
            predictions = model.transform(df_features)
            pr_auc = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderPR").evaluate(predictions)
            roc_auc = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderROC").evaluate(predictions)
            rows.append({"skill": skill, "model": model_name, "pr_auc": pr_auc, "roc_auc": roc_auc})
        y = pdf[label_col]
        for model_name, model in nonspark_models.get(skill, {}).items():
            probs = model.predict_proba(X)[:, 1]
            rows.append({"skill": skill, "model": model_name, "pr_auc": average_precision_score(y, probs),
                         "roc_auc": roc_auc_score(y, probs)})
    return rows


def main():
    tuned = load_tuned_params()
    spark = SparkSession.builder.appName("hh-ru-demo-finetune-single-process").getOrCreate()

    logger.info("Step A: fitting feature pipeline + baseline models on demo_old ...")
    old_features, top_skills, pipeline_model = load_and_engineer(spark, OLD_DATA, TOP_N_SKILLS)
    old_features.cache()
    label_cols = [f"label_{s}" for s in top_skills]
    old_pdf = old_features.select(["features_array"] + label_cols).toPandas()
    X_old = np.stack(old_pdf["features_array"].values)
    trainable_skills = evaluable_skills(old_pdf, top_skills, label_cols)
    logger.info(f"{len(trainable_skills)}/{len(top_skills)} skills have both classes in demo_old")

    spark_models, nonspark_models = {}, {}
    for skill in trainable_skills:
        label_col = f"label_{skill}"
        weighted_df = add_class_weight_column(old_features, label_col)
        spark_models[skill] = {
            name: factory("scaled_features", label_col, tuned.get(name)).fit(weighted_df)
            for name, factory in SPARK_MODEL_FACTORIES.items()
        }
        sample_weight = compute_sample_weight("balanced", old_pdf[label_col])
        nonspark_models[skill] = {}
        for name, factory in NONSPARK_MODEL_FACTORIES.items():
            model = factory(tuned.get(name))
            model.fit(X_old, old_pdf[label_col], sample_weight=sample_weight)
            nonspark_models[skill][name] = model
    logger.info("Baseline models trained (in memory).")

    logger.info("Step B: evaluating BEFORE fine-tuning on demo_eval ...")
    eval_features, _, _ = load_and_engineer(spark, EVAL_DATA, TOP_N_SKILLS, pipeline_model=pipeline_model)
    eval_features.cache()
    eval_pdf = eval_features.select(["features_array"] + label_cols).toPandas()
    X_eval = np.stack(eval_pdf["features_array"].values)
    eval_skills = [s for s in trainable_skills if s in evaluable_skills(eval_pdf, top_skills, label_cols)]

    before_rows = evaluate(spark_models, nonspark_models, eval_features, eval_pdf, X_eval, eval_skills)
    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(before_rows).to_csv("results/demo_eval_before.csv", index=False)
    logger.info(f"Wrote results/demo_eval_before.csv ({len(before_rows)} rows)")

    logger.info("Step C: fine-tuning on demo_new ...")
    new_features, _, _ = load_and_engineer(spark, NEW_DATA, TOP_N_SKILLS, pipeline_model=pipeline_model)
    new_features.cache()
    new_pdf = new_features.select(["features_array"] + label_cols).toPandas()
    X_new = np.stack(new_pdf["features_array"].values)
    combined_features = old_features.unionByName(new_features, allowMissingColumns=True)
    new_trainable_skills = evaluable_skills(new_pdf, top_skills, label_cols)

    for skill in trainable_skills:
        label_col = f"label_{skill}"
        # Spark: no incremental fit in Spark ML -- refit from scratch on
        # old+new combined (same approach as retrain_on_new_data.py).
        weighted_combined = add_class_weight_column(combined_features, label_col)
        spark_models[skill] = {
            name: factory("scaled_features", label_col, tuned.get(name)).fit(weighted_combined)
            for name, factory in SPARK_MODEL_FACTORIES.items()
        }

        if skill not in new_trainable_skills:
            continue  # new batch alone can't fine-tune this skill's non-spark models

        # Non-spark: TRUE incremental fine-tune on new rows only, continuing
        # from the in-memory baseline model/booster object directly.
        y_new = new_pdf[label_col]
        sample_weight = compute_sample_weight("balanced", y_new)
        base = nonspark_models[skill]
        updated = {}
        cb = NONSPARK_MODEL_FACTORIES["catboost"](tuned.get("catboost"))
        cb.fit(X_new, y_new, sample_weight=sample_weight, init_model=base["catboost"])
        updated["catboost"] = cb
        xgb = NONSPARK_MODEL_FACTORIES["xgboost"](tuned.get("xgboost"))
        xgb.fit(X_new, y_new, sample_weight=sample_weight, xgb_model=base["xgboost"].get_booster())
        updated["xgboost"] = xgb
        lgbm = NONSPARK_MODEL_FACTORIES["lightgbm"](tuned.get("lightgbm"))
        lgbm.fit(X_new, y_new, sample_weight=sample_weight, init_model=base["lightgbm"].booster_)
        updated["lightgbm"] = lgbm
        nonspark_models[skill] = updated
    logger.info("Fine-tuning complete.")

    logger.info("Step D: evaluating AFTER fine-tuning on the SAME demo_eval ...")
    after_rows = evaluate(spark_models, nonspark_models, eval_features, eval_pdf, X_eval, eval_skills)
    pd.DataFrame(after_rows).to_csv("results/demo_eval_after.csv", index=False)
    logger.info(f"Wrote results/demo_eval_after.csv ({len(after_rows)} rows)")

    spark.stop()
    logger.info("Demo complete. Compare results/demo_eval_before.csv vs results/demo_eval_after.csv")


if __name__ == "__main__":
    main()
