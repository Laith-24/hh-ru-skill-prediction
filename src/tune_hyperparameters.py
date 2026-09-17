"""
Hyperparameter search for every model in models_config.

Design choice: tunes on a small number of representative skills (default 3,
auto-picked as the most balanced by positive rate) rather than all top-N
skills -- 6 models x 20 skills x cross-validation would mean thousands of
fits, which isn't worth it for a course project. Override with --skills if
you want to tune on specific ones. Results across the tuning skills are
averaged into a single param set per model, applied to every skill by
train_spark_native.py / train_nonspark_models.py (they read
configs/tuned_params.json automatically if it exists).

This is meant to run occasionally, not on every training run -- expect it
to take a while.

Usage:
    python3 src/tune_hyperparameters.py --data data/vacancies.parquet
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils.class_weight import compute_sample_weight

from data_pipeline import load_and_engineer, add_class_weight_column
from models_config import SPARK_MODEL_FACTORIES, NONSPARK_MODEL_FACTORIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# stepSize (GBT) and elasticNetParam (LR) added 2026-08-22: neither was
# tuned before, and stepSize in particular is usually the single most
# impactful GBM hyperparameter. Kept to 2 values each (not 3) to avoid
# tripling those two models' already-dominant share of tuning time.
SPARK_TUNED_PARAM_NAMES = {
    "spark_random_forest": ["numTrees", "maxDepth"],
    "spark_gbt": ["maxIter", "maxDepth", "stepSize"],
    "spark_logistic_regression": ["regParam", "elasticNetParam"],
}

SPARK_PARAM_GRIDS = {
    "spark_random_forest": lambda clf: ParamGridBuilder()
        .addGrid(clf.numTrees, [50, 100, 150]).addGrid(clf.maxDepth, [5, 10, 15]).build(),
    "spark_gbt": lambda clf: ParamGridBuilder()
        .addGrid(clf.maxIter, [50, 100]).addGrid(clf.maxDepth, [3, 5, 7])
        .addGrid(clf.stepSize, [0.05, 0.1]).build(),
    "spark_logistic_regression": lambda clf: ParamGridBuilder()
        .addGrid(clf.regParam, [0.001, 0.01, 0.1]).addGrid(clf.elasticNetParam, [0.0, 0.5]).build(),
}

NONSPARK_PARAM_DISTRIBUTIONS = {
    "catboost": {"depth": [4, 6, 8], "learning_rate": [0.03, 0.1, 0.2], "iterations": [200, 300, 500]},
    "xgboost": {"max_depth": [4, 6, 8], "learning_rate": [0.03, 0.1, 0.2], "n_estimators": [200, 300, 500]},
    "lightgbm": {"max_depth": [4, 6, 8], "learning_rate": [0.03, 0.1, 0.2], "n_estimators": [200, 300, 500]},
}


def pick_representative_skills(train_df, top_skills, n=3):
    """Skills closest to a balanced 50/50 split -- give the clearest tuning
    signal, avoid picking a near-all-0 skill where PR-AUC is meaningless.

    Uses the Column API (F.avg(F.col(...))), not a raw SQL string -- skill
    names can contain spaces/Cyrillic ("аналитическое мышление"), which
    breaks unquoted identifier parsing in selectExpr's SQL text (hit for
    real: "[PARSE_SYNTAX_ERROR] Syntax error at or near 'а'").
    """
    rates = []
    for skill in top_skills:
        rate = train_df.select(F.avg(F.col(f"label_{skill}")).alias("r")).collect()[0]["r"]
        rates.append((skill, abs((rate or 0) - 0.5)))
    rates.sort(key=lambda x: x[1])
    return [skill for skill, _ in rates[:n]]


def tune_spark_models(train_df, label_col):
    weighted_train_df = add_class_weight_column(train_df, label_col)
    best = {}
    for model_name, factory in SPARK_MODEL_FACTORIES.items():
        clf = factory("scaled_features", label_col)
        grid = SPARK_PARAM_GRIDS[model_name](clf)
        evaluator = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderPR")
        cv = CrossValidator(estimator=clf, estimatorParamMaps=grid, evaluator=evaluator, numFolds=3, parallelism=2)
        cv_model = cv.fit(weighted_train_df)
        param_map = cv_model.bestModel.extractParamMap()
        wanted = set(SPARK_TUNED_PARAM_NAMES[model_name])
        best[model_name] = {p.name: v for p, v in param_map.items() if p.name in wanted}
        logger.info(f"Best params for {model_name}: {best[model_name]}")
    return best


def tune_nonspark_models(X_train, pdf_labels_for_skill):
    sample_weight = compute_sample_weight("balanced", pdf_labels_for_skill)
    best = {}
    for model_name, factory in NONSPARK_MODEL_FACTORIES.items():
        base = factory()
        search = RandomizedSearchCV(
            base, NONSPARK_PARAM_DISTRIBUTIONS[model_name],
            n_iter=8, scoring="average_precision", cv=3, random_state=42, n_jobs=-1,
        )
        search.fit(X_train, pdf_labels_for_skill, sample_weight=sample_weight)
        best[model_name] = search.best_params_
        logger.info(f"Best params for {model_name}: {best[model_name]}")
    return best


def _average_params(param_dicts_by_model, model_names):
    """Average numeric values across tuning skills; round back to int where
    the original value was integral (numTrees, maxDepth, etc.)."""
    merged = {}
    for name in model_names:
        keys = param_dicts_by_model[0][name].keys()
        merged[name] = {}
        for k in keys:
            values = [d[name][k] for d in param_dicts_by_model]
            if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
                avg = sum(values) / len(values)
                merged[name][k] = int(round(avg)) if all(float(v).is_integer() for v in values) else avg
            else:
                merged[name][k] = values[0]
    return merged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--top-n-skills", type=int, default=20)
    parser.add_argument("--skills", nargs="+", default=None,
                         help="Skills to tune on; defaults to 3 auto-picked balanced skills")
    parser.add_argument("--output", default="configs/tuned_params.json")
    args = parser.parse_args()

    spark = SparkSession.builder.appName("hh-ru-skill-prediction-tuning").getOrCreate()
    df_features, top_skills, _ = load_and_engineer(spark, args.data, args.top_n_skills)
    df_features.cache()
    # CrossValidator/RandomizedSearchCV do their own internal k-fold split for
    # tuning, so they don't need the reserved "val" slice either -- use train+val.
    train_df = df_features.filter("split IN ('train', 'val')")

    tuning_skills = args.skills or pick_representative_skills(train_df, top_skills, n=3)
    logger.info(f"Tuning on skills: {tuning_skills}")

    label_cols = [f"label_{s}" for s in tuning_skills]
    pdf = train_df.select(["features_array"] + label_cols).toPandas()
    X_train_pd = np.stack(pdf["features_array"].values)

    spark_results, nonspark_results = [], []
    for skill in tuning_skills:
        label_col = f"label_{skill}"
        spark_results.append(tune_spark_models(train_df, label_col))
        nonspark_results.append(tune_nonspark_models(X_train_pd, pdf[label_col]))

    spark.stop()

    final = {
        **_average_params(spark_results, SPARK_MODEL_FACTORIES.keys()),
        **_average_params(nonspark_results, NONSPARK_MODEL_FACTORIES.keys()),
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(final, indent=2))
    logger.info(f"Tuned params written to {out_path}")


if __name__ == "__main__":
    main()
