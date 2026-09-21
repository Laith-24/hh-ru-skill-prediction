"""
Central model registry -- the single place that defines every candidate
model, so train / tune / retrain scripts never drift out of sync.

Two families: SPARK_MODEL_FACTORIES (trained + evaluated entirely inside
Spark) and NONSPARK_MODEL_FACTORIES (pandas/numpy after the Spark ->
pandas bridge; all three support continued training from a saved model,
see retrain_on_new_data.py).

Spark factories set weightCol="class_weight" (every DataFrame passed to
.fit() must carry that column) -- without it, Spark's default 0.5
threshold essentially never fires on this dataset's imbalanced labels.
Non-Spark models are weighted per-call via sample_weight= at .fit() time
instead, since their APIs take it as a fit-time argument, not a
constructor default.
"""

from typing import Optional

from pyspark.ml.classification import (
    RandomForestClassifier,
    GBTClassifier,
    LogisticRegression as SparkLogisticRegression,
)
# catboost/xgboost/lightgbm imports deferred into their factory functions
# below so importing this module doesn't require them installed -- they
# aren't on the course's k8s pod, and Spark-native models don't need them.


def _merged(defaults: dict, overrides: Optional[dict]) -> dict:
    return {**defaults, **(overrides or {})}


# --- Spark-native models ---------------------------------------------------
# factory(features_col, label_col, params) -> unfitted Estimator

SPARK_WEIGHT_COL = "class_weight"

SPARK_MODEL_FACTORIES = {
    "spark_random_forest": lambda features_col, label_col, params=None: RandomForestClassifier(
        **_merged(dict(featuresCol=features_col, labelCol=label_col, weightCol=SPARK_WEIGHT_COL,
                        numTrees=100, maxDepth=10, minInstancesPerNode=5, seed=42), params)
    ),
    "spark_gbt": lambda features_col, label_col, params=None: GBTClassifier(
        **_merged(dict(featuresCol=features_col, labelCol=label_col, weightCol=SPARK_WEIGHT_COL,
                        maxIter=100, maxDepth=5, seed=42), params)
    ),
    "spark_logistic_regression": lambda features_col, label_col, params=None: SparkLogisticRegression(
        **_merged(dict(featuresCol=features_col, labelCol=label_col, weightCol=SPARK_WEIGHT_COL,
                        maxIter=100, regParam=0.01), params)
    ),
}

# --- Non-Spark models (pandas/numpy) ---------------------------------------
# factory(params) -> unfitted, sklearn-compatible estimator

def _catboost_factory(params=None):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(**_merged(dict(iterations=300, depth=6, verbose=False, random_seed=42), params))


def _xgboost_factory(params=None):
    from xgboost import XGBClassifier
    return XGBClassifier(**_merged(dict(n_estimators=300, max_depth=6, eval_metric="logloss", random_state=42), params))


def _lightgbm_factory(params=None):
    from lightgbm import LGBMClassifier
    return LGBMClassifier(**_merged(dict(n_estimators=300, max_depth=6, random_state=42, verbose=-1), params))


NONSPARK_MODEL_FACTORIES = {
    "catboost": _catboost_factory,
    "xgboost": _xgboost_factory,
    "lightgbm": _lightgbm_factory,
}

# Native save format per library -- needed to reload for continued training
# (init_model= / xgb_model=) in retrain_on_new_data.py.
NONSPARK_MODEL_EXTENSIONS = {
    "catboost": "cbm",
    "xgboost": "json",
    "lightgbm": "txt",
}
