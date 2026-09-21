"""
Fully Spark-native training path: every model in SPARK_MODEL_FACTORIES
(RandomForest, GBT, LogisticRegression), trained and evaluated entirely
inside Spark, per skill. Saves each fitted model for later fine-tuning.

Thresholds picked per skill/model on held-out "val" (maximize F1), on
top of class weighting -- weighting alone left these models heavily
recall-skewed since nothing corrected for where Spark's own 0.5 cutoff
falls on the shifted probability distribution. Trains on "train" only,
not "train"+"val" (same trade-off train_nonspark_models.py makes).

Usage:
    spark-submit src/train_spark_native.py --data data/vacancies.parquet
"""

import argparse
import csv
import json
import logging
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.functions import vector_to_array
from sklearn.metrics import f1_score, precision_score, recall_score

from data_pipeline import load_and_engineer, safe_skill_name, add_class_weight_column, best_f1_threshold, spark_save_path
from models_config import SPARK_MODEL_FACTORIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_tuned_params(path: str) -> dict:
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else {}


RESULT_FIELDS = ["skill", "model", "pr_auc", "roc_auc", "f1", "precision", "recall", "threshold"]


class ResultWriter:
    """Appends one row at a time rather than collecting then writing once --
    survives a mid-run crash without losing already-completed results.
    Safe to reuse across --skill-start/--skill-count chunks of the same
    --output file."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_header = not self.path.exists()

    def write(self, row: dict):
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
            if self._write_header:
                writer.writeheader()
                self._write_header = False
            writer.writerow(row)


def _probs_pdf(model, df, label_col):
    """Bring (label, P(class=1)) to pandas for a val/test slice -- small
    (a few thousand rows), just the two columns needed for threshold
    selection / sklearn metrics."""
    return model.transform(df).select(
        F.col(label_col).alias("y"),
        vector_to_array("probability")[1].alias("prob"),
    ).toPandas()


def evaluate_and_save(train_df, val_df, test_df, label_col, factory, params, model_dir):
    classifier = factory("scaled_features", label_col, params)
    model = classifier.fit(train_df)

    predictions = model.transform(test_df)
    pr_auc = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderPR").evaluate(predictions)
    roc_auc = BinaryClassificationEvaluator(labelCol=label_col, metricName="areaUnderROC").evaluate(predictions)

    val_pdf = _probs_pdf(model, val_df, label_col)
    threshold = best_f1_threshold(val_pdf["y"].to_numpy(), val_pdf["prob"].to_numpy())

    test_pdf = _probs_pdf(model, test_df, label_col)
    preds = (test_pdf["prob"] >= threshold).astype(int)
    f1 = f1_score(test_pdf["y"], preds, zero_division=0)
    precision = precision_score(test_pdf["y"], preds, zero_division=0)
    recall = recall_score(test_pdf["y"], preds, zero_division=0)

    model.write().overwrite().save(str(model_dir))
    return pr_auc, roc_auc, f1, precision, recall, threshold


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to the raw hh.ru vacancy parquet file")
    parser.add_argument("--top-n-skills", type=int, default=20)
    parser.add_argument("--output", default="results/spark_results.csv")
    parser.add_argument("--models-dir", default="models/spark")
    parser.add_argument("--tuned-params", default="configs/tuned_params.json")
    parser.add_argument("--pipeline-path", default="models/feature_pipeline",
                         help="Where to save the fitted feature pipeline, for retrain_on_new_data.py to reuse")
    parser.add_argument("--k8s", action="store_true",
                         help="Build the SparkSession against the course's real k8s:// cluster "
                              "(see spark_session.py) instead of local[*]/spark-submit's default master. "
                              "Use this when running inside the course's Jupyter driver pod.")
    parser.add_argument("--skill-start", type=int, default=0,
                         help="Index into the top-N skill list to start at, for chunking a full run across "
                              "several fresh processes -- on the k8s pod, driver memory grows with the "
                              "number of stages ever run in one long-lived SparkSession (Spark's own "
                              "shuffle/broadcast bookkeeping, not released until the session stops) and "
                              "eventually hits the driver pod's fixed 8Gi container limit. "
                              "Index-based (not skill-name-based) since several skill names are "
                              "Cyrillic-with-spaces, unsafe to pass through nested ssh/kubectl exec shells. "
                              "Combine with --pipeline-path pointing at an already-saved pipeline (from a "
                              "prior chunk) to skip refitting it.")
    parser.add_argument("--skill-count", type=int, default=None,
                         help="How many skills to process starting at --skill-start (default: all remaining).")
    args = parser.parse_args()

    if args.k8s:
        from spark_session import build_k8s_spark_session
        spark = build_k8s_spark_session("hh-ru-skill-prediction-spark")
    else:
        spark = SparkSession.builder.appName("hh-ru-skill-prediction-spark").getOrCreate()

    saved_pipeline_path = spark_save_path(args.pipeline_path, args.k8s)
    reuse_pipeline = None
    if Path(args.pipeline_path).exists():
        try:
            reuse_pipeline = PipelineModel.load(saved_pipeline_path)
            logger.info(f"Reusing already-fitted pipeline from {args.pipeline_path} (skipping refit)")
        except Exception as e:
            # Seen in practice on the k8s pod's NFS mount: a pipeline saved
            # by one process (one chunk of a --skill-start/--skill-count
            # run) sometimes isn't readable back by a later process --
            # "ValueError: RDD is empty" loading the metadata, even though
            # the save reported success. Likely an NFS write-flush
            # consistency issue between separate processes, not something
            # worth chasing further -- refit fresh instead of failing the
            # whole run over it (costs ~4 min, cheap insurance).
            logger.warning(f"Failed to load saved pipeline from {args.pipeline_path} ({e!r}); refitting fresh")

    df_features, top_skills, pipeline_model = load_and_engineer(
        spark, args.data, args.top_n_skills, pipeline_model=reuse_pipeline
    )
    df_features.cache()
    if reuse_pipeline is None:
        pipeline_model.write().overwrite().save(saved_pipeline_path)
        logger.info(f"Feature pipeline saved to {args.pipeline_path}")

    train_df = df_features.filter("split = 'train'")
    val_df = df_features.filter("split = 'val'")
    test_df = df_features.filter("split = 'test'")

    tuned = load_tuned_params(args.tuned_params)
    result_writer = ResultWriter(args.output)

    end = None if args.skill_count is None else args.skill_start + args.skill_count
    skills_to_run = top_skills[args.skill_start:end]
    logger.info(f"Running skills [{args.skill_start}:{end}] -- {len(skills_to_run)}/{len(top_skills)} total")

    for skill in skills_to_run:
        label_col = f"label_{skill}"
        class_counts = train_df.groupBy(label_col).count().collect()
        if len(class_counts) < 2:
            logger.warning(f"Skipping '{skill}': only one class present in training split")
            continue

        weighted_train_df = add_class_weight_column(train_df, label_col)

        for model_name, factory in SPARK_MODEL_FACTORIES.items():
            model_dir = spark_save_path(str(Path(args.models_dir) / model_name / safe_skill_name(skill)), args.k8s)
            pr_auc, roc_auc, f1, precision, recall, threshold = evaluate_and_save(
                weighted_train_df, val_df, test_df, label_col, factory, tuned.get(model_name), model_dir
            )
            result_writer.write({
                "skill": skill, "model": model_name, "pr_auc": pr_auc, "roc_auc": roc_auc,
                "f1": f1, "precision": precision, "recall": recall, "threshold": round(threshold, 4),
            })
            logger.info(
                f"{skill} / {model_name}: PR-AUC={pr_auc:.3f}  ROC-AUC={roc_auc:.3f}  "
                f"F1={f1:.3f}  P={precision:.3f}  R={recall:.3f}  thr={threshold:.3f}"
            )

    logger.info(f"Results written to {args.output}")

    spark.stop()


if __name__ == "__main__":
    main()
