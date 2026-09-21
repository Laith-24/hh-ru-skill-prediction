"""Extracts test-set predictions grouped by job title, to check whether
aggregating per-posting predictions across postings sharing a title
tracks the real aggregate skill frequency for that title -- the
empirical test behind the "run it across many postings as a market
signal" pitch, never verified elsewhere in this project (only per-
posting metrics were).

Caches the expensive part (Spark + pipeline fit + prediction) to
results/_market_signal_raw.parquet; see market_signal_analyze.py for
the actual (cheap, iterable) correlation analysis.

Usage: python src/market_signal_extract.py
"""
import logging
import sys
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_pipeline import load_and_engineer, safe_skill_name

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = str(ROOT / "data" / "vacancies.parquet")
CATBOOST_MODELS_DIR = ROOT / "models" / "nonspark" / "catboost"
OUTPUT_PATH = ROOT / "results" / "_market_signal_raw.parquet"
SPARK_TMP_DIR = str(ROOT / "spark-tmp")


def main():
    spark = (
        SparkSession.builder.appName("hh-ru-market-signal-extract")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.parquet.columnarReaderBatchSize", "512")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .config("spark.local.dir", SPARK_TMP_DIR)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    df_features, top_skills, _ = load_and_engineer(spark, DATA_PATH, 20)

    label_cols = [f"label_{skill}" for skill in top_skills]
    pdf = df_features.select(["split", "name", "features_array"] + label_cols).toPandas()
    spark.stop()

    test = pdf[pdf["split"] == "test"].reset_index(drop=True)
    logger.info(f"Test rows: {len(test)}")

    X_test = np.stack(test["features_array"].values)

    available_skills = []
    for skill in top_skills:
        model_path = CATBOOST_MODELS_DIR / f"{safe_skill_name(skill)}.cbm"
        if not model_path.exists():
            logger.warning(f"No saved catboost model for {skill!r}, skipping")
            continue
        clf = CatBoostClassifier()
        clf.load_model(str(model_path))
        test[f"pred_{skill}"] = clf.predict_proba(X_test)[:, 1]
        available_skills.append(skill)

    out_cols = ["name"] + [f"label_{s}" for s in available_skills] + [f"pred_{s}" for s in available_skills]
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    test[out_cols].to_parquet(OUTPUT_PATH, index=False)
    logger.info(f"Saved {len(test)} test rows x {len(available_skills)} skills to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
