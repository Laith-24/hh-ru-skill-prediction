"""One-off sanity check for the interface's prediction logic -- prints the
full ranked probability list for a given title instead of only what fits
in the UI. Duplicates predict() inline rather than importing app.py, since
app.py has top-level Streamlit calls that assume a `streamlit run` context.
Not part of the app itself.

Usage: spark-submit --master local[2] --driver-memory 4g \
    --conf spark.sql.parquet.columnarReaderBatchSize=512 \
    --conf spark.sql.shuffle.partitions=8 --conf spark.default.parallelism=8 \
    --conf spark.local.dir=<repo>/spark-tmp interface/test_predict.py "Senior Python Developer"
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from catboost import CatBoostClassifier
from pyspark.sql import SparkSession

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data_pipeline import load_and_engineer, safe_skill_name, extract_basic_features, _add_experience_and_work_format

DATA_PATH = str(ROOT / "data" / "vacancies.parquet")
CATBOOST_MODELS_DIR = ROOT / "models" / "nonspark" / "catboost"
RESULTS_PATH = ROOT / "results" / "nonspark_results_final5_word2vec.csv"
SPARK_TMP_DIR = str(ROOT / "spark-tmp")

# Must match interface/app.py's fixed defaults exactly.
EMPLOYER_PLACEHOLDER = "Не указано"
DEFAULT_CITY = "Москва"
DEFAULT_EXPERIENCE_ID = "between1And3"
DEFAULT_WORK_FORMAT_IDS = ["ON_SITE"]

ARROW_ROW_SCHEMA = pa.schema([
    pa.field("id", pa.string()), pa.field("name", pa.string()), pa.field("description", pa.string()),
    pa.field("key_skills", pa.list_(pa.struct([pa.field("name", pa.string())]))),
    pa.field("area", pa.struct([pa.field("name", pa.string())])),
    pa.field("employer", pa.struct([pa.field("name", pa.string())])),
    pa.field("salary", pa.struct([pa.field("from", pa.float64()), pa.field("to", pa.float64())])),
    pa.field("experience", pa.struct([pa.field("id", pa.string())])),
    pa.field("work_format", pa.list_(pa.struct([pa.field("id", pa.string())]))),
])


def predict(spark, pipeline_model, models, title):
    row = {
        "id": "interface-input", "name": title, "description": "",
        "key_skills": [{"name": ""}],
        "area": {"name": DEFAULT_CITY},
        "employer": {"name": EMPLOYER_PLACEHOLDER},
        "salary": {"from": None, "to": None},
        "experience": {"id": DEFAULT_EXPERIENCE_ID},
        "work_format": [{"id": wf} for wf in DEFAULT_WORK_FORMAT_IDS],
    }
    table = pa.Table.from_pylist([row], schema=ARROW_ROW_SCHEMA)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = str(Path(tmp_dir) / "input_row.parquet")
        pq.write_table(table, tmp_path)
        single_row_df = spark.read.parquet(tmp_path)
        featured = extract_basic_features(single_row_df)
        featured = _add_experience_and_work_format(featured)
        transformed = pipeline_model.transform(featured)
        feature_pdf = transformed.select("scaled_features").toPandas()
    feature_vec = np.array(feature_pdf["scaled_features"].iloc[0]).reshape(1, -1)
    results = []
    for skill, (clf, threshold) in models.items():
        prob = clf.predict_proba(feature_vec)[0, 1]
        results.append({"skill": skill, "probability": prob, "threshold": threshold, "predicted": prob >= threshold})
    return pd.DataFrame(results).sort_values("probability", ascending=False)


title = sys.argv[1] if len(sys.argv) > 1 else "Senior Python Developer"

spark = (
    SparkSession.builder.appName("hh-ru-skill-interface-test").master("local[2]")
    .config("spark.driver.memory", "4g")
    .config("spark.sql.parquet.columnarReaderBatchSize", "512")
    .config("spark.sql.shuffle.partitions", "8")
    .config("spark.default.parallelism", "8")
    .config("spark.local.dir", SPARK_TMP_DIR).getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

_, top_skills, pipeline_model = load_and_engineer(spark, DATA_PATH, 20)

thresholds = pd.read_csv(RESULTS_PATH)
thresholds = thresholds[thresholds["model"] == "catboost"].set_index("skill")["threshold"].to_dict()

models = {}
for skill in top_skills:
    model_path = CATBOOST_MODELS_DIR / f"{safe_skill_name(skill)}.cbm"
    if not model_path.exists():
        continue
    clf = CatBoostClassifier()
    clf.load_model(str(model_path))
    models[skill] = (clf, thresholds.get(skill, 0.5))

result = predict(spark, pipeline_model, models, title)
print(f"Title: {title!r}")
print(result.to_string(index=False))
spark.stop()
