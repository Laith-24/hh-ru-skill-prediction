"""
Streamlit demo: predicts which of the top-20 skills a job posting needs,
from its title alone. Employer, city, experience, and work format are
fixed to defaults rather than exposed as inputs -- testing showed
description text and employer identity add more noise than signal for
this pipeline's small, from-scratch TF-IDF/Word2Vec vocabulary.

Uses the saved CatBoost models for inference; Spark is only used once at
startup, to refit the same feature pipeline the models were trained on.

Caveat shown in the UI itself: this demonstrates the model's mechanism,
not that any single prediction is individually reliable -- validated
accuracy (PR-AUC ~0.3) is an aggregate measure across held-out postings.

Usage: streamlit run interface/app.py
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import streamlit as st
from catboost import CatBoostClassifier

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from pyspark.sql import SparkSession

from data_pipeline import load_and_engineer, safe_skill_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(PROJECT_ROOT / "data" / "vacancies.parquet")
CATBOOST_MODELS_DIR = PROJECT_ROOT / "models" / "nonspark" / "catboost"
RESULTS_PATH = PROJECT_ROOT / "results" / "nonspark_results_final5_word2vec.csv"
SPARK_TMP_DIR = str(PROJECT_ROOT / "spark-tmp")

# Fixed defaults for everything except title (measured to work best -- see CLAUDE.md).
EMPLOYER_PLACEHOLDER = "Не указано"  # falls into the generic "Other" bucket
DEFAULT_CITY = "Москва"
DEFAULT_EXPERIENCE_ID = "between1And3"
DEFAULT_WORK_FORMAT_IDS = ["ON_SITE"]


@st.cache_resource(show_spinner="Starting Spark and fitting the feature pipeline (one-time, a few minutes)...")
def load_pipeline_and_models():
    spark = (
        SparkSession.builder
        .appName("hh-ru-skill-interface")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.parquet.columnarReaderBatchSize", "512")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .config("spark.local.dir", SPARK_TMP_DIR)
        .getOrCreate()
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

    return spark, pipeline_model, top_skills, models


ARROW_ROW_SCHEMA = pa.schema([
    pa.field("id", pa.string()),
    pa.field("name", pa.string()),
    pa.field("description", pa.string()),
    pa.field("key_skills", pa.list_(pa.struct([pa.field("name", pa.string())]))),
    pa.field("area", pa.struct([pa.field("name", pa.string())])),
    pa.field("employer", pa.struct([pa.field("name", pa.string())])),
    pa.field("salary", pa.struct([pa.field("from", pa.float64()), pa.field("to", pa.float64())])),
    pa.field("experience", pa.struct([pa.field("id", pa.string())])),
    pa.field("work_format", pa.list_(pa.struct([pa.field("id", pa.string())]))),
])


def predict(spark, pipeline_model, models, title):
    row = {
        "id": "interface-input",
        "name": title,
        "description": "",
        # [{"name": ""}], not []: extract_basic_features() (frozen) misdetects
        # an empty list and picks the wrong branch. Value itself is unused.
        "key_skills": [{"name": ""}],
        "area": {"name": DEFAULT_CITY},
        "employer": {"name": EMPLOYER_PLACEHOLDER},
        "salary": {"from": None, "to": None},
        "experience": {"id": DEFAULT_EXPERIENCE_ID},
        "work_format": [{"id": wf} for wf in DEFAULT_WORK_FORMAT_IDS],
    }
    # Write+read via parquet rather than spark.createDataFrame(list, schema)
    # directly -- the latter crashes the Python worker on this setup.
    table = pa.Table.from_pylist([row], schema=ARROW_ROW_SCHEMA)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = str(Path(tmp_dir) / "input_row.parquet")
        pq.write_table(table, tmp_path)
        single_row_df = spark.read.parquet(tmp_path)

        from data_pipeline import extract_basic_features, _add_experience_and_work_format
        featured = extract_basic_features(single_row_df)
        featured = _add_experience_and_work_format(featured)
        transformed = pipeline_model.transform(featured)

        feature_pdf = transformed.select("scaled_features").toPandas()

    feature_vec = np.array(feature_pdf["scaled_features"].iloc[0]).reshape(1, -1)

    results = []
    for skill, (clf, threshold) in models.items():
        prob = clf.predict_proba(feature_vec)[0, 1]
        results.append({"skill": skill, "probability": prob, "predicted": prob >= threshold})
    return pd.DataFrame(results).sort_values("probability", ascending=False)


st.set_page_config(page_title="hh.ru Skill Predictor", page_icon="\U0001F50D")
st.title("Job Posting Skill Predictor")
st.caption(
    "Predicts required skills from the job title alone. CatBoost models, trained on 35k real hh.ru postings. "
    "Description, employer, city, experience, and work format are deliberately not inputs here -- "
    "testing showed they add noise, not signal, for this pipeline."
)
st.info(
    "This demonstrates the model's mechanism, not a claim that any single prediction is individually "
    "reliable -- the model's real, validated accuracy is measured in aggregate across thousands of held-out "
    "postings (PR-AUC ~0.3, see CLAUDE.md), which does not imply any one prediction is robust to small "
    "input changes.",
    icon="ℹ️",
)

with st.spinner("Loading models (first run only takes a few minutes) ..."):
    spark, pipeline_model, top_skills, models = load_pipeline_and_models()

with st.form("posting_form"):
    title = st.text_input("Job title", placeholder="Senior Python Developer")
    submitted = st.form_submit_button("Predict skills")

if submitted:
    if not title.strip():
        st.error("Job title is required.")
    else:
        with st.spinner("Predicting..."):
            result_df = predict(spark, pipeline_model, models, title)

        predicted = result_df[result_df["predicted"]]
        st.subheader(f"Predicted skills ({len(predicted)})")
        if predicted.empty:
            st.info("No skills crossed their decision threshold for this title.")
        else:
            for _, r in predicted.iterrows():
                st.write(f"**{r['skill']}** -- {r['probability']:.1%}")

        with st.expander("All 20 skills, ranked by probability"):
            st.dataframe(
                result_df.assign(probability=result_df["probability"].map(lambda p: f"{p:.1%}")),
                use_container_width=True, hide_index=True,
            )
