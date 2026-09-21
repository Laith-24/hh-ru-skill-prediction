"""
Streamlit demo: predicts which of the 50 most frequent skills a job posting
needs, from its title alone. The models were trained on the title's own
features (its TF-IDF and Word2Vec vectors and the seniority flags read from
it), so nothing they expect is missing from what the demo can supply.

Uses the saved CatBoost models for inference; Spark is only used once at
startup, to refit the feature pipeline that turns a title into those vectors.

Caveat shown in the UI itself: this demonstrates the model's mechanism,
not that any single prediction is individually reliable -- validated
accuracy is an aggregate measure across held-out postings.

Usage: streamlit run interface/app.py
"""
import sys
import tempfile
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import streamlit as st
from catboost import CatBoostClassifier

SRC_DIR = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from pyspark.sql import SparkSession

from data_pipeline import TITLE_FEATURE_COLS, load_and_engineer, safe_skill_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = str(PROJECT_ROOT / "data" / "vacancies.parquet")
N_SKILLS = 50
SKILL_LIST_PATH = str(PROJECT_ROOT / "configs" / "top_skills_50.json")
CATBOOST_MODELS_DIR = PROJECT_ROOT / "models" / "title_only_top50" / "catboost"
RESULTS_PATH = PROJECT_ROOT / "results" / "title_only_results_top50.csv"
SPARK_TMP_DIR = str(PROJECT_ROOT / "spark-tmp")

# The pipeline's other stages still need these columns, but the title-only models never use them.
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

    _, top_skills, pipeline_model = load_and_engineer(spark, DATA_PATH, N_SKILLS, skill_list_path=SKILL_LIST_PATH)

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

        feature_pdf = transformed.select(*TITLE_FEATURE_COLS).toPandas()

    feature_vec = np.concatenate([feature_pdf[c].iloc[0].toArray() for c in TITLE_FEATURE_COLS]).reshape(1, -1)

    results = []
    for skill, (clf, threshold) in models.items():
        score = clf.predict_proba(feature_vec)[0, 1]
        results.append({"skill": skill, "score": score, "threshold": threshold, "predicted": score >= threshold})
    return pd.DataFrame(results).sort_values("score", ascending=False)


CHART_TOP_N = 10
CHART_PURPLE, CHART_GREY, CHART_DARK = "#8A2BE2", "#B2B2B2", "#2C2D2E"
STATUS_PREDICTED, STATUS_BELOW = "predicted (reaches its threshold)", "below its threshold"


def top_skills_chart(result_df, n=CHART_TOP_N):
    """Horizontal bars for the n highest scores; the black tick on each row is that skill's own decision threshold."""
    top = result_df.head(n).copy()
    top["status"] = np.where(top["predicted"], STATUS_PREDICTED, STATUS_BELOW)
    top["label_x"] = np.maximum(top["score"], top["threshold"])  # the number goes right of the bar and of the tick, so they never overlap
    x_scale = alt.Scale(domain=[0, 1.1], nice=False)
    base = alt.Chart(top).encode(y=alt.Y("skill:N", sort=top["skill"].tolist(), title=None, axis=alt.Axis(labelLimit=300)))
    tooltip = [alt.Tooltip("skill:N"), alt.Tooltip("score:Q", format=".3f"), alt.Tooltip("threshold:Q", format=".3f")]
    bars = base.mark_bar(size=22).encode(
        x=alt.X("score:Q", scale=x_scale, axis=alt.Axis(values=[0, 0.25, 0.5, 0.75, 1.0], format=".2f", title="Score")),
        color=alt.Color("status:N", scale=alt.Scale(domain=[STATUS_PREDICTED, STATUS_BELOW], range=[CHART_PURPLE, CHART_GREY]),
                        legend=alt.Legend(title=None, orient="bottom")),
        tooltip=tooltip,
    )
    ticks = base.mark_tick(color=CHART_DARK, thickness=3, size=30).encode(x=alt.X("threshold:Q", scale=x_scale, axis=None), tooltip=tooltip)
    labels = base.mark_text(align="left", dx=8, color=CHART_DARK).encode(
        x=alt.X("label_x:Q", scale=x_scale, axis=None), text=alt.Text("score:Q", format=".2f"))
    return (bars + ticks + labels).properties(height=34 * len(top))


st.set_page_config(page_title="hh.ru Skill Predictor", page_icon="\U0001F50D")
st.title("Job Posting Skill Predictor")
st.caption(
    "Predicts required skills from the job title alone. CatBoost models for the 50 most common skills, trained on about 32k "
    "real hh.ru postings using only features of the title. On held-out titles they score macro PR-AUC 0.24 and F1 0.29. "
    "Scores come from class-weighted models, so they are not calibrated probabilities; each skill has its own decision threshold."
)
st.info(
    "This demonstrates the model's mechanism, not a claim that any single prediction is individually "
    "reliable -- the accuracy above is an aggregate over thousands of held-out postings, which does not "
    "imply any one prediction is robust to small input changes.",
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
                st.write(f"**{r['skill']}** -- score {r['score']:.2f} (threshold {r['threshold']:.2f})")

        st.subheader(f"Top {min(CHART_TOP_N, len(result_df))} skills by score")
        st.altair_chart(top_skills_chart(result_df), use_container_width=True)
        st.caption("Bar: the model's score for this title. Black tick: that skill's own decision threshold; purple bars reach it and are predicted.")

        with st.expander(f"All {len(result_df)} skills, ranked by score"):
            st.dataframe(result_df.round({"score": 3, "threshold": 3}), use_container_width=True, hide_index=True)
