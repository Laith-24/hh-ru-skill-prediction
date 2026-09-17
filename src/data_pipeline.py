"""
Shared data loading + feature engineering pipeline. Wraps
feature_engineering.py (unmodified) and adds:
  - a deterministic train/val/test split keyed on a hash of `id` (not
    rand(seed)), reproducible regardless of partitioning/execution order.
  - a cached, reusable skill list (configs/top_skills.json) and top-
    employer list (configs/top_employers.json), computed once and reused
    by every script so label columns/employer buckets stay consistent
    across different data files.
  - deduplication of near-identical reposted postings before splitting,
    to prevent train/test leakage.
"""

import json
import logging
import re
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_recall_curve

from pyspark.sql import functions as F
from pyspark.ml.functions import vector_to_array
from pyspark.ml.feature import (
    StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler, SQLTransformer,
    RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, Word2Vec,
)
from pyspark.ml import Pipeline

from feature_engineering import (
    extract_basic_features,
    get_top_skills,
    create_multi_label_columns,
)

logger = logging.getLogger(__name__)

SKILL_LIST_PATH = "configs/top_skills.json"
EMPLOYER_LIST_PATH = "configs/top_employers.json"

WORK_FORMAT_IDS = ["ON_SITE", "REMOTE", "HYBRID", "FIELD_WORK"]
TITLE_TFIDF_VOCAB_SIZE = 300
TITLE_TFIDF_MIN_DF = 5
DESCRIPTION_TFIDF_VOCAB_SIZE = 800
DESCRIPTION_TFIDF_MIN_DF = 10
WORD2VEC_VECTOR_SIZE = 100
WORD2VEC_MIN_COUNT = 5


def _skill_name_tokens(top_skills):
    """Tokenize skill names the same way RegexTokenizer splits description
    text (\\W+, lowercased, min length 2), so they can be excluded from the
    description TF-IDF vocabulary -- see build_ml_pipeline_no_leakage's
    docstring on why (67% average verbatim skill-name overlap measured
    between key_skills and description text)."""
    tokens = set()
    for skill in top_skills:
        for tok in re.split(r"\W+", skill.lower()):
            if len(tok) >= 2:
                tokens.add(tok)
    return list(tokens)

_UNSAFE_PATH_CHARS = re.compile(r'[<>:"/\\|?*]')


def safe_skill_name(skill: str) -> str:
    """Sanitize a skill name for use as a filesystem path component.

    Skill names come straight from hh.ru's free-text key_skills field, and
    some contain characters unsafe in a path: "1с: предприятие 8" has a
    colon, which Spark's model.write().save() parses as a URI scheme
    separator on Windows and crashes on
    (`Relative path in absolute URI`, hit after 16/20 skills trained
    successfully); "c/c++" has a slash, which would silently create a
    stray nested directory instead of one path component. Used everywhere
    a skill name becomes a directory/file name (train_spark_native.py,
    train_nonspark_models.py, retrain_on_new_data.py).
    """
    return _UNSAFE_PATH_CHARS.sub("_", skill)


def spark_save_path(path: str, k8s: bool) -> str:
    """Under --k8s, Spark's default filesystem is HDFS (fs.defaultFS) --
    any bare path passed to PipelineModel/model .load()/.save() resolves
    against HDFS regardless of intent, which fails while HDFS is down or
    simply isn't where the path actually lives (see CLAUDE.md). Prefixing
    with file:// forces the local/NFS filesystem instead, for both
    directions (load and save). Shared by every script that reads or
    writes a Spark ML model/pipeline under --k8s (train_spark_native.py,
    retrain_on_new_data.py, demo_evaluate.py) so they resolve paths the
    same way instead of each guessing independently.
    """
    if k8s:
        return f"file://{Path(path).resolve()}"
    return path


def spark_load_with_retry(load_fn, what, attempts=5, delay=10):
    """A save from a just-finished, separate kubectl exec process
    sometimes isn't readable back yet on this cluster's NFS mount --
    ValueError: RDD is empty loading metadata, a write-flush consistency
    race, not real corruption (confirmed during the original full
    Spark-native sweep, see CLAUDE.md). train_spark_native.py's own
    pipeline-reuse path handles this by falling back to refitting fresh,
    which is safe there since it's about to use the pipeline on the SAME
    data it would refit on. That fallback is NOT safe for eval/fine-tune
    callers (retrain_on_new_data.py, demo_evaluate.py): they need the
    EXACT saved pipeline/model, since a refit produces a different,
    incompatible feature space. So this only retries the same load, on a
    short delay, rather than substituting something else.
    """
    import time
    for attempt in range(1, attempts + 1):
        try:
            return load_fn()
        except Exception as e:
            if attempt == attempts:
                raise
            logger.warning(f"Load failed for {what} (attempt {attempt}/{attempts}): {e!r} -- retrying in {delay}s")
            time.sleep(delay)


def _dedupe_postings(df):
    """
    Drop exact-duplicate postings (same title + employer + description)
    before splitting.

    hh.ru reposts near-identical listings constantly (one crowd-task
    posting, "Главный аналитик социальных сетей" @ "Яндекс Крауд", appears
    218 times under 218 different `id`s). Since the train/val/test split is
    keyed on a hash of `id`, these duplicates land on both sides of the
    split roughly at random -- so part of "test" ends up being rows the
    model already memorized in training under a different id, inflating
    every reported metric. `id` itself is 100% unique (verified), so this
    can't be caught by deduping on id.

    Key is (name, employer, description) -- NOT (name, employer, city,
    skills_clean), which was the original key and is measurably weaker.
    Checked directly: 332 rows share byte-identical `description` text
    with another row while differing in name/employer/city (e.g. the same
    templated posting from one employer, reposted verbatim across
    multiple cities) -- those survive the old key untouched. That matters
    specifically because `description` is now the single biggest feature
    (TF-IDF), so identical description text landing on both sides of the
    split is a real residual leak even when city differs. The new key
    catches 3,603 duplicates (10.2%) vs. the old key's 2,264 (6.4%) on the
    same data -- a strict improvement, not just a different tradeoff.
    `skills_clean` was dropped from the key too: it's only used for
    labels, not features, so it doesn't affect the leakage question this
    key is answering. (Skill-list order was also checked and ruled out
    separately: sorting skills_clean before comparing changed the old
    key's duplicate count by only 6 rows.)
    """
    before = df.count()
    df = df.dropDuplicates(["name", "employer", "description"])
    after = df.count()
    logger.info(f"Deduplicated postings: {before} -> {after} rows ({before - after} exact duplicates dropped)")
    return df


def _add_experience_and_work_format(df):
    """
    Legitimate, non-leaking job-attribute features that were sitting
    unused in the raw data: `experience` (4 balanced buckets, 0% missing)
    and `work_format` (on-site/remote/hybrid/field, multi-valued, 2.4%
    missing). Both are job-posting metadata, not derived from key_skills,
    so they don't reintroduce the leakage the leaky pipeline had.
    """
    df = df.withColumn("experience_bucket", F.col("experience.id"))
    for wf_id in WORK_FORMAT_IDS:
        df = df.withColumn(
            f"work_format_{wf_id.lower()}",
            F.expr(f"CASE WHEN exists(work_format, x -> x.id = '{wf_id}') THEN 1 ELSE 0 END"),
        )
    return df


def build_ml_pipeline_no_leakage(top_employers, top_skills):
    """
    Feature pipeline for city/employer/seniority/experience/work_format --
    deliberately does not reuse feature_engineering.build_ml_pipeline().

    That pipeline assembles "skills_tfidf" (a TF-IDF vector of skills_clean)
    and "skills_count" into the feature vector, but skills_clean is the exact
    same list every label_<skill> column is derived from. That leaks each
    label into its own features almost verbatim (e.g. the "python" TF-IDF
    component predicts label_python trivially) and contradicts the stated
    goal of predicting skills from job attributes, not the skills themselves
    (see CLAUDE.md). Confirmed empirically: with the leaky pipeline, every
    model/skill combination scored PR-AUC 0.98-1.00 on a 3k-row sample.

    feature_engineering.py is treated as frozen/external, so this reproduces
    only its non-leaking half (city/employer/seniority, one-hot + scaled)
    instead of editing it in place, plus experience/work_format on top.

    `employer` is also bucketed to `top_employers` + "Other" before indexing:
    the full 35k-row dataset has 12,081 distinct employers vs. 496 cities --
    one-hot encoding it directly produced a 12,581-dim vector that OOM'd
    (needed 3.32 GiB as a dense numpy array for the non-Spark bridge alone).

    NEVER add `_search_query` here: the raw data carries a scraping-artifact
    column recording which of 55 job-role search queries ("Python
    developer", "1C developer", ...) pulled in each row. It's not skills
    data, but it's strongly correlated with them (e.g. "python"'s positive
    rate is 7.5% overall vs. 43% within rows scraped under "Python
    developer" -- a 4-9x lift, measured directly). Using it as a feature
    would reproduce the same leakage bug this function was written to fix.

    Also includes TF-IDF over the job *title* (`name`), unlike the frozen
    pipeline's TF-IDF over `skills_clean`. Title text is a legitimate job
    attribute, not the internal skills tag list -- a title literally
    containing "Python Developer" telling you Python is likely required
    is real signal, not the label leaking into the features. (The original
    team's prior presentation used this same feature and got much
    stronger F1 for it -- see CLAUDE.md.)

    Also includes TF-IDF over `description` (HTML-stripped), with the
    top_skills tokens themselves excluded from its vocabulary. Measured
    directly: on a 1,000-row sample, descriptions contain 67% of their own
    posting's key_skills verbatim on average (27% contain literally all of
    them) -- naively TF-IDFing raw description text would reintroduce much
    of the same leak skills_tfidf had, since "python"/"sql"/etc. would
    become easily-learnable tokens shared across every skill's classifier.
    Excluding the skill-name tokens (via _skill_name_tokens) blocks that
    literal-word leak while keeping legitimate context: years of
    experience mentioned, methodology/team language, related-but-distinct
    technology mentions, etc.

    Also includes Word2Vec embeddings over both title and description
    tokens (added alongside TF-IDF, not replacing it -- exact keyword
    matching and semantic similarity are complementary signals, not
    redundant). TF-IDF is pure bag-of-words: a description saying
    "Django, REST APIs, PostgreSQL" without ever using the word "python"
    gets zero TF-IDF credit toward label_python, even though any human
    reader would know Python is implied. Word2Vec's distributional
    embeddings capture that kind of relatedness. Trained on the same
    already-cleaned token columns (title_tokens / desc_tokens), so
    description's skill-name exclusion applies here too -- no separate
    leakage surface introduced. Chose Spark ML's own Word2Vec
    (distributed, native) over an external pretrained embedding model
    specifically to stay inside the Spark ecosystem this course requires,
    at the cost of weaker embeddings than a large pretrained model would
    give (trained from scratch on this dataset's ~32k rows of text, not
    billions of tokens) -- an honest trade, not a hidden one.
    """
    escaped = [e.replace("'", "''") for e in top_employers if e is not None]
    in_list = ", ".join(f"'{e}'" for e in escaped)
    employer_bucketer = SQLTransformer(statement=(
        f"SELECT *, CASE WHEN employer IN ({in_list}) THEN employer ELSE 'Other' END AS employer_bucketed "
        f"FROM __THIS__"
    ))

    city_indexer = StringIndexer(inputCol="city", outputCol="city_index", handleInvalid="keep")
    employer_indexer = StringIndexer(inputCol="employer_bucketed", outputCol="employer_index", handleInvalid="keep")
    seniority_indexer = StringIndexer(inputCol="seniority", outputCol="seniority_index", handleInvalid="keep")
    experience_indexer = StringIndexer(inputCol="experience_bucket", outputCol="experience_index", handleInvalid="keep")

    city_encoder = OneHotEncoder(inputCol="city_index", outputCol="city_vector")
    employer_encoder = OneHotEncoder(inputCol="employer_index", outputCol="employer_vector")
    seniority_encoder = OneHotEncoder(inputCol="seniority_index", outputCol="seniority_vector")
    experience_encoder = OneHotEncoder(inputCol="experience_index", outputCol="experience_vector")

    # Job titles mix Russian and English ("Senior Python Developer",
    # "Менеджер по продажам") -- a simple word-boundary tokenizer plus
    # combined RU+EN stopword lists handles both without extra NLP deps.
    title_tokenizer = RegexTokenizer(
        inputCol="name", outputCol="title_tokens_raw", pattern=r"\W+", toLowercase=True, minTokenLength=2,
    )
    title_stopwords = StopWordsRemover.loadDefaultStopWords("english") + StopWordsRemover.loadDefaultStopWords("russian")
    title_stopword_remover = StopWordsRemover(
        inputCol="title_tokens_raw", outputCol="title_tokens", stopWords=title_stopwords,
    )
    title_cv = CountVectorizer(
        inputCol="title_tokens", outputCol="title_raw_features",
        vocabSize=TITLE_TFIDF_VOCAB_SIZE, minDF=TITLE_TFIDF_MIN_DF,
    )
    title_idf = IDF(inputCol="title_raw_features", outputCol="title_tfidf")

    # description is raw HTML ("<p><strong>...</strong></p>...") -- strip
    # tags before tokenizing, same RU+EN stopwords as title plus the
    # skill-name exclusion list (see docstring above).
    description_cleaner = SQLTransformer(statement=(
        "SELECT *, regexp_replace(description, '<[^>]+>', ' ') AS description_clean FROM __THIS__"
    ))
    desc_tokenizer = RegexTokenizer(
        inputCol="description_clean", outputCol="desc_tokens_raw", pattern=r"\W+", toLowercase=True, minTokenLength=2,
    )
    desc_stopwords = (
        StopWordsRemover.loadDefaultStopWords("english")
        + StopWordsRemover.loadDefaultStopWords("russian")
        + _skill_name_tokens(top_skills)
    )
    desc_stopword_remover = StopWordsRemover(
        inputCol="desc_tokens_raw", outputCol="desc_tokens", stopWords=desc_stopwords,
    )
    desc_cv = CountVectorizer(
        inputCol="desc_tokens", outputCol="desc_raw_features",
        vocabSize=DESCRIPTION_TFIDF_VOCAB_SIZE, minDF=DESCRIPTION_TFIDF_MIN_DF,
    )
    desc_idf = IDF(inputCol="desc_raw_features", outputCol="description_tfidf")

    # Word2Vec on the same cleaned token columns TF-IDF already uses --
    # complementary signal (semantic similarity), not a replacement for
    # exact-keyword matching. See docstring above.
    title_w2v = Word2Vec(
        inputCol="title_tokens", outputCol="title_w2v",
        vectorSize=WORD2VEC_VECTOR_SIZE, minCount=WORD2VEC_MIN_COUNT, seed=42,
    )
    desc_w2v = Word2Vec(
        inputCol="desc_tokens", outputCol="description_w2v",
        vectorSize=WORD2VEC_VECTOR_SIZE, minCount=WORD2VEC_MIN_COUNT, seed=42,
    )

    work_format_cols = [f"work_format_{wf_id.lower()}" for wf_id in WORK_FORMAT_IDS]

    assembler = VectorAssembler(
        inputCols=(
            ["city_vector", "employer_vector", "seniority_vector", "experience_vector",
             "title_tfidf", "description_tfidf", "title_w2v", "description_w2v"]
            + work_format_cols
        ),
        outputCol="features",
    )
    scaler = StandardScaler(inputCol="features", outputCol="scaled_features", withStd=True, withMean=True)

    return Pipeline(stages=[
        employer_bucketer,
        city_indexer, employer_indexer, seniority_indexer, experience_indexer,
        city_encoder, employer_encoder, seniority_encoder, experience_encoder,
        title_tokenizer, title_stopword_remover, title_cv, title_idf,
        description_cleaner, desc_tokenizer, desc_stopword_remover, desc_cv, desc_idf,
        title_w2v, desc_w2v,
        assembler, scaler,
    ])


def add_class_weight_column(df, label_col, weight_col="class_weight"):
    """
    Add a balanced per-row weight column for a binary label:
    weight = n_samples / (n_classes * class_count), the standard "balanced"
    formula (same one sklearn's compute_sample_weight('balanced', y) uses).

    Needed because Spark ML's classifiers (SPARK_MODEL_FACTORIES) don't do
    any imbalance handling by default. Measured directly on this dataset's
    heavily imbalanced labels: without this, spark_random_forest's default
    0.5 threshold essentially never fires -- precision=recall=f1=0.000
    averaged across 20 skills. Every DataFrame passed to a Spark model's
    .fit() must carry this column, since the factories set
    weightCol="class_weight" unconditionally.

    Recomputed per skill (not a single static column on the full
    DataFrame), since the weight depends on which skill's label is being
    predicted -- the same row gets a different weight for label_python
    than for label_sql.
    """
    rows = df.groupBy(label_col).count().collect()
    counts = {row[label_col]: row["count"] for row in rows}
    total = sum(counts.values())
    n_classes = len(counts)
    pos_weight = total / (n_classes * counts.get(1, 1))
    neg_weight = total / (n_classes * counts.get(0, 1))
    return df.withColumn(
        weight_col,
        F.when(F.col(label_col) == 1, F.lit(pos_weight)).otherwise(F.lit(neg_weight)),
    )


def best_f1_threshold(y_true, probs, min_positives=5, default=0.5):
    """Pick the probability threshold that maximizes F1 on (y_true, probs).
    Falls back to 0.5 when there's too little positive signal to pick a
    threshold reliably (avoids overfitting the threshold to a handful of
    validation-set positives). Shared by train_nonspark_models.py and
    train_spark_native.py -- both need it on the same held-out "val" slice,
    never on "test" (that would leak test information into the reported
    F1)."""
    y_true = np.asarray(y_true)
    if y_true.sum() < min_positives:
        return default
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    if len(thresholds) == 0:
        return default
    f1s = 2 * precision * recall / (precision + recall + 1e-12)
    return float(thresholds[np.argmax(f1s[:-1])])


def _get_skill_list(df, top_n_skills, skill_list_path):
    path = Path(skill_list_path)
    if path.exists():
        skills = json.loads(path.read_text())
        logger.info(f"Reusing cached skill list from {path} ({len(skills)} skills)")
        return skills

    skills = get_top_skills(df, n=top_n_skills)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(skills, ensure_ascii=False, indent=2))
    logger.info(f"Computed and cached skill list to {path} ({len(skills)} skills)")
    return skills


def _get_top_employers(df, top_n_employers, employer_list_path):
    path = Path(employer_list_path)
    if path.exists():
        employers = json.loads(path.read_text())
        logger.info(f"Reusing cached employer list from {path} ({len(employers)} employers)")
        return employers

    rows = (
        df.groupBy("employer").count()
        .orderBy(F.desc("count"))
        .limit(top_n_employers)
        .collect()
    )
    employers = [row["employer"] for row in rows if row["employer"] is not None]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(employers, ensure_ascii=False, indent=2))
    logger.info(f"Computed and cached employer list to {path} ({len(employers)} employers)")
    return employers


def load_and_engineer(spark, data_path: str, top_n_skills: int = 20,
                       test_fraction: float = 0.2, val_fraction: float = 0.15,
                       skill_list_path: str = SKILL_LIST_PATH,
                       top_n_employers: int = 50, employer_list_path: str = EMPLOYER_LIST_PATH,
                       pipeline_model=None):
    """
    Args:
        pipeline_model: an already-fitted PipelineModel to REUSE instead of
            fitting a fresh one on this call's data. Pass this whenever more
            than one data file needs to share one consistent feature space
            -- e.g. retrain_on_new_data.py transforming both old_data and
            new_data through the SAME pipeline the original model was
            trained with, rather than each file getting its own
            independently-fit StringIndexer/CountVectorizer/Word2Vec/
            StandardScaler. This was a real, previously-undetected bug: two
            separate load_and_engineer() calls on different files produce
            "scaled_features" columns with the same name but incompatible
            meaning/dimensions (different vocab, different indices,
            different scaler stats), so unioning them (train_spark_native's
            retrain path) or fine-tuning across them (non-Spark's
            init_model=/xgb_model= path) silently trains on garbage.
            Caught via actually running the incremental-retraining demo end
            to end -- crashed with ArrayIndexOutOfBoundsException the first
            time this code path was ever executed. Doesn't affect the main
            training results anywhere else in this project: those always
            call load_and_engineer on the same single file, and Spark ML's
            fit here is deterministic (fixed seeds), so independently
            re-fitting on identical data reliably produces an equivalent
            pipeline anyway -- the bug only bites when the *data differs*
            across calls that need to share a feature space.

    Returns:
        df_features: Spark DataFrame with `scaled_features`, `features_array`,
                      one `label_<skill>` column per skill, and a `split`
                      column ("train" / "val" / "test"). "val" is a slice of
                      what used to be "train", carved out for threshold/
                      hyperparameter selection so nothing gets tuned against
                      the held-out test set; "test" itself is unchanged.
        top_skills: list of skill names being predicted (cached after the
                    first run so it stays identical across scripts/datasets).
        pipeline_model: the fitted PipelineModel used (whether newly fit or
                        the one passed in) -- pass this back in on a later
                        call to guarantee the same feature space.
    """
    if spark.conf.get("spark.master", "").startswith("k8s://") and not data_path.startswith("hdfs://"):
        # Executors on the k8s cluster don't share the driver's local/NFS
        # filesystem -- spark.read.parquet(local_path) would fail on them.
        # See local_parquet_loader.py's docstring for the full reasoning
        # (this is what lets us run without HDFS, which was down for an
        # extended period -- see CLAUDE.md).
        from local_parquet_loader import load_local_parquet_distributed
        df = load_local_parquet_distributed(spark, data_path)
    else:
        df = spark.read.parquet(data_path)
    logger.info(f"Loaded {df.count()} rows from {data_path}")

    df = extract_basic_features(df)
    df = _add_experience_and_work_format(df)
    df = _dedupe_postings(df)

    top_skills = _get_skill_list(df, top_n_skills, skill_list_path)
    df = create_multi_label_columns(df, top_skills)

    if pipeline_model is None:
        top_employers = _get_top_employers(df, top_n_employers, employer_list_path)
        pipeline = build_ml_pipeline_no_leakage(top_employers, top_skills)
        pipeline_model = pipeline.fit(df)
    df_features = pipeline_model.transform(df)

    train_cutoff = int((1 - test_fraction - val_fraction) * 100)
    test_cutoff = int((1 - test_fraction) * 100)
    h = F.abs(F.crc32(F.col("id").cast("string"))) % 100
    df_features = df_features.withColumn(
        "split",
        F.when(h < train_cutoff, "train").when(h < test_cutoff, "val").otherwise("test"),
    )
    df_features = df_features.withColumn("features_array", vector_to_array("scaled_features"))

    return df_features, top_skills, pipeline_model
