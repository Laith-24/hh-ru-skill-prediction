# hh.ru Skill Prediction

Big Data course project on real hh.ru (Russian job board) vacancy data, in
two parts:

1. **Skill prediction** — given a job posting's title and attributes (not
   its skills), predict which of the top-20 in-demand skills it needs (top 50 for the
   three boosting models).
   Multi-label: one binary classifier per skill, 6 models compared.
2. **Skill Combination Value Calculator** — given the skills you already
   know, find which skill to learn next for the biggest salary premium,
   using Association Rule Mining (Spark FPGrowth) priced against real
   salary data, adjusted for seniority/city/salary-disclosure confounds.

## Setup
```bash
pip install -r requirements.txt
```

`models/` and trained-model backups are gitignored. You can regenerate
them by running the scripts below.

## Part 1: Skill prediction

6 models, same leakage-free features (city/employer/seniority/experience/
work-format + title & description TF-IDF + Word2Vec embeddings), same
id-hash train/val/test split, threshold-tuned and class-weighted:

| Model | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| xgboost | 0.406 | 0.899 | 0.409 |
| lightgbm | 0.396 | 0.896 | 0.404 |
| catboost | 0.373 | 0.894 | 0.392 |
| spark_gbt | 0.326 | 0.855 | 0.361 |
| spark_random_forest | 0.315 | 0.895 | 0.374 |
| spark_logistic_regression | 0.291 | 0.893 | 0.362 |

The text features use a Unicode-aware tokenizer (`(?U)\W+`). The first
version used a plain `\W+`, which Java treats as ASCII-only, so Cyrillic
letters counted as separators and Russian titles and descriptions
produced no tokens. Fixing it lifted every model (macro PR-AUC for
xgboost went from 0.305 to 0.406). The hyperparameters
were tuned before that fix and have not been re-tuned.

The three Spark-native models were also trained fully distributed on the
course's Kubernetes cluster (Spark 3.1.1, three executor pods) with the
fixed tokenizer: `results/spark_results_pod_final6_unicode.csv`. Against
the local run's own results file (`results/spark_results_final6_unicode.csv`),
macro F1 differs by at most 0.0015 and macro PR-AUC by at most 0.0085 for
each model. The 60 fits took 3 h 53 min on the cluster and 1 h 45 min
locally.

The 20 most frequent skills are only a small part of what postings list,
and skills such as PHP, Kubernetes or TypeScript are not among them, so those
models can never predict them. The three boosting models were therefore also
trained on the 50 most frequent skills (`configs/top_skills_50.json`, whose
first 20 entries are the list above). The 30 extra skills are about as
learnable as the first 20:

| Model | PR-AUC, skills 1–20 | PR-AUC, skills 21–50 | PR-AUC, all 50 |
|---|---|---|---|
| xgboost | 0.401 | 0.381 | 0.389 |
| lightgbm | 0.393 | 0.379 | 0.385 |
| catboost | 0.374 | 0.358 | 0.365 |

The extra skills are rarer, so their per-skill scores are noisier. The first
column differs a little from the table above because the description features
now also leave out the words of the 30 extra skill names. The Spark models
were not trained on 50 skills.

The live demo only knows a job title, while the models above also use the
description, employer, city and experience. Given a title and an empty
description, the full-feature CatBoost models score macro PR-AUC
0.19 and F1 0.11 on the held-out titles. Models trained on the title's
features only (its TF-IDF and Word2Vec vectors and the seniority flags read
from it, `--features title`) do better on exactly that input:

| Model, title features only | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| catboost | 0.236 | 0.863 | 0.290 |
| xgboost | 0.230 | 0.860 | 0.278 |
| lightgbm | 0.229 | 0.863 | 0.279 |

```bash
# train + evaluate all 6 models
spark-submit src/train_spark_native.py --data data/vacancies.parquet
python3 src/train_nonspark_models.py --data data/vacancies.parquet
python3 src/compare_results.py

# the boosting models on the 50 most frequent skills (own list, results and models)
python3 src/train_nonspark_models.py --data data/vacancies.parquet --top-n-skills 50 \
    --skill-list-path configs/top_skills_50.json --output results/nonspark_results_top50.csv \
    --models-dir models/nonspark_top50 --pipeline-path models/feature_pipeline_top50

# the same on the title's features only (what the live demo uses)
python3 src/train_nonspark_models.py --data data/vacancies.parquet --top-n-skills 50 \
    --skill-list-path configs/top_skills_50.json --features title \
    --output results/title_only_results_top50.csv --models-dir models/title_only_top50 \
    --pipeline-path models/feature_pipeline_title_only_top50

# optional: hyperparameter search (writes configs/tuned_params.json,
# picked up automatically by the training scripts above)
python3 src/tune_hyperparameters.py --data data/vacancies.parquet

# fine-tune on newly arrived postings instead of retraining from scratch
python3 src/retrain_on_new_data.py --new-data data/new_vacancies.parquet \
    --old-data data/vacancies.parquet
```

**Live demo**: `streamlit run interface/app.py` — enter a job title, get
predicted skills from the title-only CatBoost models for 50 skills (train
them first, see the last command above). On held-out titles they score
macro PR-AUC 0.24 and F1 0.29.

## Part 2: Skill Combination Value Calculator

"If I know these skills, what should I learn next to earn more?" Mines
frequent skill combinations (Spark FPGrowth) and prices each one from
real salary data. A raw average mostly measures seniority and city, not
the skill itself, so every premium is computed within the same
experience level, city, and salary-disclosure type instead — postings
that give only a floor pay ~20% more on average than ones giving a full
range, so that gets controlled for too.

Current run (v3: 14,860 usable postings, targeting the disclosed starting
salary):

- Held-out R², same test split, three independent libraries: **CatBoost
  0.462, XGBoost 0.475, LightGBM 0.460** — convergent evidence the skill
  signal is real, not one library's artifact.
- Every reported premium is bootstrap-tested (1,000 resamples); **226/316
  (71.5%)** survive a 95% confidence interval excluding zero — only those
  are presented as findings.
- Example: Kubernetes carries a **+36,864 RUB/mo** premium (95% CI
  [25,498, 46,860]), consistent across all three models.
- A selection-bias check on who discloses salary at all (younger,
  retail-skewed, less senior/Moscow-heavy than the full market) is run
  and controlled for, not assumed away.

```bash
python3 src/skill_combo_value.py --data data/hh_vacancies_cleaned_fixed_salary.parquet --no-require-ceiling --suffix _v3
python3 src/skill_combo_bootstrap_ci.py --suffix _v3 --no-require-ceiling
python3 src/skill_combo_selection_bias.py --suffix _v3 --no-require-ceiling
python3 src/skill_salary_regression.py --suffix _v3 --no-require-ceiling
python3 src/skill_salary_catboost_city.py --suffix _v3 --no-require-ceiling
python3 src/skill_salary_gbm_comparison.py --suffix _v3 --no-require-ceiling
```

**Live demo**: `streamlit run interface/skill_value_app.py` — pick the
skills you know, get ranked, data-backed recommendations for what to
learn next. With the raw parquet in `data/`, it also shows basic
statistics and recent real postings for the skills you pick.

## Project structure
```
src/
  feature_engineering.py       # provided pipeline (unmodified)
  data_pipeline.py            # shared load + feature engineering + split + cached skill list
  models_config.py            # registry of the 6 skill-prediction models
  train_spark_native.py       # Spark-native models, per skill
  train_nonspark_models.py    # non-Spark models, via a pandas bridge
  tune_hyperparameters.py     # hyperparameter search across all 6 models
  retrain_on_new_data.py      # fine-tune/retrain on new data
  compare_results.py          # merges result sets, prints comparison
  check_data_quality.py       # data quality report (missing salary/skills, real duplicate rate)

  skill_combo_value.py            # core pivot: mine + price skill combinations
  skill_combo_bootstrap_ci.py     # bootstrap CIs on every premium
  skill_combo_selection_bias.py   # checks who discloses salary at all
  skill_salary_regression.py      # held-out Ridge regression eval
  skill_salary_catboost_city.py   # held-out CatBoost eval + SHAP city/ceiling breakdown
  skill_salary_gbm_comparison.py  # CatBoost vs XGBoost vs LightGBM cross-check

interface/
  app.py               # live demo: title -> predicted skills
  skill_value_app.py   # live demo: known skills -> what to learn next

deploy/           # local (Windows PowerShell) and pod (Kubernetes) run scripts
configs/          # cached top-20/top-50 skill lists, top-employer vocab, tuned hyperparameters
results/          # every script's output tables (small CSVs, tracked in git
                   # since the live apps read them directly at runtime)
```

## Deploy to the course cluster
```bash
./deploy/deploy_to_pod.sh                    # train all 3 Spark-native models on the real cluster
./deploy/run_finetune_demo_on_pod.sh         # fine-tune-on-new-data demo, Spark-native models only
```
Requires SSH key auth configured as Host `pod` in `~/.ssh/config`. Runs on
a real Kubernetes cluster via genuine `k8s://` native Spark submission
(`kubectl exec` into the Jupyter driver pod — there's no plain
`spark-submit` gateway on this cluster). CatBoost/XGBoost/LightGBM aren't
installed on the pod (no outbound internet there) and don't need to be:
they're non-distributed and already have solid results from local runs
(`src/train_nonspark_models.py`, `src/demo_finetune_single_process.py`).

A full 60-fit sweep runs for hours, and the driver pod's 8 GiB memory limit
can kill the driver process partway (it did twice in the last run).
`deploy/spark_sweep_on_pod.py` therefore runs the sweep in chunks of 3
skills, each in a fresh Spark session, appends every result as soon as it
is computed, and resumes where it stopped when started again. Run it in
the driver pod, in the background, after `deploy_to_pod.sh` has synced the
code and data.

## Data collection
`data/vacancies.parquet` was originally gathered via the hh.ru API using
`src/collection/` (`auth.py`, `collector.py`, from
[hh-ru-big-data-analysis](https://github.com/joullanarAli/hh-ru-big-data-analysis)):
OAuth2 client-credentials auth, rate-limited and checkpointed collection
across ~55 role-based search queries. The rest of the project just
consumes the already-collected parquet file, so this is here for
provenance rather than being part of the live pipeline.

To collect fresh data yourself, register an hh.ru API application at
https://dev.hh.ru/admin, copy `.env.example` to `.env` with your
`HH_CLIENT_ID`/`HH_CLIENT_SECRET`, then:
```bash
python3 src/collection/collect_data.py
```
Saves raw vacancies as JSON under `data/output/`; convert to parquet
before using with the rest of this project (e.g.
`pandas.read_json(...).to_parquet("data/vacancies.parquet")`).
