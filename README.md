# hh.ru Skill Prediction

Big Data course project on real hh.ru (Russian job board) vacancy data, in
two parts:

1. **Skill prediction** — given a job posting's title and attributes (not
   its skills), predict which of the top-20 in-demand skills it needs.
   Multi-label: one binary classifier per skill, 6 models compared.
2. **Skill Combination Value Calculator** — given the skills you already
   know, find which skill to learn next for the biggest salary premium,
   using Association Rule Mining (Spark FPGrowth) priced against real
   salary data, adjusted for seniority/city/salary-disclosure confounds.

## Setup
```bash
pip install -r requirements.txt
```
This project uses two data collections (both gitignored — see "Data
collection" below for how they were gathered):
- `data/vacancies.parquet` (35,368 rows) — used by Part 1 (skill
  prediction).
- `data/hh_vacancies_cleaned_fixed_salary.parquet` (28,531 rows,
  collected later) — used by Part 2 (skill combination value
  calculator). Not a superset of the first file: kept the original's
  salary-complete postings, dropped most incomplete ones, added ~11,900
  new postings. Collected specifically for far better salary-field
  completeness, which Part 2 needs and Part 1 never used.

`models/` and trained-model backups are also gitignored — regenerate
them by running the scripts below.

## Part 1: Skill prediction

6 models, same leakage-free features (city/employer/seniority/experience/
work-format + title & description TF-IDF + Word2Vec embeddings), same
id-hash train/val/test split, threshold-tuned and class-weighted:

| Model | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| xgboost | 0.305 | 0.846 | 0.321 |
| lightgbm | 0.302 | 0.846 | 0.313 |
| catboost | 0.293 | 0.842 | 0.314 |
| spark_gbt | 0.260 | 0.818 | 0.306 |
| spark_random_forest | 0.245 | 0.850 | 0.299 |
| spark_logistic_regression | 0.226 | 0.850 | 0.301 |

The three Spark-native models were also trained fully distributed on the
course's real Kubernetes cluster, not just locally — results matched the
local run within 0.01–0.02 PR-AUC.

```bash
# train + evaluate all 6 models
spark-submit src/train_spark_native.py --data data/vacancies.parquet
python3 src/train_nonspark_models.py --data data/vacancies.parquet
python3 src/compare_results.py

# optional: hyperparameter search (writes configs/tuned_params.json,
# picked up automatically by the training scripts above)
python3 src/tune_hyperparameters.py --data data/vacancies.parquet

# fine-tune on newly arrived postings instead of retraining from scratch
python3 src/retrain_on_new_data.py --new-data data/new_vacancies.parquet \
    --old-data data/vacancies.parquet
```

**Live demo**: `streamlit run interface/app.py` — enter a job title, get
predicted skills. Deliberately title-only: adding description text
actually made predictions noisier, not better (see the app's docstring).

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
learn next.

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
configs/          # cached top-skill/top-employer vocab, tuned hyperparameters
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
