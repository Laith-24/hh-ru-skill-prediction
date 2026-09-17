#!/usr/bin/env bash
# Demo the fine-tune-on-new-data capability on the real course cluster,
# via the same k8s:// submission mechanism deploy_to_pod.sh uses (kubectl
# exec into the Jupyter driver pod -- there is no spark-submit gateway
# here; see CLAUDE.md).
#
# Runs src/demo_finetune_single_process_pod.py -- ALL FOUR demo steps
# (baseline train, evaluate before, retrain on old+new, evaluate after)
# in ONE Spark session/process, never round-tripping a model through
# PipelineModel.load()/model.load(). A first attempt used separate
# kubectl exec invocations per step instead (train_spark_native.py ->
# demo_evaluate.py -> retrain_on_new_data.py -> demo_evaluate.py) and
# failed reproducibly: confirmed directly on the pod that a saved
# pipeline/model's metadata/ directory held only an empty _SUCCESS
# marker, no content file. Root cause: executor pods don't share the
# driver's NFS mount, so the metadata write's single partition can land
# on an executor's own ephemeral local disk instead of shared storage,
# invisible to any later process -- a structural limit of this cluster,
# not a slow-to-appear write (retrying the load 5x over 50s didn't help).
# Same class of problem, different cause, as the local demo's own
# PipelineModel.load() issue on Windows -- fixed the same way: never
# reload mid-run, keep everything in memory (see
# src/demo_finetune_single_process_pod.py and
# src/demo_finetune_single_process.py's docstring).
#
# Scoped to the 3 Spark-native models only (RandomForest/GBT/Logistic
# Regression), not all 6: catboost/xgboost/lightgbm aren't installed on
# the pod (no outbound internet there to pip install them) and don't need
# this cluster at all -- they're non-distributed, driver-only. The full
# 6-model before/after comparison already exists from a local run (see
# src/demo_finetune_single_process.py and CLAUDE.md's fine-tuning-demo
# section).
#
# Uses the SAME demo/old/new/eval split already generated locally
# (data/demo/*.parquet, via src/make_demo_split.py) rather than
# regenerating it on the pod, so both runs compare on identical rows.
# Writes to demo_eval_before_pod.csv / demo_eval_after_pod.csv (not the
# existing demo_eval_before.csv / after.csv) so this never overwrites the
# local full-6-model results.
#
# Prerequisite: same as deploy_to_pod.sh (SSH key auth as Host "pod";
# data/demo/*.parquet already generated locally).
# Usage: ./deploy/run_finetune_demo_on_pod.sh

set -euo pipefail

POD_HOST="pod"
NAMESPACE="lahmad-475790"
NFS_REMOTE_DIR="/nfs/home/${NAMESPACE}/hh-ru-skill-prediction"
POD_PROJECT_DIR="/home/jovyan/nfs-home/hh-ru-skill-prediction"
SPARK_ENV='export SPARK_HOME=/usr/local/spark; export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9-src.zip:$PYTHONPATH;'

if [ ! -f data/demo/demo_old.parquet ]; then
    echo "data/demo/*.parquet not found -- run this first:" >&2
    echo "  python3 src/make_demo_split.py --data data/vacancies.parquet --out-dir data/demo" >&2
    exit 1
fi

echo "Finding the current jupyter-spark pod (name changes every redeploy) ..."
POD=$(ssh "$POD_HOST" "kubectl get pods -n $NAMESPACE -l app=jupyter-spark -o jsonpath='{.items[0].metadata.name}'")
if [ -z "$POD" ]; then
    echo "No running jupyter-spark pod found in namespace $NAMESPACE. Deploy it first (see CLAUDE.md)." >&2
    exit 1
fi
echo "Using pod: $POD"

echo "Syncing code to $POD_HOST:$NFS_REMOTE_DIR (tar-over-ssh -- plain 'scp -r' fails on this Windows OpenSSH client for directories) ..."
ssh "$POD_HOST" "mkdir -p $NFS_REMOTE_DIR"
tar czf - --exclude 'data' --exclude '.venv' --exclude '__pycache__' --exclude 'results' --exclude 'models' --exclude 'models_backup_final_table' --exclude 'catboost_info' --exclude 'spark-tmp' \
    src configs deploy requirements.txt README.md .gitignore \
    | ssh "$POD_HOST" "cd $NFS_REMOTE_DIR && tar xzf -"

echo "Syncing data/demo/*.parquet to $POD_HOST:$NFS_REMOTE_DIR/data/demo/ ..."
ssh "$POD_HOST" "mkdir -p $NFS_REMOTE_DIR/data/demo"
tar czf - data/demo/demo_old.parquet data/demo/demo_new.parquet data/demo/demo_eval.parquet \
    | ssh "$POD_HOST" "cd $NFS_REMOTE_DIR && tar xzf -"

echo "Running fine-tuning demo on pod (Spark-native models only, single process) ..."
ssh "$POD_HOST" "kubectl exec -n $NAMESPACE $POD -- sh -c '
    $SPARK_ENV
    cd $POD_PROJECT_DIR &&
    python3 src/demo_finetune_single_process_pod.py
'"

echo "Pulling demo results back locally ..."
mkdir -p results
ssh "$POD_HOST" "cd $NFS_REMOTE_DIR && tar czf - results/demo_eval_before_pod.csv results/demo_eval_after_pod.csv" | tar xzf -

echo ""
echo "Done. Compare results/demo_eval_before_pod.csv vs results/demo_eval_after_pod.csv"
echo "for the Spark-native before/after on the real cluster. For the full 6-model"
echo "comparison (including catboost/xgboost/lightgbm), see the local run:"
echo "results/demo_eval_before.csv / demo_eval_after.csv (src/demo_finetune_single_process.py)."
