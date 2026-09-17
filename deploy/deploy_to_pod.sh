#!/usr/bin/env bash
# Deploys and runs Spark-native training (RandomForest/GBT/LogisticRegression)
# on the course's k8s cluster, via genuine Kubernetes-native Spark
# (.master("k8s://..."), see src/spark_session.py) -- kubectl exec into the
# Jupyter driver pod, since there's no spark-submit gateway here.
#
# Executor pods don't share the driver's NFS filesystem, so data loading
# goes through src/local_parquet_loader.py instead of needing HDFS as
# shared storage: the driver reads the parquet via pyarrow and ships it to
# executors directly (load_and_engineer() does this automatically whenever
# --k8s is set), so this script only needs the file on the driver's NFS path.
#
# Only runs train_spark_native.py -- catboost/xgboost/lightgbm are
# non-distributed and driver-only, so they don't need this cluster and
# already have solid results from local runs.
#
# Prerequisite: SSH key auth for the pod, Host "pod" in ~/.ssh/config.
# Usage: ./deploy/deploy_to_pod.sh

set -euo pipefail

POD_HOST="pod"
NAMESPACE="lahmad-475790"
NFS_REMOTE_DIR="/nfs/home/${NAMESPACE}/hh-ru-skill-prediction"
LOCAL_DATA_FILE="data/vacancies.parquet"
POD_DATA_PATH="/home/jovyan/nfs-home/hh-ru-skill-prediction/data/vacancies.parquet"
SPARK_ENV='export SPARK_HOME=/usr/local/spark; export PYTHONPATH=$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.10.9-src.zip:$PYTHONPATH;'

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
    src configs deploy requirements.txt CLAUDE.md README.md .gitignore \
    | ssh "$POD_HOST" "cd $NFS_REMOTE_DIR && tar xzf -"

echo "Syncing $LOCAL_DATA_FILE to $POD_HOST:$NFS_REMOTE_DIR/data/ (this is a ~500MB file, may take a while) ..."
ssh "$POD_HOST" "mkdir -p $NFS_REMOTE_DIR/data"
tar czf - "$LOCAL_DATA_FILE" | ssh "$POD_HOST" "cd $NFS_REMOTE_DIR && tar xzf -"

echo "Running train_spark_native.py --k8s on the pod ..."
ssh "$POD_HOST" "kubectl exec -n $NAMESPACE $POD -- sh -c '
    $SPARK_ENV
    cd /home/jovyan/nfs-home/hh-ru-skill-prediction/src &&
    python3 train_spark_native.py --data $POD_DATA_PATH --k8s
'"

echo "Pulling results back locally ..."
mkdir -p results
ssh "$POD_HOST" "cd $NFS_REMOTE_DIR && tar czf - results" | tar xzf -

echo "Done. See results/spark_results.csv."
