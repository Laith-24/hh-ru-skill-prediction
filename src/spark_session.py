"""
Builds a SparkSession for the course's actual submission mechanism: genuine
Kubernetes-native Spark (`.master("k8s://...")`), driver running in-process
inside the Jupyter pod, executors launched as separate pods from the
course's `spark-executor` image. Config values match the course's own
`SparkLab-Template.ipynb` (the reference notebook mounted read-only at
`~/shared-data/notebooks/BDML/` in every student's pod) -- this is NOT the
plain spark-submit/local[*] pattern the rest of this project's local
scripts use.

Only used when --k8s is passed to a training script; local runs are
untouched (plain SparkSession.builder.appName(...).getOrCreate()).
"""

import os
import socket

from pyspark.sql import SparkSession

LOGIN = "lahmad-475790"  # gateway.st login == Kubernetes namespace


def build_k8s_spark_session(app_name: str) -> SparkSession:
    local_ip = socket.gethostbyname(socket.gethostname())
    os.makedirs("/tmp/spark", exist_ok=True)

    return (
        SparkSession.builder
        .appName(app_name)
        .master("k8s://https://10.32.7.103:6443")
        .config("spark.driver.host", local_ip)
        .config("spark.driver.bindAddress", "0.0.0.0")
        # Disabled: Spark's UI/listener-bus accumulates per-stage bookkeeping
        # in driver memory for the session's whole lifetime, a real
        # contributor to an OOM killer crash on a long multi-skill run.
        .config("spark.ui.enabled", "false")
        .config("spark.driver.cores", "1")
        .config("spark.driver.memory", "4g")
        .config("spark.executor.instances", "3")
        .config("spark.executor.cores", "2")
        .config("spark.kubernetes.executor.request.cores", "0.1")
        .config("spark.executor.memory", "6g")
        .config("spark.kubernetes.memoryOverheadFactor", "0.2")
        .config("spark.memory.fraction", "0.6")
        .config("spark.memory.storageFraction", "0.5")
        .config("spark.network.timeout", "180s")
        .config("spark.sql.autoBroadcastJoinThreshold", "-1")
        # 32, not Spark's default 200: heavy oversharding for 3 executors x
        # 2 cores. Not 16 either -- that concentrated the wide cached
        # feature vectors into fewer, larger partitions and contributed to
        # OOM; 32 is a compromise between the two failure modes.
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.default.parallelism", "32")
        # Raised retry tolerance: full-scale runs hit both a transient CNI
        # sandbox failure and genuine memory pressure during a heavy
        # shuffle stage -- retrying is strictly better than aborting the
        # whole job over one bad attempt, and the namespace has spare quota.
        .config("spark.stage.maxConsecutiveAttempts", "10")
        .config("spark.task.maxFailures", "8")
        .config("spark.kubernetes.namespace", LOGIN)
        .config("spark.kubernetes.driver.label.appname", app_name)
        .config("spark.kubernetes.executor.label.appname", app_name)
        .config("spark.kubernetes.container.image", f"node03.st:5000/spark-executor:{LOGIN}")
        .config("spark.kubernetes.container.image.pullPolicy", "Always")
        .config("spark.kubernetes.executor.deleteOnTermination", "true")
        .config("spark.local.dir", "/tmp/spark")
        .getOrCreate()
    )
