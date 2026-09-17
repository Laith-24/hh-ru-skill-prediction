"""Standalone smoke test for spark_session.build_k8s_spark_session -- verifies
executor pods can actually be scheduled and communicate with the driver,
independent of HDFS (which is down as of 2026-08-24, see CLAUDE.md).
Usage (from inside the jupyter driver pod): python3 k8s_smoke_test.py
"""
from spark_session import build_k8s_spark_session

spark = build_k8s_spark_session("hh-ru-k8s-smoke-test")
n = spark.range(1000).rdd.map(lambda row: row.id * 2).sum()
print(f"SMOKE_TEST_RESULT={n}")  # expect 999000
spark.stop()
