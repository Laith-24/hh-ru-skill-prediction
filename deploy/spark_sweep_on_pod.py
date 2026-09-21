"""Runs the three Spark-native models over the 20 skills on the course cluster in chunks of 3 skills, one fresh driver
process per chunk, because a single long session runs into the driver pod's 8 GiB memory limit. Results are appended after
every fit; starting the script again continues with the first skill that has fewer than 3 model rows in the output file.

Run it inside the jupyter driver pod, in the background, after deploy_to_pod.sh has synced the code and data:
    nohup python3 /home/jovyan/nfs-home/hh-ru-skill-prediction/deploy/spark_sweep_on_pod.py > sweep.out 2>&1 &
Progress is in sweep.out and the per-chunk logs in logs_final6/. Uses the pod's Python 3.8 and no extra packages."""
import csv
import json
import os
import signal
import subprocess
import time
from pathlib import Path

ROOT = Path("/home/jovyan/nfs-home/hh-ru-skill-prediction")
OUT_REL = "results/spark_results_pod_final6_unicode.csv"
OUT = ROOT / OUT_REL
MODELS = "models/spark_pod_final6"
PIPE = "models/feature_pipeline_pod_final6"
DATA = str(ROOT / "data" / "vacancies.parquet")
LOGS = ROOT / "logs_final6"
CHUNK = 3
MAX_STALLS = 4
CHUNK_TIMEOUT_S = int(2.5 * 3600)

LOGS.mkdir(exist_ok=True)
skills = json.load(open(ROOT / "configs" / "top_skills.json", encoding="utf-8"))
env = dict(os.environ)
env["SPARK_HOME"] = "/usr/local/spark"
env["PYTHONPATH"] = "/usr/local/spark/python:/usr/local/spark/python/lib/py4j-0.10.9-src.zip:" + env.get("PYTHONPATH", "")
env["PYTHONIOENCODING"] = "utf-8"


def say(msg):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), msg, flush=True)


def rows():
    if not OUT.exists():
        return []
    with open(OUT, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def done_skills():
    seen = {}
    for r in rows():
        seen.setdefault(r["skill"], set()).add(r["model"])
    return {s for s, m in seen.items() if len(m) >= 3}


say(f"sweep started: {len(skills)} skills, chunks of {CHUNK}, output {OUT_REL}, pipeline {PIPE}")
stalls = 0
attempt = 0
while True:
    done = done_skills()
    pending = [i for i, s in enumerate(skills) if s not in done]
    if not pending:
        break
    start = pending[0]
    attempt += 1
    log = LOGS / f"chunk_{start:02d}_try{attempt:02d}.log"
    cmd = ["python3", "src/train_spark_native.py", "--data", DATA, "--k8s", "--output", OUT_REL, "--models-dir", MODELS,
           "--pipeline-path", PIPE, "--skill-start", str(start), "--skill-count", str(CHUNK)]
    say(f"START chunk skills[{start}:{start + CHUNK}] ({len(done)}/{len(skills)} skills done) -> {log.name}")
    t0 = time.time()
    with open(log, "wb") as lf:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = proc.wait(timeout=CHUNK_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            rc = -999
            say("WATCHDOG killed the chunk after 2.5 h")
    say(f"END chunk skills[{start}:{start + CHUNK}] rc={rc} in {(time.time() - t0) / 60:.1f} min; skills done now: {len(done_skills())}/{len(skills)}")
    if done_skills() == done:
        stalls += 1
        say(f"no progress in this chunk ({stalls}/{MAX_STALLS})")
        if stalls >= MAX_STALLS:
            say("GIVING UP: four chunks in a row made no progress")
            break
    else:
        stalls = 0

# one row per (skill, model), in skill order
allrows = rows()
if allrows:
    last = {}
    for r in allrows:
        last[(r["skill"], r["model"])] = r
    order = {s: i for i, s in enumerate(skills)}
    out = sorted(last.values(), key=lambda r: (order.get(r["skill"], 999), r["model"]))
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(allrows[0].keys()))
        w.writeheader()
        w.writerows(out)
    say(f"deduplicated: {len(out)} rows, {len({r['skill'] for r in out})} skills")
    for m in sorted({r["model"] for r in out}):
        v = [float(r["pr_auc"]) for r in out if r["model"] == m]
        say(f"  {m}: macro PR-AUC {sum(v) / len(v):.4f} over {len(v)} skills")
say("=== ALL DONE ===")
