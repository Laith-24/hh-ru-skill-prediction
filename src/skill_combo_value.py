"""
Skill Combination Value Calculator -- the pivot: instead of predicting
which skills a posting needs, finds which skill COMBINATIONS carry a real
salary premium, using Association Rule Mining (Spark FPGrowth) plus a
confound-adjusted comparison (stratified by experience/city/ceiling-
disclosure) so the premium isn't just measuring seniority and city.

Salary restricted to RUR only (currency is mixed even among complete-
salary rows); gross salaries converted to net via Russia's flat 13% tax
rate (an approximation, not exact brackets).

Deliberately mines the full skill vocabulary, not just the cached top-20
list -- FPGrowth doesn't need a pre-chosen label set.

--no-require-ceiling (v3): targets salary.from alone instead of requiring
a full salary range, since ~45% of salaried postings give only a floor.
Nearly doubles the usable sample; report these premiums as "starting-
salary lift," not total compensation -- a narrower claim than the v1/v2
numbers, not an upgrade to them.

Usage: python src/skill_combo_value.py [--data PATH] [--suffix _v2]
    [--no-require-ceiling]
--suffix is appended to every output filename, so a second data file's
results don't overwrite the first run's.
"""
import argparse
import logging
import tempfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyspark.ml.fpm import FPGrowth
from pyspark.sql import SparkSession

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "vacancies.parquet"
SPARK_TMP_DIR = str(ROOT / "spark-tmp")

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.4
MIN_STRATUM_N = 5       # each side of a with/without comparison needs at least this many rows in a stratum to count
MIN_STRATA_USED = 2     # need at least this many usable strata to trust an adjusted premium
NET_MULTIPLIER_IF_GROSS = 0.87  # Russia's flat 13% personal income tax, approximate

MAJOR_CITIES = ["Москва", "Санкт-Петербург"]


def _get(struct, key):
    if struct is None:
        return None
    try:
        return struct.get(key)
    except AttributeError:
        return struct[key] if key in struct else None


def load_usable_postings(data_path=DATA_PATH, require_ceiling=True) -> pd.DataFrame:
    df = pd.read_parquet(
        data_path,
        columns=["id", "name", "employer", "description", "key_skills", "salary", "area", "experience"],
    )
    total = len(df)

    # Same dedup key as data_pipeline._dedupe_postings (name, employer, description) --
    # done here in pandas since this script never needs a Spark session for anything
    # except the FPGrowth call itself.
    df["employer_name"] = df["employer"].apply(lambda x: _get(x, "name"))
    before = len(df)
    df = df.drop_duplicates(subset=["name", "employer_name", "description"]).reset_index(drop=True)
    logger.info(f"Deduplicated postings: {before} -> {len(df)} rows")

    df["skills"] = df["key_skills"].apply(
        lambda lst: sorted({s.get("name", "").strip() for s in lst if s and s.get("name", "").strip()})
        if lst is not None else []
    )
    has_skills = df["skills"].apply(lambda s: len(s) >= 2)  # need >=2 to ever form a "combination"

    currency = df["salary"].apply(lambda x: _get(x, "currency"))
    frm = df["salary"].apply(lambda x: _get(x, "from"))
    to = df["salary"].apply(lambda x: _get(x, "to"))
    gross = df["salary"].apply(lambda x: _get(x, "gross"))
    has_full_salary = frm.notna() & to.notna()
    is_rur = currency == "RUR"

    if require_ceiling:
        usable_mask = has_skills & has_full_salary & is_rur
        mode_desc = "also with complete salary (from+to)"
    else:
        usable_mask = has_skills & frm.notna() & is_rur
        mode_desc = "also with salary.from present (to optional -- v3 mode)"
    logger.info(
        f"Rows: {total} total -> {has_skills.sum()} with >=2 tagged skills -> "
        f"{usable_mask.sum()} {mode_desc} and RUR-only (final usable set)"
    )

    usable = df.loc[usable_mask, ["id", "skills"]].copy()
    is_gross = gross[usable_mask].fillna(False)

    if require_ceiling:
        target = (frm[usable_mask] + to[usable_mask]) / 2
    else:
        # v3: target is salary.from ALONE for every usable row (not just the
        # ceiling-less ones) so the target definition stays consistent across
        # the whole population -- has_ceiling below is a separate feature,
        # not a switch in what's being predicted. See module docstring for
        # why this isn't redundant with experience/city.
        target = frm[usable_mask]
    usable["salary_net"] = np.where(is_gross, target * NET_MULTIPLIER_IF_GROSS, target)
    usable["has_ceiling"] = to[usable_mask].notna().values

    area_name = df["area"].apply(lambda x: _get(x, "name"))
    usable["city_bucket"] = area_name[usable_mask].apply(lambda c: c if c in MAJOR_CITIES else "Other")

    exp_id = df["experience"].apply(lambda x: _get(x, "id"))
    usable["experience_bucket"] = exp_id[usable_mask]

    usable["skills_set"] = usable["skills"].apply(set)
    return usable.reset_index(drop=True)


def run_fpgrowth(usable: pd.DataFrame):
    spark = (
        SparkSession.builder.appName("hh-ru-skill-combo-value")
        .master("local[2]")
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        .config("spark.local.dir", SPARK_TMP_DIR)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    items_pdf = usable[["id", "skills"]].rename(columns={"skills": "items"})
    table = pa.Table.from_pandas(items_pdf, preserve_index=False)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = str(Path(tmp_dir) / "items.parquet")
        pq.write_table(table, tmp_path)
        items_df = spark.read.parquet(tmp_path)
        items_df = items_df.repartition(4).cache()
        items_df.count()  # materialize before FPGrowth to keep timing/logs clean

        fpgrowth = FPGrowth(itemsCol="items", minSupport=MIN_SUPPORT, minConfidence=MIN_CONFIDENCE)
        model = fpgrowth.fit(items_df)

        freq_itemsets = model.freqItemsets.toPandas()
        rules = model.associationRules.toPandas()

    spark.stop()
    logger.info(f"FPGrowth found {len(freq_itemsets)} frequent itemsets, {len(rules)} association rules")
    return freq_itemsets, rules


def compute_premium(usable: pd.DataFrame, itemset: frozenset):
    has_mask = usable["skills_set"].apply(lambda s: itemset.issubset(s))
    n_with = int(has_mask.sum())
    n_without = int((~has_mask).sum())
    if n_with == 0 or n_without == 0:
        return None

    raw_premium = usable.loc[has_mask, "salary_net"].mean() - usable.loc[~has_mask, "salary_net"].mean()

    weighted_sum, weight_total, strata_used = 0.0, 0, 0
    for _, group in usable.groupby(["experience_bucket", "city_bucket", "has_ceiling"]):
        g_mask = has_mask.loc[group.index]
        g_has, g_not = group[g_mask], group[~g_mask]
        if len(g_has) >= MIN_STRATUM_N and len(g_not) >= MIN_STRATUM_N:
            diff = g_has["salary_net"].mean() - g_not["salary_net"].mean()
            weighted_sum += diff * len(g_has)
            weight_total += len(g_has)
            strata_used += 1

    adjusted_premium = weighted_sum / weight_total if strata_used >= MIN_STRATA_USED and weight_total > 0 else None
    return {
        "n_with": n_with,
        "raw_premium": round(raw_premium, 0),
        "adjusted_premium": round(adjusted_premium, 0) if adjusted_premium is not None else None,
        "strata_used": strata_used,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--suffix", default="", help="Appended to every output filename")
    parser.add_argument("--require-ceiling", action=argparse.BooleanOptionalAction, default=True,
                         help="Default True (v1/v2 behavior: require salary.to, target=avg(from,to)). "
                              "--no-require-ceiling for v3: target=salary.from alone, has_ceiling as a feature.")
    args = parser.parse_args()

    usable = load_usable_postings(args.data, args.require_ceiling)
    if len(usable) < 200:
        logger.warning("Usable sample is quite small -- results below will be noisy, treat as exploratory.")

    freq_itemsets, rules = run_fpgrowth(usable)

    freq_itemsets["size"] = freq_itemsets["items"].apply(len)
    singles = freq_itemsets[freq_itemsets["size"] == 1].copy()
    combos = freq_itemsets[freq_itemsets["size"] >= 2].copy()
    logger.info(f"{len(singles)} frequent single skills, {len(combos)} frequent combinations (size >= 2)")

    single_premiums = {}
    for _, row in singles.iterrows():
        skill = row["items"][0]
        result = compute_premium(usable, frozenset(row["items"]))
        if result:
            single_premiums[skill] = result["adjusted_premium"]

    combo_rows = []
    for _, row in combos.iterrows():
        itemset = frozenset(row["items"])
        result = compute_premium(usable, itemset)
        if result is None:
            continue
        best_individual = max(
            (single_premiums.get(s) for s in itemset if single_premiums.get(s) is not None),
            default=None,
        )
        synergy = (
            result["adjusted_premium"] - best_individual
            if result["adjusted_premium"] is not None and best_individual is not None
            else None
        )
        combo_rows.append({
            "combination": " + ".join(sorted(row["items"])),
            "size": row["size"],
            "freq_count": row["freq"],
            **result,
            "best_individual_premium": best_individual,
            "synergy": round(synergy, 0) if synergy is not None else None,
        })

    combo_df = pd.DataFrame(combo_rows)
    single_df = pd.DataFrame([
        {"skill": s, "n_with": singles.loc[singles["items"].apply(lambda x: x[0] == s), "freq"].iloc[0],
         "adjusted_premium": p}
        for s, p in single_premiums.items()
    ]).sort_values("adjusted_premium", ascending=False, na_position="last")

    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    combo_df.sort_values("adjusted_premium", ascending=False, na_position="last").to_csv(
        results_dir / f"skill_combo_value{args.suffix}.csv", index=False
    )
    single_df.to_csv(results_dir / f"skill_single_value{args.suffix}.csv", index=False)
    rules.to_csv(results_dir / f"skill_combo_rules{args.suffix}.csv", index=False)

    pd.set_option("display.width", 140)
    print(f"\nUsable postings: {len(usable)}")
    print(f"\n=== Top single skills by adjusted salary premium (vs. not having it) ===")
    print(single_df.head(15).to_string(index=False))

    priced = combo_df[combo_df["adjusted_premium"].notna()].sort_values("adjusted_premium", ascending=False)
    print(f"\n=== Top combinations by adjusted salary premium ({len(priced)}/{len(combo_df)} could be priced) ===")
    print(priced.head(20).to_string(index=False))

    gap = combo_df.dropna(subset=["adjusted_premium"]).copy()
    gap["gap"] = (gap["raw_premium"] - gap["adjusted_premium"]).abs()
    print(f"\n=== Where the confound adjustment changed the answer the most ===")
    print(gap.sort_values("gap", ascending=False).head(10)[
        ["combination", "n_with", "raw_premium", "adjusted_premium", "gap"]
    ].to_string(index=False))

    synergy_df = combo_df.dropna(subset=["synergy"]).sort_values("synergy", ascending=False)
    print(f"\n=== Top synergy: combo pays more than its best individual skill alone ===")
    print(synergy_df.head(10)[
        ["combination", "n_with", "adjusted_premium", "best_individual_premium", "synergy"]
    ].to_string(index=False))

    print(f"\nTop association rules (if you know X, you likely also need Y):")
    rules_sorted = rules.sort_values("lift", ascending=False)
    print(rules_sorted.head(15).to_string(index=False))

    print(f"\nFull results saved to results/skill_combo_value{args.suffix}.csv, "
          f"skill_single_value{args.suffix}.csv, skill_combo_rules{args.suffix}.csv")


if __name__ == "__main__":
    main()
