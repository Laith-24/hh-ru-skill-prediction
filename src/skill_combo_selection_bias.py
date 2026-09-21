"""
Selection bias check for the Skill Combination Value Calculator: the
"usable" population (tagged skills + disclosed RUB salary) is a small
fraction of the full dataset, and every premium/regression finding here
implicitly assumes that subset represents the broader market -- never
checked before this. Compares the usable subset against the full
population on experience and city (both 0% missing, so exact), plus a
skill-mix comparison isolating just the salary-disclosure effect.

Usage: python src/skill_combo_selection_bias.py [--data PATH] [--suffix _v2]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import chisquare

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from skill_combo_value import DATA_PATH, load_usable_postings, _get

MAJOR_CITIES = ["Москва", "Санкт-Петербург"]
MIN_SKILL_COUNT = 45


def chi_square_check(name, full_counts, usable_counts):
    categories = sorted(set(full_counts.index) | set(usable_counts.index))
    full_total = full_counts.sum()
    usable_total = usable_counts.sum()
    full_pct = (full_counts.reindex(categories, fill_value=0) / full_total * 100)
    usable_pct = (usable_counts.reindex(categories, fill_value=0) / usable_total * 100)

    table = pd.DataFrame({"full_pct": full_pct.round(1), "usable_pct": usable_pct.round(1)})
    table["pct_point_diff"] = (table["usable_pct"] - table["full_pct"]).round(1)
    print(f"\n=== {name} ===")
    print(table.to_string())

    expected = (full_pct / 100) * usable_total
    observed = usable_counts.reindex(categories, fill_value=0)
    mask = expected > 0
    stat, p = chisquare(observed[mask], f_exp=expected[mask])
    verdict = "DIFFERS significantly" if p < 0.05 else "no significant difference"
    print(f"Chi-square: stat={stat:.1f}, p={p:.2e} -- {verdict} from the full population (alpha=0.05)")
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--suffix", default="")
    parser.add_argument("--require-ceiling", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.data, columns=["key_skills", "salary", "area", "experience"])
    print(f"Full dataset: {len(df)} rows")

    full_exp = df["experience"].apply(lambda x: _get(x, "id")).value_counts()
    full_city_raw = df["area"].apply(lambda x: _get(x, "name"))
    full_city = full_city_raw.apply(lambda c: c if c in MAJOR_CITIES else "Other").value_counts()

    df["skills"] = df["key_skills"].apply(
        lambda lst: sorted({s.get("name", "").strip() for s in lst if s and s.get("name", "").strip()})
        if lst is not None else []
    )
    tagged = df[df["skills"].apply(lambda s: len(s) >= 2)]
    print(f"Postings with >=2 tagged skills: {len(tagged)}")

    usable = load_usable_postings(args.data, args.require_ceiling)
    print(f"Usable (tagged skills + disclosed RUR salary): {len(usable)} ({len(usable) / len(df):.1%} of full dataset)")

    chi_square_check("Experience: full dataset vs. usable subset", full_exp, usable["experience_bucket"].value_counts())
    chi_square_check("City: full dataset vs. usable subset", full_city, usable["city_bucket"].value_counts())

    skill_counts_tagged = pd.Series([s for skills in tagged["skills"] for s in skills]).value_counts()
    skill_counts_usable = pd.Series([s for skills in usable["skills"] for s in skills]).value_counts()

    common_skills = skill_counts_tagged[skill_counts_tagged >= MIN_SKILL_COUNT].index
    rate_tagged = skill_counts_tagged[common_skills] / len(tagged) * 100
    rate_usable = skill_counts_usable.reindex(common_skills, fill_value=0) / len(usable) * 100

    skill_compare = pd.DataFrame({"tagged_pct": rate_tagged.round(2), "usable_pct": rate_usable.round(2)})
    skill_compare["ratio"] = (skill_compare["usable_pct"] / skill_compare["tagged_pct"]).round(2)
    skill_compare = skill_compare.sort_values("ratio", ascending=False)

    print(f"\n=== Skill mix: postings with tagged skills (n={len(tagged)}) vs. usable subset (n={len(usable)}) ===")
    print("Ratio > 1: over-represented among salary-disclosing postings. Ratio < 1: under-represented.")
    print(f"\nTop 10 most OVER-represented in the usable subset:\n{skill_compare.head(10).to_string()}")
    print(f"\nTop 10 most UNDER-represented in the usable subset:\n{skill_compare.tail(10).to_string()}")

    results_dir = ROOT / "results"
    skill_compare.to_csv(results_dir / f"selection_bias_skill_mix{args.suffix}.csv")
    print(f"\nSaved to results/selection_bias_skill_mix{args.suffix}.csv")


if __name__ == "__main__":
    main()
