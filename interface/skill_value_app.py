"""
Streamlit demo for the Skill Combination Value Calculator (see
src/skill_combo_value.py): given skills you know, ranks what to learn
next by salary premium, from real hh.ru postings priced via Association
Rule Mining (Spark FPGrowth) plus a confound-adjusted salary comparison
(experience, city, and whether a salary ceiling was disclosed), not a
raw average.

Reads pre-computed tables from results/*.csv -- src/skill_combo_value.py
already did the Spark/FPGrowth work offline, so this starts instantly
with no Spark session or model loading at request time. Currently on v3
(14,860 usable postings). Scope and caveats (starting-salary vs. total
compensation, sample sizes, what's controlled for) are shown in the app
itself, not just here.

Usage: streamlit run interface/skill_value_app.py --server.port 8502
"""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
SUFFIX = "_v3"


@st.cache_data(show_spinner="Loading skill value data...")
def load_data():
    combos = pd.read_csv(RESULTS_DIR / f"skill_combo_value{SUFFIX}.csv")
    singles = pd.read_csv(RESULTS_DIR / f"skill_single_value{SUFFIX}.csv")
    rules = pd.read_csv(RESULTS_DIR / f"skill_combo_rules{SUFFIX}.csv")

    combos = combos.dropna(subset=["adjusted_premium"]).copy()
    singles = singles.dropna(subset=["adjusted_premium"]).copy()
    singles = singles.sort_values("adjusted_premium", ascending=False).reset_index(drop=True)

    combos["skills"] = combos["combination"].apply(lambda s: [x.strip() for x in s.split(" + ")])
    rules["antecedent_list"] = rules["antecedent"].apply(ast.literal_eval)
    rules["consequent_list"] = rules["consequent"].apply(ast.literal_eval)

    return combos, singles, rules


def format_rub(value):
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:,.0f} RUB/mo".replace(",", " ")


def find_recommendations(known_skills, combos):
    """Returns (dataframe, mode). mode is "exact" (a real priced combination
    covering all known skills plus one more), "pairwise" (no exact match --
    each candidate's value is a weighted average of its pairwise premium
    with each known skill individually), or "none" (no data at all)."""
    known_set = set(known_skills)
    n_known = len(known_set)

    exact_mask = combos["skills"].apply(lambda s: len(s) == n_known + 1 and known_set.issubset(set(s)))
    exact = combos[exact_mask].copy()
    if not exact.empty:
        exact["suggested_skill"] = exact["skills"].apply(lambda s: [x for x in s if x not in known_set][0])
        return exact.sort_values("adjusted_premium", ascending=False), "exact"

    pairwise_rows = []
    for k in known_skills:
        k_mask = combos["skills"].apply(lambda s: k in s and len(s) == 2)
        for _, row in combos[k_mask].iterrows():
            other = [x for x in row["skills"] if x != k][0]
            if other in known_set:
                continue
            pairwise_rows.append({
                "known_skill": k, "suggested_skill": other,
                "adjusted_premium": row["adjusted_premium"], "n_with": row["n_with"],
            })

    if not pairwise_rows:
        return pd.DataFrame(), "none"

    pairwise_df = pd.DataFrame(pairwise_rows)
    agg_rows = []
    for candidate, group in pairwise_df.groupby("suggested_skill"):
        agg_rows.append({
            "suggested_skill": candidate,
            "adjusted_premium": np.average(group["adjusted_premium"], weights=group["n_with"]),
            "n_with": int(group["n_with"].sum()),
            "based_on": ", ".join(sorted(group["known_skill"].unique())),
        })
    return pd.DataFrame(agg_rows).sort_values("adjusted_premium", ascending=False), "pairwise"


combos, singles, rules = load_data()

st.set_page_config(page_title="Skill Value Calculator", page_icon="\U0001F4B0")
st.title("Skill Combination Value Calculator")
st.caption(
    "If you know some skills, what should you learn next to earn more? Built from real hh.ru postings "
    "using Association Rule Mining (Spark's FPGrowth) to find skill combinations, then a salary "
    "comparison that controls for experience level, city, and whether a salary ceiling was even "
    "disclosed -- not a raw average -- to price them fairly."
)
st.info(
    "Numbers below are STARTING-salary lift, not total compensation -- based on the minimum/floor "
    "figure postings give, since about half of postings with any salary info only state a floor with "
    "no ceiling. Based on 14,860 postings that both tag skills and disclose at least a starting salary. "
    "Most premiums below rest on 150-470 matching postings each -- real signal, not precise numbers. "
    "Adjusted for experience level, city, and ceiling-disclosure (postings giving only a floor pay "
    "~20% more on average than ones giving a full range), but not for every possible factor -- "
    "employer, for instance, isn't controlled for. Pick more than one skill and, if there's no exact "
    "match for that combination, recommendations fall back to an estimate -- labeled as such, not "
    "shown as a direct measurement.",
    icon="ℹ️",
)

all_skills = singles["skill"].tolist()
known_skills = st.multiselect("Skills you already know", all_skills, placeholder="Start typing a skill...")

if known_skills:
    st.subheader(", ".join(known_skills))

    own_rows = singles[singles["skill"].isin(known_skills)].sort_values("adjusted_premium", ascending=False)
    for _, row in own_rows.iterrows():
        rank = int(singles.index[singles["skill"] == row["skill"]][0]) + 1
        st.write(
            f"**{row['skill']}** alone: **{format_rub(row['adjusted_premium'])}** "
            f"(rank #{rank} of {len(singles)}, based on {int(row['n_with'])} postings)"
        )

    recommendations, mode = find_recommendations(known_skills, combos)

    st.subheader("What to learn next, ranked by pay")
    if mode == "exact":
        st.caption("Based on real postings with exactly this combination plus one more skill.")
    elif mode == "pairwise":
        st.caption(
            "No postings in our data have exactly this combination, so each candidate below is "
            "estimated by averaging its pairwise value with your known skills individually -- an "
            "approximation, not a direct measurement."
        )

    if recommendations.empty:
        st.write("No data-backed recommendation available for this combination of skills yet.")
    else:
        for _, r in recommendations.head(10).iterrows():
            if mode == "exact":
                synergy_note = ""
                if pd.notna(r["synergy"]) and r["synergy"] > 0:
                    synergy_note = f"  ({format_rub(r['synergy'])} more than either skill alone)"
                st.write(
                    f"**+ {r['suggested_skill']}** → combined **{format_rub(r['adjusted_premium'])}**"
                    f"{synergy_note}  *(based on {int(r['n_with'])} postings)*"
                )
            else:
                st.write(
                    f"**+ {r['suggested_skill']}** → estimated **{format_rub(r['adjusted_premium'])}** "
                    f"*(from pairing with {r['based_on']}, {int(r['n_with'])} postings total)*"
                )

    known_set = set(known_skills)
    rule_mask = rules["antecedent_list"].apply(lambda a: set(a).issubset(known_set))
    skill_rules = rules[rule_mask].copy()
    # guard each step against an already-empty frame -- pandas' .apply() on a
    # zero-row Series can return something that's lost the original columns,
    # which crashed sort_values("lift") below with a KeyError for skills
    # (like C#) that have no frequent pairs at all
    if not skill_rules.empty:
        consequent_mask = skill_rules["consequent_list"].apply(lambda c: not set(c).issubset(known_set))
        skill_rules = skill_rules[consequent_mask]
    if not skill_rules.empty:
        skill_rules = skill_rules.sort_values("lift", ascending=False)

    st.subheader("What tends to be required alongside them")
    st.caption("Co-occurrence, not salary -- how often postings needing what you know also need another skill.")
    if skill_rules.empty:
        st.write("No strong co-occurrence pattern met our data threshold for this combination.")
    else:
        for _, r in skill_rules.head(10).iterrows():
            antecedent = ", ".join(r["antecedent_list"])
            consequent = ", ".join(r["consequent_list"])
            st.write(
                f"**{consequent}** — {r['confidence']:.0%} of postings needing {antecedent} "
                f"also need this (lift {r['lift']:.1f})"
            )
else:
    st.subheader("Top-paying skill combinations right now")
    top = combos.sort_values("adjusted_premium", ascending=False).head(10)
    for _, r in top.iterrows():
        st.write(f"**{r['combination']}** → {format_rub(r['adjusted_premium'])}  *(based on {int(r['n_with'])} postings)*")

with st.expander("All skills ranked by salary premium"):
    st.dataframe(
        singles.assign(adjusted_premium=singles["adjusted_premium"].map(format_rub))[
            ["skill", "n_with", "adjusted_premium"]
        ],
        width="stretch", hide_index=True,
    )

with st.expander("All skill combinations ranked by salary premium"):
    st.dataframe(
        combos.sort_values("adjusted_premium", ascending=False).assign(
            adjusted_premium=lambda d: d["adjusted_premium"].map(format_rub),
            raw_premium=lambda d: d["raw_premium"].map(format_rub),
        )[["combination", "n_with", "raw_premium", "adjusted_premium", "strata_used"]],
        width="stretch", hide_index=True,
    )
