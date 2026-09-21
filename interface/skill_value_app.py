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

If data/hh_vacancies_cleaned_fixed_salary.parquet is present it also shows
simple statistics and recent real postings for the selected skills; without
it that part is skipped.

Usage: streamlit run interface/skill_value_app.py --server.port 8502
"""
import ast
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
SUFFIX = "_v3"
MIN_SUPPORT_POSTINGS = 149  # FPGrowth minSupport 0.01 x 14,860 usable postings, rounded up
DATA_PATH = PROJECT_ROOT / "data" / "hh_vacancies_cleaned_fixed_salary.parquet"
EXPERIENCE_LABELS = {"noExperience": "No experience", "between1And3": "1-3 years",
                     "between3And6": "3-6 years", "moreThan6": "6+ years"}
CITY_LABELS = {"Москва": "Moscow", "Санкт-Петербург": "Saint Petersburg", "Other": "Other cities"}


@st.cache_data(show_spinner="Loading skill value data...")
def load_data():
    combos = pd.read_csv(RESULTS_DIR / f"skill_combo_value{SUFFIX}.csv")
    singles = pd.read_csv(RESULTS_DIR / f"skill_single_value{SUFFIX}.csv")
    rules = pd.read_csv(RESULTS_DIR / f"skill_combo_rules{SUFFIX}.csv")
    ci = pd.read_csv(RESULTS_DIR / f"skill_combo_bootstrap_ci{SUFFIX}.csv").set_index("name")

    combos = combos.dropna(subset=["adjusted_premium"]).copy()
    singles = singles.dropna(subset=["adjusted_premium"]).copy()
    singles = singles.sort_values("adjusted_premium", ascending=False).reset_index(drop=True)

    combos["skills"] = combos["combination"].apply(lambda s: [x.strip() for x in s.split(" + ")])
    rules["antecedent_list"] = rules["antecedent"].apply(ast.literal_eval)
    rules["consequent_list"] = rules["consequent"].apply(ast.literal_eval)

    return combos, singles, rules, ci


@st.cache_resource(show_spinner="Loading postings...")
def load_postings(data_mtime):
    """The usable postings behind the premiums, plus title, employer, city, posted salary and link.
    None if the file isn't the one the saved results came from. data_mtime only keys the cache."""
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    from skill_combo_value import load_usable_postings

    usable = load_usable_postings(str(DATA_PATH), require_ceiling=False)
    saved = pd.read_csv(RESULTS_DIR / f"skill_single_value{SUFFIX}.csv").set_index("skill")["n_with"]
    counts = pd.Series([s for skills in usable["skills"] for s in skills]).value_counts()
    if not counts.reindex(saved.index).eq(saved).all():
        return None

    cols = ["id", "name", "employer", "area", "salary", "alternate_url", "published_at"]
    details = pd.read_parquet(DATA_PATH, columns=cols)
    details["employer"] = details["employer"].apply(lambda e: e.get("name") if e else None)
    details["city"] = details["area"].apply(lambda a: a.get("name") if a else None)
    for key in ("from", "to", "gross"):
        details[f"salary_{key}"] = details["salary"].apply(lambda s: s.get(key) if s else None)
    details["published_at"] = pd.to_datetime(details["published_at"], utc=True)
    return usable.merge(details.drop(columns=["area", "salary"]), on="id")


def format_rub(value):
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:,.0f} RUB/mo".replace(",", " ")


def format_ci(low, high):
    low, high = (f"{v:+,.0f}".replace(",", " ") for v in (low, high))
    return f"[{low}, {high}] RUB/mo"


def format_number(value):
    return f"{value:,.0f}".replace(",", " ")


def md_escape(text):
    return re.sub(r"([\\`*_\[\]<>$])", r"\\\1", text or "")


def posted_salary(p):
    if pd.notna(p["salary_from"]) and pd.notna(p["salary_to"]):
        text = f"{format_number(p['salary_from'])}-{format_number(p['salary_to'])} RUB"
    elif pd.notna(p["salary_from"]):
        text = f"from {format_number(p['salary_from'])} RUB"
    else:
        text = f"up to {format_number(p['salary_to'])} RUB"
    tax = {True: "before tax", False: "after tax"}.get(p["salary_gross"])
    return f"{text}, {tax}" if tax else text


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


def show_no_recommendation(known_skills, combos, singles, ci):
    without_pairs = [k for k in known_skills if not combos["skills"].apply(lambda s: k in s).any()]
    for k in without_pairs:
        own = singles[singles["skill"] == k].iloc[0]
        n = int(own["n_with"])
        st.write(
            f"No recommendation for **{k}**: it appears in {n} postings, but no other skill appears together "
            f"with it in at least {MIN_SUPPORT_POSTINGS} postings (the FPGrowth support threshold), so no "
            f"frequent pairs exist for it."
        )
        st.write(
            f"**{k}** on its own: **{format_rub(own['adjusted_premium'])}**, "
            f"95% CI {format_ci(ci.loc[k, 'ci_low_95'], ci.loc[k, 'ci_high_95'])}, based on {n} postings."
        )
    if len(without_pairs) < len(known_skills):
        rest = "the rest of your selection" if without_pairs else "your selection"
        st.write(f"Every skill that pairs frequently with {rest} is already selected, so there is nothing left to suggest.")


def profile_rows(df):
    q1, median, q3 = df["salary_net"].quantile([0.25, 0.5, 0.75])
    rows = {
        "Median starting salary (net)": f"{format_number(median)} RUB/mo",
        "Middle half of salaries": f"{format_number(q1)} to {format_number(q3)} RUB/mo",
    }
    for key, label in CITY_LABELS.items():
        rows[f"City: {label}"] = f"{(df['city_bucket'] == key).mean():.0%}"
    for key, label in EXPERIENCE_LABELS.items():
        rows[f"Experience: {label}"] = f"{(df['experience_bucket'] == key).mean():.0%}"
    rows["Only a starting salary stated (no ceiling)"] = f"{(~df['has_ceiling'].astype(bool)).mean():.0%}"
    return rows


def show_posting_profile(selected):
    postings = load_postings(DATA_PATH.stat().st_mtime) if DATA_PATH.exists() else None
    if postings is None:
        st.caption(
            "Real postings and statistics need data/hh_vacancies_cleaned_fixed_salary.parquet from the same "
            "run as the saved results, and that file isn't included in the repo."
        )
        return

    chosen = set(selected)
    matched = postings[postings["skills_set"].apply(chosen.issubset)]
    label = " + ".join(selected)
    if matched.empty:
        st.write(f"None of the {len(postings):,} usable postings list {label} together.")
        return

    n = len(matched)
    st.write(f"**{n:,}** of the {len(postings):,} usable postings list {label} ({n / len(postings):.1%}).")
    st.table(pd.DataFrame({"These postings": profile_rows(matched), "All usable postings": profile_rows(postings)}))
    st.caption(
        "Net starting salary (salary.from), not adjusted for anything. These postings also differ in seniority "
        "and city, which is why the adjusted premium above is the fairer comparison."
    )

    alongside = Counter(s for skills in matched["skills"] for s in skills if s not in chosen)
    if alongside:
        top = " · ".join(f"{s} ({c / n:.0%})" for s, c in alongside.most_common(8))
        st.write(f"**Most often listed alongside:** {top}")

    recent = matched.sort_values(["published_at", "id"], ascending=False).drop_duplicates("employer").head(5)
    lines = [
        f"- [{md_escape(p['name'])}]({p['alternate_url']}) — {p['city']} · {posted_salary(p)} · "
        f"{EXPERIENCE_LABELS.get(p['experience_bucket'], p['experience_bucket'])}"
        for _, p in recent.iterrows()
    ]
    st.write(f"**Recent postings that list {'them' if len(selected) > 1 else 'it'}:**")
    st.markdown("\n".join(lines))
    first, last = postings["published_at"].min(), postings["published_at"].max()
    st.caption(f"Postings published {first:%b %Y} to {last:%b %Y}; some links may lead to closed vacancies.")


combos, singles, rules, ci = load_data()

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
        show_no_recommendation(known_skills, combos, singles, ci)
        st.subheader("What the postings look like")
        show_posting_profile(known_skills)
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

    if not recommendations.empty and DATA_PATH.exists():
        with st.expander("What the postings look like"):
            show_posting_profile(known_skills)
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
