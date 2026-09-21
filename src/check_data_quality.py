"""
Corrected data quality check for the hh.ru vacancy dataset.

An earlier version of this check reported 0% missing for salary/skills
and 0 duplicates -- wrong, because it checked whether the CONTAINER field
was present, not whether the value inside was populated (a posting with
no salary still has `salary: {from: null, to: null}`, a non-null dict),
and because id-based dedup misses hh.ru reposts (same listing, new id) --
this project's own pipeline dedupes on (name, employer, description)
instead, which actually catches them.

Usage: python check_data_quality.py [path-to-file.json-or-.parquet]
Defaults to data/output/combined_vacancies_with_salary.json if no path given.
"""
import sys
from pathlib import Path

import pandas as pd

DEFAULT_PATH = "data/output/combined_vacancies_with_salary.json"


def _get(struct, key):
    if struct is None:
        return None
    try:
        return struct.get(key)
    except AttributeError:
        return struct[key] if key in struct else None


def load(path):
    path = Path(path)
    if path.suffix == ".json":
        return pd.read_json(path)
    return pd.read_parquet(path)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH

    print("STEP 1: Loading Data")
    print("-" * 60)
    print(f"  Loading: {path}")
    df = load(path)
    print(f"    -> {len(df):,} vacancies")
    print(f"\nTOTAL VACANCIES LOADED: {len(df):,}")

    print("\n" + "=" * 70)
    print("DATA QUALITY ASSESSMENT")
    print("=" * 70)

    # Vacancy ID / Job Title -- plain scalar fields, a top-level null check
    # is actually correct here (verified: genuinely 0% missing).
    id_missing = int(df["id"].isnull().sum())
    print(f"  Vacancy ID: {id_missing} missing ({id_missing / len(df):.1%})")

    title_missing = int(df["name"].isnull().sum())
    print(f"  Job Title: {title_missing} missing ({title_missing / len(df):.1%})")

    # Employer / Location -- struct fields. Check the value INSIDE
    # (employer.name / area.name), not whether the struct exists.
    employer_name = df["employer"].apply(lambda x: _get(x, "name"))
    employer_missing = int(employer_name.isnull().sum())
    print(f"  Employer: {employer_missing} missing ({employer_missing / len(df):.1%})")

    area_name = df["area"].apply(lambda x: _get(x, "name"))
    area_missing = int(area_name.isnull().sum())
    print(f"  Location: {area_missing} missing ({area_missing / len(df):.1%})")

    # Salary Info -- the field the original check got most wrong. Usable
    # salary needs BOTH from and to populated; check those specifically,
    # not the outer struct (which is present on every row regardless).
    salary_from = df["salary"].apply(lambda x: _get(x, "from"))
    salary_to = df["salary"].apply(lambda x: _get(x, "to"))
    salary_missing = int((salary_from.isnull() | salary_to.isnull()).sum())
    print(f"  Salary Info: {salary_missing} missing ({salary_missing / len(df):.1%})")
    print(f"    (from missing: {salary_from.isnull().mean():.1%}, to missing: {salary_to.isnull().mean():.1%})")

    # Skills -- an empty list is zero tagged skills, not "data present."
    skills_missing = int(df["key_skills"].apply(lambda x: x is None or len(x) == 0).sum())
    print(f"  Skills: {skills_missing} missing ({skills_missing / len(df):.1%})")

    print("\nDUPLICATE RECORDS:")
    id_dupes = int(df["id"].duplicated().sum())
    print(f"  Duplicate IDs: {id_dupes} ({id_dupes / len(df):.2%}) -- not meaningful, reposts get a new id each time")

    content_dupes = int(
        df.assign(employer_name=employer_name)
        .duplicated(subset=["name", "employer_name", "description"])
        .sum()
    )
    print(f"  Duplicate postings (title + employer + description): {content_dupes} ({content_dupes / len(df):.2%})")

    # A single blended score can hide exactly the gap that mattered here
    # (four always-complete fields would average out two badly-incomplete
    # ones into something that still looks fine) -- shown for continuity
    # with the original report's format, but read the breakdown above
    # first, not this number alone.
    fields_missing = id_missing + title_missing + employer_missing + area_missing + salary_missing + skills_missing
    completeness = 1 - fields_missing / (6 * len(df))
    dedup_rate = 1 - content_dupes / len(df)
    score = (completeness * 0.5 + dedup_rate * 0.5) * 100
    print(f"\nDATA QUALITY SCORE: {score:.1f}/100")
    print("(a blended score like this can still hide field-level gaps -- the breakdown above is the real answer)")


if __name__ == "__main__":
    main()
