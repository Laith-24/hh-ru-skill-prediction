"""
Runs the hh.ru API collector (src/collection/, from
joullanarAli/hh-ru-big-data-analysis, kept as provided) and saves raw
vacancies as JSON -- this is what originally produced
data/vacancies.parquet. Not wired into this project's own pipeline; kept
for provenance. Needs an hh.ru API app (see .env.example; register one
at https://dev.hh.ru/admin).

Usage: python3 src/collection/collect_data.py [--max-pages-per-query 15]
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from collection.auth import HHAuth
from collection.collector import HHCollector, SEARCH_QUERIES, REGIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-pages-per-query", type=int, default=15)
    parser.add_argument("--only-with-salary", action="store_true")
    parser.add_argument("--out", default=None, help="Output JSON path (default: data/output/hh_vacancies_<timestamp>.json)")
    args = parser.parse_args()

    load_dotenv()
    client_id = os.getenv("HH_CLIENT_ID")
    client_secret = os.getenv("HH_CLIENT_SECRET")
    if not client_id or not client_secret:
        parser.error("HH_CLIENT_ID / HH_CLIENT_SECRET not set -- copy .env.example to .env and fill them in")

    auth = HHAuth(client_id, client_secret)
    token = auth.get_access_token()
    if not token:
        raise SystemExit("Authentication failed")

    collector = HHCollector(token)
    collector.collect_large_dataset(
        SEARCH_QUERIES, list(REGIONS.values()),
        max_pages_per_query=args.max_pages_per_query,
        only_with_salary=args.only_with_salary,
    )
    path = collector.save_final(args.out)
    print(f"Saved {len(collector.vacancies)} vacancies to {path}")


if __name__ == "__main__":
    main()
