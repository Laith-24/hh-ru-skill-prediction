"""
Simulate a realistic "new postings arrived" scenario for demoing
retrain_on_new_data.py, since there's no genuinely new incoming data --
just the one static parquet file. Splits the full dataset into three
disjoint, clearly-labeled slices:
  - demo_old.parquet:  the "already trained on" batch
  - demo_new.parquet:  the "just arrived" batch to fine-tune on
  - demo_eval.parquet: held out from BOTH, used only to check whether the
                        fine-tuned model actually generalizes better

Kept small (4,000 rows total) so the demo runs in minutes, not hours --
its purpose is to exercise and show the mechanism, not to produce
production-quality demo models.

Usage:
    python3 src/make_demo_split.py --data data/vacancies.parquet --out-dir data/demo
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/vacancies.parquet")
    parser.add_argument("--out-dir", default="data/demo")
    parser.add_argument("--total", type=int, default=4000)
    parser.add_argument("--old-size", type=int, default=2400)
    parser.add_argument("--new-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    t = pq.read_table(args.data)
    total_rows = t.num_rows

    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(total_rows)[: args.total]
    old_idx = idx[: args.old_size]
    new_idx = idx[args.old_size: args.old_size + args.new_size]
    eval_idx = idx[args.old_size + args.new_size:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    old = t.take(pa.array(old_idx))
    new = t.take(pa.array(new_idx))
    ev = t.take(pa.array(eval_idx))

    pq.write_table(old, str(out_dir / "demo_old.parquet"))
    pq.write_table(new, str(out_dir / "demo_new.parquet"))
    pq.write_table(ev, str(out_dir / "demo_eval.parquet"))

    print(f"old: {old.num_rows} rows, new: {new.num_rows} rows, eval: {ev.num_rows} rows")


if __name__ == "__main__":
    main()
