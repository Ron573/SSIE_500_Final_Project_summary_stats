"""
quick_stats.py – display summary.csv and a few derived numbers
"""

import argparse
import pandas as pd
from pathlib import Path


def main() -> None:
    # ---------- 1. Command-line arguments -------------
    parser = argparse.ArgumentParser(
        description="Display a summary.csv and some derived numbers"
    )
    parser.add_argument(
        "table",
        help="Path to summary.csv that you want to inspect"
    )
    parser.add_argument(
        "-o", "--output",
        help="Optional path: write a copy of the table there"
    )
    args = parser.parse_args()

    # ---------- 2. Load the CSV -----------------------
    df = pd.read_csv(args.table)

    # ---------- 3. Show derived numbers ---------------
    print("Rows :", len(df))
    print("Cols :", df.shape[1])
    print()
    print(df.describe(include="all"))

    # ---------- 4. Optional write-out -----------------
    if args.output:
        Path(args.output).write_text(df.to_csv(index=False))
        print(f"\nTable copied to {args.output}")


if __name__ == "__main__":
    main()

