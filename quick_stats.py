#!/usr/bin/env python3
"""
quick_stats.py – print summary.csv and some derived numbers
"""

import argparse
import pandas as pd
from pathlib import Path


def main() -> None:
    # ---------- 1. Command-line arguments ------------------------------
    parser = argparse.ArgumentParser(
        description="Display a summary.csv and some derived numbers"
    )
    parser.add_argument("table", help="Path to summary.csv that you want to inspect")
    parser.add_argument("-o", "--output",
                        help="Optional path: write a copy of the table there")
    args = parser.parse_args()

    # ---------- 2. Load the CSV ----------------------------------------
    tbl_path = Path(args.table).expanduser()
    if not tbl_path.exists():
        parser.error(f"{tbl_path} does not exist")

    df = pd.read_csv(tbl_path, index_col=0)

    # ---------- 3. Calculations & nicely formatted output --------------
    print("\n==== Summary table ====\n")
    print(df.to_string())

    # 3a. Derived manuscript numbers
    H_low  = df["unigram_H"].min()
    H_high = df["unigram_H"].max()
    E_low  = df.loc[df["unigram_H"].idxmin(), "landauer_25C"]
    E_high = df.loc[df["unigram_H"].idxmax(), "landauer_25C"]

    # 3b. Paper-ready numbers
    print("\n==== Paste-into-PDF values ====")
    BYTES_PER_WORD = 5
    df["plain_len"] = df["total_words"] * BYTES_PER_WORD

    CR  = 100 * (1 - df["compressed_len"] / df["plain_len"])
    dKL = 100 * (1 - df["comp_KL_P3"]   / df["KL_P3"])

    print(f"CR_min   = {CR.min():4.1f} %")
    print(f"CR_max   = {CR.max():4.1f} %")
    print(f"ΔKL mean = {dKL.mean():4.1f} %\n")

    print(f"H_low    = {H_low :6.2f} bits/word")
    print(f"H_high   = {H_high:6.2f} bits/word")
    print(f"E_low    = {E_low :6.2e} J/word (25 °C)")
    print(f"E_high   = {E_high:6.2e} J/word (25 °C)")

    # ---------- 4. Optional output copy --------------------------------
    if args.output:
        df.to_csv(args.output)
        print(f"\nTable written to {args.output}")


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()

