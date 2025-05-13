#!/usr/bin/env python3
# ------------------------------------------------------------
#  chi2_v1.3.py   –  Adaptive-pool χ² with zero-cell protection
#  Ron B., May-2025
# ------------------------------------------------------------
"""
Usage
-----
python3 chi2_v1.3.py  data/corpus_counts.csv  -o chi2_results_v1.3.csv

Input
-----
CSV with a header: first column = language (or id),
all remaining columns = observed counts for the k categories.

Output
------
CSV with χ², p-value, df, pooling information (see README).
A sidecar audit/ folder contains a JSON file per language that shows
the pooled table actually used in the test.
"""
import argparse, json, math, os, sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chisquare

MIN_EXPECTED = 5         # Cochran rule
EPSILON      = 0.5       # Haldane–Anscombe correction


# ------------------------------------------------------------
# Helper: collapse low-expected categories until all E>=5
# ------------------------------------------------------------
def adaptive_pooling(observed, min_expected=MIN_EXPECTED):
    """
    observed : 1-D np.array of counts (length k)
    Returns
        pooled_counts : 1-D np.array  (length ≤ k)
        pooled        : bool          (True if any merge happened)
    Algorithm
    ---------
    Repeatedly merge the two *smallest* adjacent categories until
    every expected count (n_total/len(counts)) ≥ min_expected.
    """
    counts = observed.astype(float).copy()
    pooled = False
    while True:
        n = counts.sum()
        if n == 0:
            # All zeros – trivially return a single cell so χ²=0, p=1
            return np.array([0.0]), pooled
        expected = n / len(counts)
        if expected >= min_expected:
            return counts, pooled
        # need to merge
        pooled = True
        # index of smallest category
        i = counts.argmin()
        if i == 0:
            j = 1          # merge with right neighbour
        elif i == len(counts) - 1:
            j = i - 1      # merge with left neighbour
        else:
            # merge with the smaller of the two neighbours
            j = i - 1 if counts[i - 1] <= counts[i + 1] else i + 1
        # Perform merge (keep lower index)
        counts[i] += counts[j]
        counts = np.delete(counts, j)


def chi2_for_row(counts):
    """
    counts : iterable of ints
    Returns tuple (chi2_adj, p_adj, df, pool_method, eps_used, pooled_counts)
    """
    counts = np.asarray(counts, dtype=float)
    pooled_counts, pooled = adaptive_pooling(counts)

    if pooled_counts.size == 1:
        # degenerate (all in one cell) – χ² undefined, set to 0
        return 0.0, 1.0, 0, "pooled" if pooled else "none", 0.0, pooled_counts

    n = pooled_counts.sum()
    expected = np.full_like(pooled_counts, n / pooled_counts.size)

    if np.any(expected == 0):
        # Shouldn’t happen after pooling, but guard anyway
        pooled_counts = pooled_counts + EPSILON
        expected = expected + EPSILON
        eps_used = EPSILON
        pool_method = "eps"
    else:
        eps_used = 0.0
        pool_method = "pooled" if pooled else "none"

    chi2, p = chisquare(pooled_counts, f_exp=expected)
    df = pooled_counts.size - 1
    return chi2, p, df, pool_method, eps_used, pooled_counts


def main():
    ap = argparse.ArgumentParser(description="Zero-safe χ² with adaptive pooling")
    ap.add_argument("infile",  help="Input counts CSV")
    ap.add_argument("-o", "--outfile", default="chi2_results_v1.3.csv",
                    help="Output CSV (default: %(default)s)")
    args = ap.parse_args()

    df_in  = pd.read_csv(args.infile)
    if df_in.shape[1] < 3:
        sys.exit("Need at least 1 id column + 2 count columns")

    id_col = df_in.columns[0]
    cat_cols = df_in.columns[1:]

    out_rows = []
    audit_dir = Path("audit")
    audit_dir.mkdir(exist_ok=True)

    for _, row in df_in.iterrows():
        lang_id = row[id_col]
        counts  = row[cat_cols].to_numpy(dtype=float)

        chi2, p, df, pool_method, eps_used, pooled_counts = chi2_for_row(counts)

        out_rows.append(dict(
            language      = lang_id,
            total_obs     = int(counts.sum()),
            df            = int(df),
            chi2_adj      = round(chi2, 5),
            p_value_adj   = round(p, 6),
            pool_method   = pool_method,
            eps_used      = eps_used
        ))

        # dump audit file
        audit_file = audit_dir / f"{lang_id}_pooled.json"
        with open(audit_file, "w") as fh:
            json.dump({
                "original_counts": counts.tolist(),
                "pooled_counts"  : pooled_counts.tolist(),
                "chi2_adj"       : chi2,
                "p_value_adj"    : p,
                "df"             : df,
                "pool_method"    : pool_method,
                "eps_used"       : eps_used
            }, fh, indent=2)

    pd.DataFrame(out_rows).to_csv(args.outfile, index=False)
    print(f"Wrote {args.outfile}  (+ {len(out_rows)} audit files in {audit_dir}/)")


if __name__ == "__main__":
    main()

