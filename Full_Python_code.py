#!/usr/bin/env python3
# build_markov_and_landauer.py
#
# One-file, reproducible analysis pipeline.

import argparse, json, math, random, statistics, sys, time
from collections import Counter, defaultdict
from heapq import heappush, heappop
from itertools import islice
from pathlib import Path
from typing import Dict, List, Tuple

KB   = 1.380_649e-23        # J/K
TEMP = 300.0                # K
LN2  = math.log(2.0)

# ----------------------------------------------------------------------
# 1.  Utilities
# ----------------------------------------------------------------------
def read_text(path: Path) -> str:
    import unicodedata
    with path.open('rb') as f:
        raw = f.read()
    txt = raw.decode('utf-8', 'strict')
    return unicodedata.normalize('NFC', txt)

def sliding_windows(seq, order):
    it = iter(seq)
    ctx = tuple(islice(it, order))
    for sym in it:
        yield ctx, sym
        ctx = (*ctx[1:], sym)

def build_kplus1_counts(seq: str, order: int) -> Dict[Tuple[str, ...], Counter]:
    counts: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
    for ctx, sym in sliding_windows(seq, order):
        counts[ctx][sym] += 1
    return counts

def entropy_from_counts(counts: Dict[Tuple[str, ...], Counter]) -> float:
    total_events = sum(sum(cnts.values()) for cnts in counts.values())
    H = 0.0
    for cnts in counts.values():
        Z = sum(cnts.values())
        for n in cnts.values():
            p = n / Z
            H -= (Z / total_events) * p * math.log2(p)
    return H

# ----------------------------------------------------------------------
# 2.  Canonical Huffman coding
# ----------------------------------------------------------------------
def huffman_code(freqs: Counter) -> Dict[str, str]:
    heap = [[wt, [sym, ""]] for sym, wt in freqs.items()]
    if len(heap) == 1:                 # edge-case: single symbol file
        wt, [sym, _] = heap[0]
        return {sym: "0"}
    for item in heap:
        heappush([], item)       # satisfy type checker
    heapq = heap
    while len(heapq) > 1:
        lo = heappop(heapq)
        hi = heappop(heapq)
        for pair in lo[1:]:
            pair[1] = "0" + pair[1]
        for pair in hi[1:]:
            pair[1] = "1" + pair[1]
        heappush(heapq, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    code = {}
    for wt, symbits in heapq[0][1:]:
        sym, bits = symbits
        code[sym] = bits
    return code

def encode_to_bits(seq: str, code: Dict[str, str]) -> List[int]:
    bits = []
    for ch in seq:
        bits.extend(1 if b == "1" else 0 for b in code[ch])
    return bits

# ----------------------------------------------------------------------
# 3.  Landauer energy
# ----------------------------------------------------------------------
def landauer(bits: int) -> float:
    return bits * KB * TEMP * LN2

# ----------------------------------------------------------------------
# 4.  Bootstrap CI
# ----------------------------------------------------------------------
def ci95(values, B=2000, seed=42):
    rnd = random.Random(seed)
    means = []
    n = len(values)
    for _ in range(B):
        sample = [values[rnd.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    lo = means[int(0.025*B)]
    hi = means[int(0.975*B)]
    return statistics.fmean(values), lo, hi

# ----------------------------------------------------------------------
# 5.  χ² goodness-of-fit (3rd-order vs unigram = 1st-order)
# ----------------------------------------------------------------------
def chi2_order_test(quad_counts, unigram) -> Tuple[float, int, float]:
    chisq, dof = 0.0, 0
    total_tokens = sum(unigram.values())
    P_uni = {s: n/total_tokens for s, n in unigram.items()}
    for ctx, cnts in quad_counts.items():
        Nctx = sum(cnts.values())
        for sym, obs in cnts.items():
            exp = P_uni.get(sym, 0) * Nctx
            if exp > 0:
                chisq += (obs - exp) ** 2 / exp
                dof   += 1
    dof = max(dof-1, 1)
    try:
        from scipy.stats import chi2
        p = 1 - chi2.cdf(chisq, dof)
    except ImportError:
        p = float('nan')
    return chisq, dof, p

# ----------------------------------------------------------------------
# 6.  Per-file processing
# ----------------------------------------------------------------------
def analyse_file(path: Path, order: int = 3):
    text = read_text(path)
    n_chars = len(text)
    if n_chars < order + 1:
        raise ValueError(f"{path} too short for order {order}")
    # char-level Markov
    M3_counts = build_kplus1_counts(text, order)
    H3        = entropy_from_counts(M3_counts)
    PPL       = 2 ** H3

    # Huffman
    freqs = Counter(text)
    code  = huffman_code(freqs)
    bits  = encode_to_bits(text, code)
    n_bits_raw  = n_chars * 8
    n_bits_huff = len(bits)

    # bit-level Markov
    bit_str = ''.join('1' if b else '0' for b in bits)
    Bit_counts = build_kplus1_counts(bit_str, order)
    H3_bit = entropy_from_counts(Bit_counts)
    PPL_bit = 2 ** H3_bit

    # energies
    E_raw  = landauer(n_bits_raw)
    E_huff = landauer(n_bits_huff)

    return {
        "chars": n_chars,
        "bits_raw": n_bits_raw,
        "bits_huff": n_bits_huff,
        "ratio": n_bits_huff / n_bits_raw,
        "E_raw": E_raw,
        "E_huff": E_huff,
        "H3_raw": H3,
        "PPL_raw": PPL,
        "H3_huff": H3_bit,
        "PPL_huff": PPL_bit,
        "M3_counts": M3_counts,          # for χ² later
    }

# ----------------------------------------------------------------------
# 7.  Main driver
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="High-order Markov + Landauer pipeline")
    ap.add_argument("files", nargs="+", help="text files to analyse")
    ap.add_argument("-o", "--outdir", default="results",
                    help="directory for artefacts")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "tables").mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)

    all_stats   = {}
    unigram     = Counter()
    quad_counts = defaultdict(Counter)

    print("Processing files …")
    t0 = time.time()
    for fp in map(Path, args.files):
        st = analyse_file(fp)
        all_stats[fp.name] = st
        unigram.update(list(read_text(fp)))
        # 4-gram counts for χ² test (context len 3)
        quad_counts.update(st["M3_counts"])
        # drop heavy counts to save RAM before JSON
        st["M3_counts"] = None
    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s")

    # χ² order selection
    chi2_val, dof, p = chi2_order_test(quad_counts, unigram)
    all_stats["__meta"] = {"chi2": (chi2_val, dof, p)}

    # corpus-level summaries
    ratios  = [d["ratio"] for d in all_stats.values() if not d.startswith("__")]
    savings = [(1-d["ratio"])*100 for d in ratios]
    pplpost = [d["PPL_huff"] for d in all_stats.values() if not d.startswith("__")]

    r_mean, r_lo, r_hi = ci95(ratios)
    s_mean, s_lo, s_hi = ci95(savings)
    p_mean, p_lo, p_hi = ci95(pplpost)

    all_stats["__meta"].update({
        "alphabet": len(unigram),
        "chars_total": sum(d["chars"] for d in all_stats.values()
                           if not d.startswith("__")),
        "ratio_mean": r_mean,
        "energy_save_mean": s_mean,
        "ppl_post_mean": p_mean
    })

    # ------------------------------------------------------------------
    # JSON dumps
    # ------------------------------------------------------------------
    with (outdir / "markov_pre.json").open("w") as f:
        json.dump({fn: st["H3_raw"] for fn, st in all_stats.items()
                   if not fn.startswith("__")}, f)
    with (outdir / "markov_post.json").open("w") as f:
        json.dump({fn: st["H3_huff"] for fn, st in all_stats.items()
                   if not fn.startswith("__")}, f)

    # ------------------------------------------------------------------
    # LaTeX emit
    # ------------------------------------------------------------------
    emit_tex(outdir, all_stats, (r_mean, r_lo, r_hi),
             (s_mean, s_lo, s_hi), (p_mean, p_lo, p_hi))

    print("Artefacts written to", outdir)

# ----------------------------------------------------------------------
# 8.  LaTeX writer
# ----------------------------------------------------------------------
def emit_tex(outdir: Path, stats, ci_r, ci_s, ci_p):
    r_mean, r_lo, r_hi = ci_r
    s_mean, s_lo, s_hi = ci_s
    p_mean, p_lo, p_hi = ci_p
    chi2_val, dof, pval = stats["__meta"]["chi2"]

    # results.tex (global macros)
    with (outdir / "results.tex").open("w") as f:
        f.write(f"\\newcommand{{\\Nfiles}}{{{len(stats)-1}}}\n")
        f.write(f"\\newcommand{{\\AlphabetSize}}{{{stats['__meta']['alphabet']}}}\n")
        f.write(f"\\newcommand{{\\NcharsTotal}}{{{stats['__meta']['chars_total']}}}\n")
        f.write(f"\\newcommand{{\\MeanComprRatio}}{{{r_mean:.3f}}}\n")
        f.write(f"\\newcommand{{\\MeanComprRatioLo}}{{{r_lo:.3f}}}\n")
        f.write(f"\\newcommand{{\\MeanComprRatioHi}}{{{r_hi:.3f}}}\n")
        f.write(f"\\newcommand{{\\MeanEnergySaving}}{{{s_mean:.1f}}}\n")
        f.write(f"\\newcommand{{\\MeanEnergySavingLo}}{{{s_lo:.1f}}}\n")
        f.write(f"\\newcommand{{\\MeanEnergySavingHi}}{{{s_hi:.1f}}}\n")
        f.write(f"\\newcommand{{\\MeanPostPPL}}{{{p_mean:.1f}}}\n")
        f.write(f"\\newcommand{{\\MeanPostPPLLo}}{{{p_lo:.1f}}}\n")
        f.write(f"\\newcommand{{\\MeanPostPPLHi}}{{{p_hi:.1f}}}\n")
        f.write(f"\\newcommand{{\\ChiTwo}}{{{chi2_val:.0f}}}\n")
        f.write(f"\\newcommand{{\\ChiTwoDF}}{{{dof}}}\n")
        f.write(f"\\newcommand{{\\ChiTwoP}}{{{pval:.3e}}}\n")

    # per-file rows
    with (outdir / "tables" / "file_rows.tex").open("w") as g:
        for fname, d in stats.items():
            if fname.startswith("__"):
                continue
            g.write(f"{fname} & {d['chars']} & {d['bits_raw']} & "
                    f"{d['bits_huff']} & {d['ratio']:.3f} & "
                    f"{d['E_raw']:.2e} & {d['E_huff']:.2e} & "
                    f"{(1-d['ratio'])*100:.1f}\\\\\n")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()

