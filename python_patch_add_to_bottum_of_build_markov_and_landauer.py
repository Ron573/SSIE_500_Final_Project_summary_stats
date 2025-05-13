# ----------------------------------------------------------------------
#  A1.  Utility: bootstrap mean + 95 % CI
# ----------------------------------------------------------------------
def ci95(values, B=2000, seed=42):
    import random, statistics
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
#  A2.  χ² goodness-of-fit for order selection
# ----------------------------------------------------------------------
def chi2_gof(obs_counts, order):
    """
    Compare observed 4-gram counts to counts predicted by a lower-order
    Markov model.  Returns (χ², dof, p).
    If order==1 we test bigram model vs unigram prediction, etc.
    """
    import math
    chisq = 0.0
    dof   = 0
    for ctx, cnts in obs_counts.items():
        if len(ctx) != order:
            continue            # skip other orders
        total = sum(cnts.values())
        # expected probability from (order-1) model = marginal counts
        marg = defaultdict(int)
        for sym, n in cnts.items():
            marg[sym] += n
        for sym, n in cnts.items():
            p_exp = marg[sym] / total
            e = p_exp * total
            if e > 0:
                chisq += (n - e) ** 2 / e
                dof   += 1
    from scipy.stats import chi2
    p_val = 1 - chi2.cdf(chisq, dof-1)
    return chisq, dof-1, p_val

# ----------------------------------------------------------------------
#  A3.  Emit LaTeX with confidence intervals & χ² stats
# ----------------------------------------------------------------------
def emit_tex_numbers(outdir, stats):
    """Extended to include 95 % CIs and χ²."""
    from statistics import fmean
    ratios   = [d['ratio']     for d in stats.values() if not d.startswith("__")]
    savings  = [(1-d['ratio'])*100 for d in stats.values() if not d.startswith("__")]
    ppl_post = [d['PPL_huff']  for d in stats.values() if not d.startswith("__")]

    r_mean, r_lo, r_hi = ci95(ratios)
    s_mean, s_lo, s_hi = ci95(savings)
    p_mean, p_lo, p_hi = ci95(ppl_post)

    with open(outdir / "results.tex", "w") as f:
        f.write(f"\\newcommand{{\\MeanComprRatio}}{{{r_mean:.3f}}}\n")
        f.write(f"\\newcommand{{\\MeanComprRatioLo}}{{{r_lo:.3f}}}\n")
        f.write(f"\\newcommand{{\\MeanComprRatioHi}}{{{r_hi:.3f}}}\n")
        f.write(f"\\newcommand{{\\MeanEnergySaving}}{{{s_mean:.1f}}}\n")
        f.write(f"\\newcommand{{\\MeanEnergySavingLo}}{{{s_lo:.1f}}}\n")
        f.write(f"\\newcommand{{\\MeanEnergySavingHi}}{{{s_hi:.1f}}}\n")
        f.write(f"\\newcommand{{\\MeanPostPPL}}{{{p_mean:.1f}}}\n")
        f.write(f"\\newcommand{{\\MeanPostPPLLo}}{{{p_lo:.1f}}}\n")
        f.write(f"\\newcommand{{\\MeanPostPPLHi}}{{{p_hi:.1f}}}\n")
        # χ² result from order selection (pooled corpus counts)
        chi2_val, dof, p = stats['__meta']['chi2']
        f.write(f"\\newcommand{{\\ChiTwo}}{{{chi2_val:.0f}}}\n")
        f.write(f"\\newcommand{{\\ChiTwoDF}}{{{dof}}}\n")
        f.write(f"\\newcommand{{\\ChiTwoP}}{{{p:.3e}}}\n")

