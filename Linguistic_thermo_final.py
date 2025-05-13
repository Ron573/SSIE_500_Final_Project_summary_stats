#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thermodynamic Cost of Linguistic Compression
Complete end-to-end pipeline, Markov order = 3, auto-ZIP outputs
"""
import os, re, math, json, heapq, datetime, shutil
from collections import Counter, defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, chi2 as scipy_chi2_dist
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D    # noqa: F401
import seaborn as sns

# ── Configuration ────────────────────────────────────────────────────
MARKOV_ORDER        = 3                 # ← only one knob to change
N_CHARS_TO_EXTRACT  = 50_000

BASE_PROJECT_PATH   = "/Users/RonB/Desktop/SSIE_500_Final_Project/Final_Project_CLEAN/"
BASE_LANG_PATH      = os.path.join(BASE_PROJECT_PATH, "Languages")

TIMESTAMP           = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
MAIN_OUT            = os.path.join(BASE_PROJECT_PATH, f"Analysis_Results_{TIMESTAMP}")
TRUNCATED_PATH      = os.path.join(MAIN_OUT, "Truncated_Texts")
PLOTS_PATH          = os.path.join(MAIN_OUT, "Plots")
TABLES_PATH         = os.path.join(MAIN_OUT, "Tables")
ENCODED_PATH        = os.path.join(MAIN_OUT, "Encoded_Bit_Streams")
MODELS_PATH         = os.path.join(MAIN_OUT, "Markov_Models")

for p in (MAIN_OUT, TRUNCATED_PATH, PLOTS_PATH, TABLES_PATH,
          ENCODED_PATH, MODELS_PATH):
    os.makedirs(p, exist_ok=True)

ORIGINAL_FILES = {
    "English":  os.path.join(BASE_LANG_PATH, "English.tex"),
    "Spanish":  os.path.join(BASE_LANG_PATH, "Spanish.tex"),
    "Russian":  os.path.join(BASE_LANG_PATH, "Russian.tex"),
    "Chinese":  os.path.join(BASE_LANG_PATH, "Chinese.tex"),
    "Hindi":    os.path.join(BASE_LANG_PATH, "Hindi.tex"),
    "Turkish":  os.path.join(BASE_LANG_PATH, "Turkish.tex"),
}

K_B   = 1.380649e-23
LN_2  = math.log(2)
TEMPS = {"0C": 273.15, "20C": 293.15, "25C": 298.15, "37C": 310.15}
REF_T = "25C"

# ── Tiny LaTeX stripper & tokenizer ──────────────────────────────────
LATEX_CMD = re.compile(r"\\[a-zA-Z@]+(\[[^\]]*\])?(\{[^}]*\})?")
LATEX_ENV = re.compile(r"\\begin\{.*?}|\\end\{.*?}", re.DOTALL)
def strip_latex(s: str) -> str:
    s = LATEX_ENV.sub(" ", s)
    s = LATEX_CMD.sub(" ", s)
    s = re.sub(r"[{}]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def read_truncate(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return strip_latex(f.read())[:N_CHARS_TO_EXTRACT]

tokenize = lambda txt: re.findall(r"\S+", txt.lower())

# ── Info-theory helpers ──────────────────────────────────────────────
def H(counter):
    tot = sum(counter.values())
    return 0 if tot == 0 else -sum((c/tot)*math.log2(c/tot) for c in counter.values())

def huffman(counter):
    if not counter: return {}, 0
    heap = [[wt, [sym, ""]] for sym, wt in counter.items()]
    heapq.heapify(heap)
    if len(heap)==1:
        sym,_=heap[0][1]; return {sym:"0"},1
    while len(heap)>1:
        lo,hi = heapq.heappop(heap), heapq.heappop(heap)
        for p in lo[1:]: p[1]="0"+p[1]
        for p in hi[1:]: p[1]="1"+p[1]
        heapq.heappush(heap,[lo[0]+hi[0]]+lo[1:]+hi[1:])
    codes=dict(heap[0][1:])
    tot=sum(counter.values())
    return codes, sum(len(codes[s])*counter[s] for s in codes)/tot

def landauer(ent_bits, T): return ent_bits*K_B*LN_2*T

# ── Markov helpers ───────────────────────────────────────────────────
def build_markov(seq, order):
    model=defaultdict(Counter)
    for i in range(len(seq)-order):
        h=tuple(seq[i:i+order]); model[h][seq[i+order]]+=1
    return model

def ngram_H(seq, order):
    if order==0: return H(Counter(seq))
    if len(seq)<order: return 0
    grams=Counter(tuple(seq[i:i+order]) for i in range(len(seq)-order+1))
    tot=sum(grams.values())
    return -sum((c/tot)*math.log2(c/tot) for c in grams.values())

def KL_markov(pk, p0, tot_words):
    kl,trans = 0,0
    for h,nxt in pk.items():
        ht=sum(nxt.values())
        for w,c in nxt.items():
            pkp=c/ht; p0p=p0.get(w,0)/tot_words
            if pkp>0 and p0p>0:
                kl+=c*math.log2(pkp/p0p)
            trans+=c
    return kl/trans if trans else 0

def chi2_markov(m1, m2):
    """
    likelihood-ratio χ² (G-test) between two order-3 Markov models
    RETURNS  (total_χ², total_dof, global_p)
    """
    total_chi, total_df = 0.0, 0
    for h in set(m1) | set(m2):
        r1, r2 = m1.get(h, Counter()), m2.get(h, Counter())

        # keep only symbols that occur at least once in *either* row
        vocab = [v for v in set(r1) | set(r2) if r1.get(v, 0) + r2.get(v, 0) > 0]
        if len(vocab) < 2:
            continue                                      # nothing testable

        tbl = np.array([[r1.get(v, 0) for v in vocab],
                        [r2.get(v, 0) for v in vocab]])

        # if any column sum is zero, expected counts will be zero → drop it
        keep = tbl.sum(axis=0) > 0
        tbl  = tbl[:, keep]
        if tbl.shape[1] < 2:
            continue

        try:
            chi, p, df, _ = chi2_contingency(tbl, lambda_="log-likelihood")
            total_chi += chi
            total_df  += df
        except ValueError:          # still too sparse → skip this history
            continue

    global_p = 1 - scipy_chi2_dist.cdf(total_chi, total_df) if total_df else 1.0
    return total_chi, total_df, global_p

# ── Plot helpers (same as previous version, omitted for brevity)──────
def quick_bar(counter,label,file):
    if not counter: return
    words,counts=zip(*counter.most_common(30))
    plt.figure(figsize=(14,6)); plt.bar(words,counts)
    plt.xticks(rotation=60,ha="right"); plt.title(label); plt.tight_layout()
    plt.savefig(file); plt.close()

# ── Main loop ────────────────────────────────────────────────────────
metrics={}
for lang, path in ORIGINAL_FILES.items():
    print("●",lang)
    if not os.path.isfile(path): print("  file missing"); continue
    txt=read_truncate(path)
    with open(os.path.join(TRUNCATED_PATH,f"{lang}.txt"),"w",encoding="utf-8") as f: f.write(txt)
    words=tokenize(txt); tot=len(words)
    if tot<100: print("  too short"); continue
    freq=Counter(words); H1=H(freq)
    codes,avglen=huffman(freq)
    land={t:landauer(H1,T) for t,T in TEMPS.items()}
    mk  = build_markov(words,MARKOV_ORDER)
    H3  = ngram_H(words,MARKOV_ORDER)
    KL3 = KL_markov(mk,freq,tot)
    encoded="".join(codes[w] for w in words)
    with open(os.path.join(ENCODED_PATH,f"{lang}.bits"),"w") as f: f.write(encoded[:10000])
    # compressed stream analysis
    bits=list(encoded); freqb=Counter(bits); H1b=H(freqb)
    mkb=build_markov(bits,MARKOV_ORDER)
    H3b=ngram_H(bits,MARKOV_ORDER)
    KL3b=KL_markov(mkb,freqb,len(bits))
    chi_chi,chi_df,chi_p = chi2_markov(mk,mkb)
    # store
    row=dict(total_words=tot, unigram_H=H1, trigram_H=H3, KL_P3=KL3,
             avg_huff=avglen, landauer_25C=land["25C"],
             compressed_len=len(bits), comp_unigram_H=H1b,
             comp_trigram_H=H3b, comp_KL_P3=KL3b,
             chi2_pre_vs_post=chi_chi, chi_df=chi_df, chi_p=chi_p)
    metrics[lang]=row
    # word-frequency plot
    quick_bar(freq,f"Top-30 words – {lang}",
              os.path.join(PLOTS_PATH,f"{lang}_top30.png"))

# pairwise χ² between languages
for (l1,p1),(l2,p2) in combinations(list(ORIGINAL_FILES.items()),2):
    mk1=build_markov(tokenize(read_truncate(p1)),MARKOV_ORDER)
    mk2=build_markov(tokenize(read_truncate(p2)),MARKOV_ORDER)
    chi,df,p=chi2_markov(mk1,mk2)
    with open(os.path.join(TABLES_PATH,
               f"chi2_{l1}_vs_{l2}_P{MARKOV_ORDER}.json"),"w") as f:
        json.dump(dict(chi2=chi,df=df,p=p),f,indent=2)

# summary tables
df=pd.DataFrame(metrics).T
csv_path=os.path.join(TABLES_PATH,"summary.csv")
df.to_csv(csv_path)
print("✔ summary ->",csv_path)

# ── Auto-ZIP everything ──────────────────────────────────────────────
zip_name = os.path.join(BASE_PROJECT_PATH,
                        f"ThermoLanguageOutputs_{TIMESTAMP}")
shutil.make_archive(zip_name, "zip", MAIN_OUT)
print("✔ zipped ->",zip_name+".zip")



    













    





            




    













