import pandas as pd, numpy as np, os
tbl = "/Users/RonB/Desktop/SSIE_500_Final_Project/Final_Project_CLEAN/Analysis_Results_20250512_082408/Tables/summary.csv"
df  = pd.read_csv(tbl, index_col=0)

print("\n==== Summary table ====\n")
print(df.to_string())

# manuscript numbers
H_low  = df['unigram_H'].min()
H_high = df['unigram_H'].max()
E_low  = df.loc[df['unigram_H'].idxmin(), 'landauer_25C']
E_high = df.loc[df['unigram_H'].idxmax(), 'landauer_25C']
CR     = 100*(1 - df['compressed_len']/(df['total_words']*df['avg_huff']))
dKL    = 100*(1 - df['comp_KL_P3']/df['KL_P3'])

print("\n==== Paste-into-PDF values ====")
print(f"H_low   = {H_low:.2f}  bits/word")
print(f"H_high  = {H_high:.2f}  bits/word")
print(f"E_low   = {E_low:.2e}  J/word  (25 °C)")
print(f"E_high  = {E_high:.2e}  J/word  (25 °C)")
print(f"CR_min  = {CR.min():.1f} %")
print(f"CR_max  = {CR.max():.1f} %")
print(f"ΔKL mean= {dKL.mean():.1f} %")

