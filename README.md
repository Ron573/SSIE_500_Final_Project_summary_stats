
Opening `main.tex` in TeXShop (or running `latexmk -pdf main.tex`) then yields a camera-ready PDF with every number already filled in.

---

## 📂 Repository layout


# 1. Analyse every UTF-8 text file in the repo
python build_markov_and_landauer.py $(git ls-files '*.txt')

# 2. Build the paper
latexmk -pdf main.tex               # or open in TeXShop and click “Typeset”


# High-Order Markov Analysis & Landauer-Limited Energy Savings  
_Third-order character models, canonical Huffman compression, and thermodynamic lower bounds_

---

## ✨ What is this?

This repository contains the complete, reproducible pipeline and manuscript for our study of

* per-file **third-order Markov** statistics on a plain-text code corpus,  
* **Huffman compression** effectiveness,  
* the resulting **Landauer minimum erase energy** at 300 K, and  
* a χ²-based comparison against a first-order model.

Running one Python script (`build_markov_and_landauer.py`) produces:


Opening `main.tex` in TeXShop (or running `latexmk -pdf main.tex`) then yields a camera-ready PDF with every number already filled in.

---

## 📂 Repository layout


---

## 🛠️ Requirements

| Software | Tested with | Notes |
|----------|-------------|-------|
| Python   | 3.9 – 3.12  | Standard library + **SciPy** (for χ² p-value) |
| TeX      | TeX Live 2022+/MacTeX 2022+ | Needs `pgfplots`, `biber`, `siunitx`, `booktabs`, `hyperref` |
| Git      | any         | Only for dataset retrieval / version control |

Install Python deps once:

```bash
python -m pip install --upgrade pip
pip install scipy
# 1. Analyse every UTF-8 text file in the repo
python build_markov_and_landauer.py $(git ls-files '*.txt')

# 2. Build the paper
latexmk -pdf main.tex               # or open in TeXShop and click “Typeset”
Your Name (2025). “High-Order Markov Analysis and Landauer-Limited
Energy Savings under Huffman Compression.” arXiv:<preprint-id>.
@misc{Your2025Markov,
  author       = {Your, Name},
  title        = {High-Order Markov Analysis and Landauer-Limited Energy Savings under Huffman Compression},
  year         = {2025},
  eprint       = {xxxx.xxxxx},
  archivePrefix= {arXiv}
}
