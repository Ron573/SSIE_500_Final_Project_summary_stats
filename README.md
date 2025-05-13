<!--
========================================================================
  High-Order Markov Analysis & Landauer-Limited Energy Savings
  README.md — full walk-through, reproducible from a blank machine
========================================================================
-->

<p align="center">
  <img src="docs/banner.svg" width="70%" alt="Project banner">
</p>

[![Build & Reproducibility](https://github.com/YOURNAME/markov-landauer/actions/workflows/ci.yml/badge.svg)](…)
[![DOI](https://zenodo.org/badge/…)](…)

> “If it isn’t reproducible, it doesn’t exist.” — paraphrasing D. Knuth  

This repository contains **all code, data, and manuscript sources** required to
• compute third-order Markov statistics for a text corpus,  
• compress that corpus with canonical Huffman codes,  
• translate the saved bits into *Landauer* energy at 300 K, and  
• Compile a camera-ready PDF describing the results.

The entire workflow is automated, version-controlled, and continuously tested
on GitHub Actions and the reference Docker image published at Docker Hub.

---

## 🚀 One-line quick start (Linux/macOS)

```bash
curl -fsSL https://raw.githubusercontent.com/YOURNAME/markov-landauer/main/bootstrap.sh | bash
Opening `main.tex` in TeXShop (or running `latexmk -pdf main.tex`) then yields a camera-ready PDF with every number already filled in.
Repository roadmap---.
├── bootstrap.sh              # fire-and-forget installer
├── environment.yml           # conda env spec (Python + TeX + tooling)
├── build_markov_and_landauer.py
├── Makefile                  # canonical command entry points
├── data/                     # (ignored) place your *.txt corpus here
├── examples/                 # tiny toy corpus, passes CI
├── results/                  # auto-generated artefacts (ignored)
├── figures/                  # auto-generated plots (ignored)
├── main.tex                  # manuscript root
├── sections/                 # introduction, methods, …
├── refs.bib                  # BibLaTeX bibliography
├── docs/                     # diagrams, banner, extra markdown
└── .github/                  # CI workflow, templates

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
