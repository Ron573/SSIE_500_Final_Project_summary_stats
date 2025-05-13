# 1.  Analyse every text file in the repo
python build_markov_and_landauer.py --batch "$(git ls-files '*.txt')"

# 2.  Compile the PDF (TeXShop users can just click "Typeset")
latexmk -pdf main.tex        # or TeXShop ⇢ Typeset

