#!/usr/bin/env python3
"""
Build third-order Markov transition matrices for
  • raw text
  • Huffman-compressed bitstream
and compute Landauer erase-energy bounds.

Usage
-----
python build_markov_and_landauer.py input.txt
"""

from __future__ import annotations
import argparse, math, heapq, json, itertools, sys
from collections import Counter, defaultdict
from pathlib import Path

K_BOLTZ = 1.380649e-23        # J/K
LN2     = math.log(2.0)
ROOM_T  = 300.0               # Kelvin (adjust if needed)
E_LANDAUER = K_BOLTZ * ROOM_T * LN2   # ≈ 2.85e-21 J

# ----------------------------------------------------------------------
# 1) Third-order Markov matrix builder
# ----------------------------------------------------------------------
def third_order_counts(sequence: list[str|int]) -> dict[str, Counter]:
    """Return counts[context][next] for a sequence of symbols."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for a, b, c, nxt in zip(sequence, sequence[1:], sequence[2:], sequence[3:]):
        ctx = (a, b, c)
        counts[ctx][nxt] += 1
    return counts

def normalise_counts(counts: dict[str, Counter]) -> dict[str, dict]:
    """Convert nested Counters to probability dicts."""
    probs: dict[str, dict] = {}
    for ctx, cnt in counts.items():
        total = sum(cnt.values())
        probs[ctx] = {sym: n / total for sym, n in cnt.items()}
    return probs

# ----------------------------------------------------------------------
# 2) Huffman coding utils
# ----------------------------------------------------------------------
class Node:
    def __init__(self, freq, sym=None, left=None, right=None):
        self.freq, self.sym, self.left, self.right = freq, sym, left, right
    def __lt__(self, other): return self.freq < other.freq

def build_huffman_codes(data: list[str]) -> dict[str, str]:
    freq = Counter(data)
    heap = [Node(f, s) for s, f in freq.items()]
    heapq.heapify(heap)
    if len(heap) == 1:          # Edge-case: only one symbol
        return {heap[0].sym: "0"}
    while len(heap) > 1:
        n1, n2 = heapq.heappop(heap), heapq.heappop(heap)
        heapq.heappush(heap, Node(n1.freq + n2.freq, left=n1, right=n2))
    root = heap[0]
    codes: dict[str, str] = {}
    def _walk(node, prefix=""):
        if node.sym is not None:
            codes[node.sym] = prefix
        else:
            _walk(node.left,  prefix + "0")
            _walk(node.right, prefix + "1")
    _walk(root)
    return codes

def encode_huffman(data: list[str], codes: dict[str, str]) -> list[str]:
    bitstring = "".join(codes[sym] for sym in data)
    return list(bitstring)          # return list of '0'/'1' chars

# ----------------------------------------------------------------------
# 3) Driver
# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="3rd-order Markov + Landauer demo")
    ap.add_argument("file", type=Path, help="Plain-text file to analyse")
    ap.add_argument("-o", "--outdir", type=Path,
                    help="Directory to dump JSON matrices (default: alongside input)")
    args = ap.parse_args()

    text = args.file.read_text(encoding="utf-8", errors="replace")
    symbols = list(text)           # char-level model; swap in tokeniser if needed

    # Pre-Huffman ----------------------------------------------------------------
    pre_counts  = third_order_counts(symbols)
    pre_matrix  = normalise_counts(pre_counts)
    pre_bits    = len(text.encode("utf-8")) * 8
    pre_energy  = pre_bits * E_LANDAUER

    # Huffman --------------------------------------------------------------------
    codes       = build_huffman_codes(symbols)
    bit_seq     = encode_huffman(symbols, codes)
    post_counts = third_order_counts(bit_seq)
    post_matrix = normalise_counts(post_counts)
    post_bits   = len(bit_seq)
    post_energy = post_bits * E_LANDAUER
    compression_ratio = post_bits / pre_bits

    # ---------------------------------------------------------------------------
    # 4) Output
    # ---------------------------------------------------------------------------
    outdir = args.outdir or args.file.parent
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "markov_pre.json").write_text(json.dumps(pre_matrix, indent=2))
    (outdir / "markov_post.json").write_text(json.dumps(post_matrix, indent=2))

    print("─────────────────────────────────")
    print("Third-order Markov matrices written to")
    print("  •", outdir / "markov_pre.json")
    print("  •", outdir / "markov_post.json")
    print("Each JSON entry looks like")
    print('  ["c1","c2","c3"] : {"x": 0.23, "y": 0.77, ...}')
    print("─────────────────────────────────\n")

    # Landauer report
    print("Landauer erase-energy @ 300 K")
    print(f"  Original  : {pre_bits:>12,d} bits → {pre_energy:10.3e}  J")
    print(f"  Huffman   : {post_bits:>12,d} bits → {post_energy:10.3e}  J")
    print(f"  Compression ratio (bits) : {compression_ratio:.3f}×")
    print(f"  Energy savings (theoretical) : {(1-compression_ratio)*100:5.1f}%")

if __name__ == "__main__":
    main()

