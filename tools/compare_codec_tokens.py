#!/usr/bin/env python3
"""
Compare codec tokens from different seeds to identify patterns
that correlate with audio noise/artifacts.

Usage:
    python tools/compare_codec_tokens.py codec_dumps/seed42.txt codec_dumps/seed7.txt ...
"""

import sys
import numpy as np
from collections import Counter


def load_codes(path):
    """Load codec token dump: returns [n_steps, 16] int array."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            tokens = [int(x) for x in line.split('\t')]
            rows.append(tokens)
    return np.array(rows, dtype=np.int64)


def analyze_codes(name, codes):
    """Print statistics about a codec token matrix."""
    n_steps, n_groups = codes.shape
    print(f"\n{'='*60}")
    print(f"  {name}: {n_steps} steps x {n_groups} groups")
    print(f"{'='*60}")

    # CB0 (first codebook - talker output)
    cb0 = codes[:, 0]
    print(f"\n  CB0 (talker): min={cb0.min()}, max={cb0.max()}, "
          f"unique={len(np.unique(cb0))}/{n_steps}")
    print(f"    Mean: {cb0.mean():.1f}, Std: {cb0.std():.1f}")

    # Token frequency distribution for CB0
    counter = Counter(cb0.tolist())
    most_common = counter.most_common(5)
    print(f"    Top-5 tokens: {most_common}")

    # Check for repetition in CB0
    repeats = sum(1 for i in range(1, len(cb0)) if cb0[i] == cb0[i-1])
    print(f"    Adjacent repeats: {repeats}/{n_steps-1}")

    # Sub-codes (code predictor output, groups 1-15)
    sub = codes[:, 1:]
    print(f"\n  Sub-codes (groups 1-15):")
    print(f"    Overall min={sub.min()}, max={sub.max()}")
    print(f"    Overall mean: {sub.mean():.1f}, std: {sub.std():.1f}")

    # Per-group statistics
    for g in range(min(4, n_groups-1)):  # first 4 sub-groups
        col = codes[:, g+1]
        print(f"    Group {g+1}: min={col.min()}, max={col.max()}, "
              f"mean={col.mean():.1f}, std={col.std():.1f}, "
              f"unique={len(np.unique(col))}")

    # Check for anomalous patterns: sub-codes stuck at 0 or at max
    zeros_per_step = (sub == 0).sum(axis=1)
    max_vals = (sub == 2047).sum(axis=1)
    print(f"\n  Anomaly check:")
    print(f"    Steps with >10 zero sub-codes: {(zeros_per_step > 10).sum()}")
    print(f"    Steps with any max-val (2047): {(max_vals > 0).sum()}")
    print(f"    Mean zeros per step: {zeros_per_step.mean():.2f}")

    # Entropy of each sub-code group
    print(f"\n  Per-group entropy (bits):")
    for g in range(n_groups-1):
        col = codes[:, g+1]
        counts = np.bincount(col, minlength=2048)
        probs = counts[counts > 0] / counts.sum()
        entropy = -np.sum(probs * np.log2(probs))
        print(f"    Group {g+1:2d}: {entropy:.2f} bits")

    return codes


def compare_pair(name1, codes1, name2, codes2):
    """Compare two codec token matrices."""
    print(f"\n{'='*60}")
    print(f"  COMPARISON: {name1} vs {name2}")
    print(f"{'='*60}")

    n1, n2 = codes1.shape[0], codes2.shape[0]
    n_common = min(n1, n2)

    # CB0 comparison
    cb0_1 = codes1[:n_common, 0]
    cb0_2 = codes2[:n_common, 0]
    cb0_same = (cb0_1 == cb0_2).sum()
    print(f"\n  CB0 overlap: {cb0_same}/{n_common} steps ({100*cb0_same/n_common:.1f}%)")

    # Per-step sub-code similarity
    sub1 = codes1[:n_common, 1:]
    sub2 = codes2[:n_common, 1:]
    per_step_same = (sub1 == sub2).sum(axis=1)
    print(f"  Sub-code overlap per step: mean={per_step_same.mean():.1f}/15, "
          f"min={per_step_same.min()}, max={per_step_same.max()}")

    # Token value distribution shift
    for g in [0, 1, 7, 14]:  # sample groups
        v1 = codes1[:n_common, g]
        v2 = codes2[:n_common, g]
        print(f"  Group {g:2d}: mean diff={v2.mean()-v1.mean():+.1f}, "
              f"std ratio={v2.std()/(v1.std()+1e-10):.2f}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_codec_tokens.py <file1.txt> [file2.txt ...]")
        sys.exit(1)

    files = sys.argv[1:]
    all_codes = {}

    for f in files:
        name = f.split('/')[-1].replace('.txt', '')
        codes = load_codes(f)
        all_codes[name] = analyze_codes(name, codes)

    # Pairwise comparison
    names = list(all_codes.keys())
    if len(names) >= 2:
        for i in range(len(names)):
            for j in range(i+1, len(names)):
                compare_pair(names[i], all_codes[names[i]],
                            names[j], all_codes[names[j]])


if __name__ == "__main__":
    main()
