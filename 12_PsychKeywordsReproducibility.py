#!/usr/bin/env python3
"""
12_PsychKeywordsReproducibility.py - PSYCH_KEYWORDS F-Code Subset Reproducibility

Checks the construction methodology for the 37 F-codes in PSYCH_KEYWORDS (used by
the Keyword-Extraction and Keyword+CoT strategies, 6e_Eval_KeywordAugmented.py /
6f_Eval_KeywordAugmentedCoT.py), and specifically whether selection was performed
on training data only (to avoid test set leakage).

This script:
  1. Computes F-code frequency and cumulative coverage on the TRAINING set only,
     and reports how many codes are needed to reach 90% cumulative coverage.
  2. Compares the current 37-code PSYCH_KEYWORDS F-code subset against that
     training-set frequency ranking, listing exactly which codes differ.
  3. Checks whether the frequency ranking itself is sensitive to which split it
     is computed from, by independently computing the top-42 F-codes from train
     and test, and reporting the overlap between them. High overlap means the
     ranking (and therefore any list built from it) is not meaningfully
     advantaged by which split happened to be used.

Output:
    outputs/12_PsychKeywordsReproducibility/fcode_frequency_train.csv
    outputs/12_PsychKeywordsReproducibility/split_stability_comparison.csv
    outputs/12_PsychKeywordsReproducibility/summary.json

Usage:
    python 12_PsychKeywordsReproducibility.py
"""

import json
from collections import Counter
from pathlib import Path

import pandas as pd

from config import (
    RANDOM_SEED, set_all_seeds,
    TRAIN_DATA_PATH, TEST_DATA_PATH,
    OUTPUT_DIR_12_PSYCH_KEYWORDS,
)

set_all_seeds(RANDOM_SEED)

# The F-code subset of PSYCH_KEYWORDS, as currently used in 6e/6f.
CURRENT_FCODES = {
    'F01.50', 'F02.80', 'F02.81', 'F03.90', 'F03.91', 'F05', 'F10.10', 'F10.11',
    'F10.20', 'F10.21', 'F11.10', 'F11.20', 'F12.10', 'F14.10', 'F17.200', 'F17.210',
    'F20.0', 'F20.9', 'F25.9', 'F31.81', 'F31.9', 'F32.3', 'F32.9', 'F33.2', 'F39',
    'F40.240', 'F41.0', 'F41.1', 'F41.8', 'F41.9', 'F42.9', 'F43.10', 'F43.20',
    'F43.23', 'F60.3', 'F79', 'F90.9',
}

COVERAGE_TARGET = 90.0
STABILITY_CHECK_K = 42  # matches the actual F-code count in the final PSYCH_KEYWORDS list


def format_f_code(code: str) -> str:
    code = code.strip()
    if len(code) > 3 and '.' not in code:
        return f"{code[:3]}.{code[3:]}"
    return code


def fcode_counter(df: pd.DataFrame) -> Counter:
    df = df[df['f_codes_str'].notna()]
    all_codes = []
    for codes_str in df['f_codes_str']:
        all_codes.extend(format_f_code(c) for c in str(codes_str).split(','))
    return Counter(all_codes)


def ranked_coverage(counter: Counter):
    """Return [(rank, code, count, pct_of_instances, cumulative_pct), ...]."""
    total = sum(counter.values())
    rows = []
    cum = 0
    for rank, (code, count) in enumerate(counter.most_common(), 1):
        cum += count
        rows.append((rank, code, count, count / total * 100, cum / total * 100))
    return rows


def rank_for_coverage(rows, target_pct: float) -> int:
    for rank, _, _, _, cum_pct in rows:
        if cum_pct >= target_pct:
            return rank
    return len(rows)


def main():
    out_dir = Path(OUTPUT_DIR_12_PSYCH_KEYWORDS)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PSYCH_KEYWORDS F-CODE SUBSET REPRODUCIBILITY")
    print("=" * 70)

    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    # --- 1. Training-set frequency and coverage ---
    train_counter = fcode_counter(train_df)
    train_rows = ranked_coverage(train_counter)
    train_k90 = rank_for_coverage(train_rows, COVERAGE_TARGET)

    print(f"\nTraining set: {len(train_df):,} samples, "
          f"{sum(train_counter.values()):,} F-code instances, "
          f"{len(train_counter)} unique codes")
    print(f"Codes needed for {COVERAGE_TARGET:.0f}% cumulative coverage: {train_k90}")
    print(f"Cumulative coverage at rank 37: {train_rows[36][4]:.2f}%")

    freq_df = pd.DataFrame(train_rows, columns=['rank', 'f_code', 'count', 'pct_of_instances', 'cumulative_pct'])
    freq_csv = out_dir / 'fcode_frequency_train.csv'
    freq_df.to_csv(freq_csv, index=False)
    print(f"\nSaved: {freq_csv}")

    # --- 2. Current list vs fresh training-set recomputation ---
    train_top90_codes = set(code for _, code, *_ in train_rows[:train_k90])
    only_in_current = sorted(CURRENT_FCODES - train_top90_codes)
    only_in_recomputed = sorted(train_top90_codes - CURRENT_FCODES)
    overlap = len(CURRENT_FCODES & train_top90_codes)

    print(f"\nCurrent 37-code list vs training-set top-{train_k90} "
          f"({COVERAGE_TARGET:.0f}% coverage):")
    print(f"  Overlap: {overlap}/{len(CURRENT_FCODES)}")
    print(f"  In current list only: {only_in_current}")
    print(f"  In recomputed list only: {only_in_recomputed}")

    # --- 3. Split-stability check (leakage robustness) ---
    stability_rows = []
    split_top_k = {}
    for name, df in [('train', train_df), ('test', test_df)]:
        counter = fcode_counter(df)
        rows = ranked_coverage(counter)
        split_top_k[name] = set(code for _, code, *_ in rows[:STABILITY_CHECK_K])
        k90 = rank_for_coverage(rows, COVERAGE_TARGET)
        stability_rows.append({
            'split': name,
            'n_samples': len(df),
            'n_fcode_instances': sum(counter.values()),
            f'rank_at_{COVERAGE_TARGET:.0f}pct_coverage': k90,
            f'coverage_at_top_{STABILITY_CHECK_K}': round(rows[STABILITY_CHECK_K - 1][4], 2),
        })

    common = split_top_k['train'] & split_top_k['test']
    only_train = sorted(split_top_k['train'] - split_top_k['test'])
    only_test = sorted(split_top_k['test'] - split_top_k['train'])
    print(f"\nSplit-stability check (top-{STABILITY_CHECK_K} F-codes by frequency): "
          f"train vs test overlap {len(common)}/{STABILITY_CHECK_K} "
          f"| only in train: {only_train} | only in test: {only_test}")

    stability_df = pd.DataFrame(stability_rows)
    stability_csv = out_dir / 'split_stability_comparison.csv'
    stability_df.to_csv(stability_csv, index=False)
    print(f"\nSaved: {stability_csv}")

    summary = {
        'current_list_size': len(CURRENT_FCODES),
        'train_rank_at_90pct_coverage': train_k90,
        'train_coverage_at_rank_37': round(train_rows[36][4], 2),
        'current_vs_recomputed_overlap': overlap,
        'current_vs_recomputed_only_in_current': only_in_current,
        'current_vs_recomputed_only_in_recomputed': only_in_recomputed,
        'split_stability_k': STABILITY_CHECK_K,
        'split_stability_train_vs_test_overlap': len(common),
        'split_stability_only_in_train': only_train,
        'split_stability_only_in_test': only_test,
    }
    summary_path = out_dir / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {summary_path}")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
