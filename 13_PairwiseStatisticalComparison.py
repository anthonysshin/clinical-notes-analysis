#!/usr/bin/env python3
"""
13_PairwiseStatisticalComparison.py - Pairwise Strategy Comparison with Correction
(Reviewer 1, Comment 3 / Reviewer 2, Comment 7)

Reviewer 1 noted that the paper claims "Chain-of-Thought and Keyword + CoT form
a distinct upper tier" and separately asserts several other strategy-vs-strategy
differences (e.g. CoT vs. Few-Shot, CoT vs. Rule-Constrained, Few-Shot vs.
Zero-Shot), but only one pair (Keyword + CoT vs. CoT) was ever given a formal
significance test. The rest were asserted from eyeballing overlapping bootstrap
CIs on a chart. With 6 strategies there are 15 possible pairs; testing that many
without correction risks false positives from chance alone.

Reviewer 2 separately noted that the one test that WAS run (Wilcoxon signed-rank
on per-sample F1 scores) does not directly match the effect actually reported
(the aggregate micro-F1 difference).

This script answers both by:
  1. Running a matched test on the exact reported effect (micro-F1 difference)
     for every one of the 15 pairs among the 6 evaluated strategies, using a
     paired bootstrap CI and a paired permutation test (not Wilcoxon on
     per-sample scores, which tests a related but different quantity).
  2. Applying a Holm-Bonferroni correction across the 15 tests, so the
     "distinct upper tier" claim (and any other pairwise claim) is judged
     against a threshold that accounts for testing many pairs at once.

Text handling: reads existing per-sample prediction outputs only; no chunking,
inference, or GPU is involved.

Output:
    outputs/13_PairwiseStatisticalComparison/pairwise_comparison.csv
    outputs/13_PairwiseStatisticalComparison/pairwise_comparison.json

Usage:
    python 13_PairwiseStatisticalComparison.py
"""

import json
import glob
import warnings
from itertools import combinations
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from config import RANDOM_SEED, set_all_seeds, OUTPUT_DIR

warnings.filterwarnings('ignore')
set_all_seeds(RANDOM_SEED)

N_BOOTSTRAP = 2000
N_PERMUTATIONS = 10000
OUTPUT_DIR_13 = f'{OUTPUT_DIR}/13_PairwiseStatisticalComparison'

# The 6 fine-tuned-model strategies compared in Table 2 / Figure 2. Base-model
# (never fine-tuned) conditions 6g1/6g2 are a separate comparison and are not
# part of the "distinct upper tier" claim under review here.
STRATEGIES = [
    ('6a_ZeroShotBaseline', 'Zero-Shot'),
    ('6b_FewShotExemplar', 'Few-Shot'),
    ('6c_RuleConstrained', 'Rule-Constrained'),
    ('6d_ChainOfThought', 'Chain-of-Thought'),
    ('6e_KeywordAugmented', 'Keyword-Extraction'),
    ('6f_KeywordAugmentedCoT', 'Keyword + CoT'),
]


def micro_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * p * r / (p + r) if (p + r) > 0 else 0.0), p, r


def load_aligned(strategy_dir):
    """Load per-sample results indexed by sample_id, for paired comparison."""
    files = sorted(glob.glob(f'outputs/{strategy_dir}/evaluation_results_*.json'))
    if not files:
        return None
    with open(files[-1]) as f:
        j = json.load(f)
    df = pd.DataFrame([s for s in j['sample_results'] if 'tp' in s])
    df = df.set_index('sample_id').sort_index()
    return df[['tp', 'fp', 'fn']], files[-1]


def paired_bootstrap_ci(tpA, fpA, fnA, tpB, fpB, fnB, n_iter=N_BOOTSTRAP, seed=RANDOM_SEED):
    """95% CI for micro-F1(A) - micro-F1(B), resampling the SAME paired indices
    for both strategies each iteration (they share the same test samples)."""
    rng = np.random.default_rng(seed)
    n = len(tpA)
    diffs = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        f1a, _, _ = micro_f1(tpA[idx].sum(), fpA[idx].sum(), fnA[idx].sum())
        f1b, _, _ = micro_f1(tpB[idx].sum(), fpB[idx].sum(), fnB[idx].sum())
        diffs[i] = f1a - f1b
    return np.percentile(diffs, [2.5, 97.5]), diffs


def paired_permutation_pvalue(tpA, fpA, fnA, tpB, fpB, fnB, observed_diff,
                               n_iter=N_PERMUTATIONS, seed=RANDOM_SEED):
    """Two-sided paired permutation test on the micro-F1 difference.

    Null hypothesis: for each test sample, strategy A's and strategy B's
    (tp, fp, fn) outcome are exchangeable (i.e. which strategy produced which
    outcome is arbitrary). Under this null, we randomly swap the A/B label
    per-sample and recompute the micro-F1 difference many times to build a
    null distribution, then see how extreme the real (unswapped) difference is.
    """
    rng = np.random.default_rng(seed)
    n = len(tpA)
    null_diffs = np.empty(n_iter)
    for i in range(n_iter):
        swap = rng.integers(0, 2, n).astype(bool)
        tp1 = np.where(swap, tpB, tpA)
        fp1 = np.where(swap, fpB, fpA)
        fn1 = np.where(swap, fnB, fnA)
        tp2 = np.where(swap, tpA, tpB)
        fp2 = np.where(swap, fpA, fpB)
        fn2 = np.where(swap, fnA, fnB)
        f1a, _, _ = micro_f1(tp1.sum(), fp1.sum(), fn1.sum())
        f1b, _, _ = micro_f1(tp2.sum(), fp2.sum(), fn2.sum())
        null_diffs[i] = f1a - f1b
    p_two_sided = (np.sum(np.abs(null_diffs) >= abs(observed_diff)) + 1) / (n_iter + 1)
    return p_two_sided


def holm_bonferroni(p_values):
    """Holm-Bonferroni step-down correction. Returns adjusted p-values in the
    original order. More powerful than plain Bonferroni while controlling the
    same family-wise error rate (the chance of at least one false positive
    across all tests)."""
    p_values = np.asarray(p_values)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = (m - rank) * p_values[idx]
        running_max = max(running_max, adj)
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def main():
    out_dir = Path(OUTPUT_DIR_13)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PAIRWISE STRATEGY COMPARISON WITH CORRECTION (R1-3 / R2-7)")
    print("=" * 70)

    data = {}
    for strategy_dir, label in STRATEGIES:
        loaded = load_aligned(strategy_dir)
        if loaded is None:
            print(f"  SKIP {strategy_dir}: no evaluation_results_*.json found")
            continue
        df, src_file = loaded
        data[strategy_dir] = {'label': label, 'df': df, 'source_file': src_file}

    labels = {k: v['label'] for k, v in data.items()}
    strategy_keys = list(data.keys())

    # Sanity check: every strategy must be scored on the exact same sample_ids,
    # since this is a paired (not independent-samples) comparison.
    ref_index = data[strategy_keys[0]]['df'].index
    for k in strategy_keys[1:]:
        if not data[k]['df'].index.equals(ref_index):
            raise ValueError(
                f"{k} does not share the same sample_id index as {strategy_keys[0]}; "
                "paired comparison requires identical, aligned test samples."
            )
    print(f"\nConfirmed: all {len(strategy_keys)} strategies scored on the same "
          f"{len(ref_index)} aligned test samples.\n")

    pairs = list(combinations(strategy_keys, 2))
    print(f"Running {len(pairs)} pairwise comparisons "
          f"({N_BOOTSTRAP} bootstrap iters + {N_PERMUTATIONS} permutations each)...\n")

    rows = []
    for a_key, b_key in pairs:
        dfa, dfb = data[a_key]['df'], data[b_key]['df']
        tpA, fpA, fnA = dfa.tp.values, dfa.fp.values, dfa.fn.values
        tpB, fpB, fnB = dfb.tp.values, dfb.fp.values, dfb.fn.values

        f1a, _, _ = micro_f1(tpA.sum(), fpA.sum(), fnA.sum())
        f1b, _, _ = micro_f1(tpB.sum(), fpB.sum(), fnB.sum())
        observed_diff = f1a - f1b

        (ci_lo, ci_hi), _ = paired_bootstrap_ci(tpA, fpA, fnA, tpB, fpB, fnB)
        p_raw = paired_permutation_pvalue(tpA, fpA, fnA, tpB, fpB, fnB, observed_diff)

        rows.append({
            'strategy_a': labels[a_key],
            'strategy_b': labels[b_key],
            'micro_f1_a': round(f1a, 4),
            'micro_f1_b': round(f1b, 4),
            'difference_a_minus_b': round(observed_diff, 4),
            'ci_95_lo': round(ci_lo, 4),
            'ci_95_hi': round(ci_hi, 4),
            'p_raw': p_raw,
        })
        print(f"  {labels[a_key]:20s} vs {labels[b_key]:20s}  "
              f"diff={observed_diff:+.4f}  CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]  p_raw={p_raw:.4f}")

    results_df = pd.DataFrame(rows)
    results_df['p_holm'] = holm_bonferroni(results_df['p_raw'].values)
    results_df['significant_after_correction'] = results_df['p_holm'] < 0.05
    results_df = results_df.sort_values('p_raw').reset_index(drop=True)

    print("\n" + "=" * 70)
    print("RESULTS (sorted by raw p-value, Holm-Bonferroni corrected across "
          f"{len(pairs)} comparisons)")
    print("=" * 70)
    for _, r in results_df.iterrows():
        sig = "SIGNIFICANT" if r['significant_after_correction'] else "not significant"
        print(f"  {r['strategy_a']:20s} vs {r['strategy_b']:20s}  "
              f"diff={r['difference_a_minus_b']:+.4f}  p_raw={r['p_raw']:.4f}  "
              f"p_holm={r['p_holm']:.4f}  [{sig}]")

    csv_path = out_dir / 'pairwise_comparison.csv'
    results_df.to_csv(csv_path, index=False)

    n_sig = int(results_df['significant_after_correction'].sum())
    report = {
        'analysis': 'R1-3 / R2-7 pairwise strategy comparison with Holm-Bonferroni correction',
        'timestamp': datetime.now().isoformat(),
        'random_seed': RANDOM_SEED,
        'n_bootstrap_iterations': N_BOOTSTRAP,
        'n_permutation_iterations': N_PERMUTATIONS,
        'n_strategies': len(strategy_keys),
        'n_pairs_tested': len(pairs),
        'n_significant_after_correction': n_sig,
        'method': (
            'Paired bootstrap 95% CI and paired permutation test (per-sample A/B '
            'swap) on the micro-F1 difference for each pair, matching the effect '
            'actually reported in the manuscript (aggregate micro-F1), rather than '
            'a Wilcoxon signed-rank test on per-sample F1 scores. Holm-Bonferroni '
            'correction applied across all pairs tested.'
        ),
        'pairs': results_df.to_dict(orient='records'),
    }
    with open(out_dir / 'pairwise_comparison.json', 'w') as f:
        json.dump(report, f, indent=2)

    print(f"\n{n_sig} of {len(pairs)} pairs remain significant after correction.")
    print(f"\nOutputs written to: {out_dir}/")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
