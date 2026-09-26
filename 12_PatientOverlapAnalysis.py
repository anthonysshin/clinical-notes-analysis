#!/usr/bin/env python3
"""
12_PatientOverlapAnalysis.py - Patient-Level Leakage Analysis (Reviewer 2, Comment 2)

Reviewer 2 noted that the train/val/test split (Supplementary Table S2) was performed
at the encounter level, so the same patient (subject_id) can appear in more than one
split. This script answers the reviewer's two requests without retraining:

  1. "Report the extent of patient overlap across splits."
     -> Computes patient-level overlap counts/percentages between train/val/test.

  2. "Ideally, repeat the main analyses using patient-disjoint splits. If this is
     not feasible, the potential impact of patient overlap should be clearly
     acknowledged as a limitation."
     -> Full patient-disjoint re-split + retrain was judged infeasible before the
        revision deadline (see reviewer response). Instead, this script re-scores
        every already-completed evaluation strategy (6a-6g2) on the leak-free
        subset of the test set -- the encounters whose patient does NOT appear in
        train -- using the existing per-sample prediction outputs. This is a valid
        patient-disjoint evaluation for those encounters, at no retraining cost.
     -> Also reports the F1 gap (patient-overlapping vs. patient-disjoint test
        encounters) with a paired bootstrap 95% CI, for each strategy, and a
        patient-clustered bootstrap CI for the headline Keyword+CoT result (the
        encounter-level bootstrap used elsewhere in the paper is a mild
        underestimate of true uncertainty when patients contribute >1 encounter).

Text handling: this script only reads existing data/eval outputs; no chunking,
inference, or GPU is involved.

Output:
    outputs/12_PatientOverlapAnalysis/patient_overlap_stats.json
    outputs/12_PatientOverlapAnalysis/per_strategy_overlap_gap.csv
    outputs/12_PatientOverlapAnalysis/leak_free_vs_reported_summary.csv
    outputs/12_PatientOverlapAnalysis/patient_overlap_gap.png

Usage:
    python 12_PatientOverlapAnalysis.py
"""

import json
import glob
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch

from config import (
    RANDOM_SEED, set_all_seeds,
    TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH,
    OUTPUT_DIR_12_PATIENT_OVERLAP,
    STRATEGY_COLORS, apply_figure_style,
)

warnings.filterwarnings('ignore')
set_all_seeds(RANDOM_SEED)
apply_figure_style()

N_BOOTSTRAP = 2000

# Evaluation strategies to re-score, in the order reported in the paper.
# (dir_name, display_label)
STRATEGIES = [
    ('6a_ZeroShotBaseline', 'Zero-Shot'),
    ('6b_FewShotExemplar', 'Few-Shot'),
    ('6c_RuleConstrained', 'Rule-Constrained'),
    ('6d_ChainOfThought', 'Chain-of-Thought'),
    ('6e_KeywordAugmented', 'Keyword-Extraction'),
    ('6f_KeywordAugmentedCoT', 'Keyword + CoT'),
    ('6g1_BaseModel_ZeroShot', 'Base Model (Zero-Shot)'),
    ('6g2_BaseModel_CoT', 'Base Model (CoT)'),
]


def micro_f1(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * p * r / (p + r) if (p + r) > 0 else 0.0), p, r


def load_split_patients():
    """Load subject_id / hadm_id membership for each split."""
    train = pd.read_csv(TRAIN_DATA_PATH, usecols=['hadm_id', 'subject_id'])
    val = pd.read_csv(VAL_DATA_PATH, usecols=['hadm_id', 'subject_id'])
    test = pd.read_csv(TEST_DATA_PATH, usecols=['hadm_id', 'subject_id'])
    return train, val, test


def compute_overlap_stats(train, val, test):
    train_p, val_p, test_p = set(train.subject_id), set(val.subject_id), set(test.subject_id)

    test_in_train_enc = test.subject_id.isin(train_p).sum()
    val_in_train_enc = val.subject_id.isin(train_p).sum()

    counts = pd.concat([train, val, test]).subject_id.value_counts()

    stats = {
        'n_encounters': {'train': len(train), 'val': len(val), 'test': len(test)},
        'n_unique_patients': {'train': len(train_p), 'val': len(val_p), 'test': len(test_p)},
        'shared_patients': {
            'train_val': len(train_p & val_p),
            'train_test': len(train_p & test_p),
            'val_test': len(val_p & test_p),
        },
        'test_patients_also_in_train': {
            'n_patients': len(test_p & train_p),
            'pct_of_test_patients': round(100 * len(test_p & train_p) / len(test_p), 1),
        },
        'test_encounters_with_patient_in_train': {
            'n_encounters': int(test_in_train_enc),
            'pct_of_test_encounters': round(100 * test_in_train_enc / len(test), 1),
        },
        'val_encounters_with_patient_in_train': {
            'n_encounters': int(val_in_train_enc),
            'pct_of_val_encounters': round(100 * val_in_train_enc / len(val), 1),
        },
        'patients_with_multiple_encounters': {
            'n_patients': int((counts > 1).sum()),
            'pct_of_all_patients': round(100 * (counts > 1).mean(), 1),
            'mean_encounters_per_patient': round(counts.mean(), 2),
            'median_encounters_per_patient': float(counts.median()),
            'max_encounters_for_one_patient': int(counts.max()),
        },
    }
    return stats


def load_latest_eval(strategy_dir):
    files = sorted(glob.glob(f'outputs/{strategy_dir}/evaluation_results_*.json'))
    if not files:
        return None
    with open(files[-1]) as f:
        j = json.load(f)
    df = pd.DataFrame([s for s in j['sample_results'] if 'tp' in s and 'subject_id' in s])
    return df, files[-1]


def bootstrap_gap_ci(df, train_patients, n_iter=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Paired bootstrap 95% CI for (F1 | patient in train) - (F1 | patient not in train).
    Both quantities are computed on the SAME resampled draw over the subgroup-only
    rows (overlap-only vs. disjoint-only), so this answers: 'does the model do
    worse on patients it never saw, compared to patients with repeat encounters?'"""
    rng = np.random.default_rng(seed)
    ov = df.subject_id.isin(train_patients).values
    tp, fp, fn = df.tp.values, df.fp.values, df.fn.values
    n = len(df)
    gaps = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        m = ov[idx]
        f1_in, _, _ = micro_f1(tp[idx][m].sum(), fp[idx][m].sum(), fn[idx][m].sum())
        f1_out, _, _ = micro_f1(tp[idx][~m].sum(), fp[idx][~m].sum(), fn[idx][~m].sum())
        gaps[i] = f1_in - f1_out
    return np.percentile(gaps, [2.5, 97.5])


def bootstrap_reported_vs_disjoint_ci(df, train_patients, n_iter=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Paired bootstrap 95% CI for (reported F1, full test set) - (F1, disjoint-only
    subset), both computed from the SAME resample of all n rows. This is the
    quantity actually quoted in the manuscript/response letter (e.g. 0.675 to
    0.643 for Keyword + CoT), so its CI must be computed this way, not via
    bootstrap_gap_ci (which answers a different question -- see above)."""
    rng = np.random.default_rng(seed)
    disjoint_mask = ~df.subject_id.isin(train_patients).values
    tp, fp, fn = df.tp.values, df.fp.values, df.fn.values
    n = len(df)
    gaps = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        f1_reported, _, _ = micro_f1(tp[idx].sum(), fp[idx].sum(), fn[idx].sum())
        d = disjoint_mask[idx]
        f1_disjoint, _, _ = micro_f1(tp[idx][d].sum(), fp[idx][d].sum(), fn[idx][d].sum())
        gaps[i] = f1_reported - f1_disjoint
    return np.percentile(gaps, [2.5, 97.5])


def patient_clustered_bootstrap_ci(df, n_iter=N_BOOTSTRAP, seed=RANDOM_SEED):
    """Cluster bootstrap: resample PATIENTS (not encounters) with replacement.
    More conservative than the encounter-level bootstrap used elsewhere in the
    paper when patients contribute more than one test encounter."""
    rng = np.random.default_rng(seed)
    groups = df.groupby('subject_id')[['tp', 'fp', 'fn']].sum()
    patients = groups.index.values
    tp, fp, fn = groups.tp.values, groups.fp.values, groups.fn.values
    n = len(patients)
    f1s = np.empty(n_iter)
    for i in range(n_iter):
        idx = rng.integers(0, n, n)
        f1s[i], _, _ = micro_f1(tp[idx].sum(), fp[idx].sum(), fn[idx].sum())
    return np.percentile(f1s, [2.5, 97.5])


def main():
    out_dir = Path(OUTPUT_DIR_12_PATIENT_OVERLAP)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PATIENT OVERLAP ANALYSIS (Reviewer 2, Comment 2)")
    print("=" * 70)

    train, val, test = load_split_patients()
    train_patients = set(train.subject_id)

    print("\n[1/3] Computing patient overlap across splits...")
    overlap_stats = compute_overlap_stats(train, val, test)
    print(json.dumps(overlap_stats, indent=2))

    print("\n[2/3] Re-scoring each strategy on patient-disjoint test encounters...")
    rows = []
    for strategy_dir, label in STRATEGIES:
        loaded = load_latest_eval(strategy_dir)
        if loaded is None:
            print(f"  SKIP {strategy_dir}: no evaluation_results_*.json found")
            continue
        df, src_file = loaded
        df['patient_in_train'] = df.subject_id.isin(train_patients)

        overall_f1, overall_p, overall_r = micro_f1(df.tp.sum(), df.fp.sum(), df.fn.sum())

        disjoint = df[~df.patient_in_train]
        overlap = df[df.patient_in_train]
        f1_disjoint, p_disjoint, r_disjoint = micro_f1(disjoint.tp.sum(), disjoint.fp.sum(), disjoint.fn.sum())
        f1_overlap, p_overlap, r_overlap = micro_f1(overlap.tp.sum(), overlap.fp.sum(), overlap.fn.sum())

        em_disjoint = disjoint.perfect_match.mean() if len(disjoint) else float('nan')
        em_overlap = overlap.perfect_match.mean() if len(overlap) else float('nan')

        # Two distinct comparisons, each with its own matching CI:
        #  (1) overlap-only vs. disjoint-only patients (bigger gap)
        #  (2) reported (full test set) vs. disjoint-only (smaller gap -- this is
        #      the number actually quoted in the manuscript, e.g. 0.675 to 0.643)
        ci_lo, ci_hi = bootstrap_gap_ci(df, train_patients)
        rep_ci_lo, rep_ci_hi = bootstrap_reported_vs_disjoint_ci(df, train_patients)

        rows.append({
            'strategy': strategy_dir,
            'label': label,
            'n_total': len(df),
            'n_patient_disjoint': len(disjoint),
            'n_patient_overlap': len(overlap),
            'micro_f1_reported_full_test': round(overall_f1, 4),
            'micro_f1_patient_disjoint_only': round(f1_disjoint, 4),
            'micro_f1_patient_overlap_only': round(f1_overlap, 4),
            'gap_overlap_minus_disjoint': round(f1_overlap - f1_disjoint, 4),
            'gap_95ci_lo': round(ci_lo, 4),
            'gap_95ci_hi': round(ci_hi, 4),
            'gap_reported_minus_disjoint': round(overall_f1 - f1_disjoint, 4),
            'gap_reported_minus_disjoint_95ci_lo': round(rep_ci_lo, 4),
            'gap_reported_minus_disjoint_95ci_hi': round(rep_ci_hi, 4),
            'exact_match_patient_disjoint_only': round(em_disjoint, 4),
            'exact_match_patient_overlap_only': round(em_overlap, 4),
            'source_file': src_file,
        })
        print(f"  {label:26s} full={overall_f1:.4f}  disjoint={f1_disjoint:.4f}  "
              f"overlap={f1_overlap:.4f}  overlap-vs-disjoint gap={f1_overlap - f1_disjoint:+.4f} "
              f"[{ci_lo:+.4f}, {ci_hi:+.4f}]  reported-vs-disjoint gap={overall_f1 - f1_disjoint:+.4f} "
              f"[{rep_ci_lo:+.4f}, {rep_ci_hi:+.4f}]")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(out_dir / 'per_strategy_overlap_gap.csv', index=False)

    print("\n[3/3] Patient-clustered bootstrap CI for the headline strategy (Keyword + CoT)...")
    best = load_latest_eval('6f_KeywordAugmentedCoT')
    clustered_summary = {}
    if best is not None:
        df_best, _ = best
        enc_f1, _, _ = micro_f1(df_best.tp.sum(), df_best.fp.sum(), df_best.fn.sum())
        rng = np.random.default_rng(RANDOM_SEED)
        n = len(df_best)
        tp, fp, fn = df_best.tp.values, df_best.fp.values, df_best.fn.values
        enc_f1s = np.empty(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            idx = rng.integers(0, n, n)
            enc_f1s[i], _, _ = micro_f1(tp[idx].sum(), fp[idx].sum(), fn[idx].sum())
        enc_ci = np.percentile(enc_f1s, [2.5, 97.5])
        patient_ci = patient_clustered_bootstrap_ci(df_best)
        clustered_summary = {
            'strategy': '6f_KeywordAugmentedCoT (Keyword + CoT)',
            'point_estimate_micro_f1': round(enc_f1, 4),
            'encounter_level_bootstrap_95ci': [round(x, 4) for x in enc_ci],
            'patient_clustered_bootstrap_95ci': [round(x, 4) for x in patient_ci],
            'note': 'Patient-clustered CI resamples patients rather than encounters; '
                    'expected to be wider when patients contribute >1 test encounter.',
        }
        print(json.dumps(clustered_summary, indent=2))

    # Combined output
    full_report = {
        'analysis': 'R2-2 patient overlap / patient-disjoint re-scoring',
        'timestamp': datetime.now().isoformat(),
        'random_seed': RANDOM_SEED,
        'n_bootstrap_iterations': N_BOOTSTRAP,
        'overlap_stats': overlap_stats,
        'clustered_bootstrap_headline_strategy': clustered_summary,
    }
    with open(out_dir / 'patient_overlap_stats.json', 'w') as f:
        json.dump(full_report, f, indent=2)

    # Summary table: reported (full test) vs. patient-disjoint-only micro-F1
    summary_rows = []
    for r in rows:
        summary_rows.append({
            'Strategy': r['label'],
            'Reported micro-F1 (full test, n=%d)' % r['n_total']: r['micro_f1_reported_full_test'],
            'Patient-disjoint micro-F1 (n=%d)' % r['n_patient_disjoint']: r['micro_f1_patient_disjoint_only'],
            'Difference': r['gap_reported_minus_disjoint'],
            'Difference 95% CI lo': r['gap_reported_minus_disjoint_95ci_lo'],
            'Difference 95% CI hi': r['gap_reported_minus_disjoint_95ci_hi'],
        })
    pd.DataFrame(summary_rows).to_csv(out_dir / 'leak_free_vs_reported_summary.csv', index=False)

    # Figure: reported vs. patient-disjoint micro-F1 per strategy
    if rows:
        fig, ax = plt.subplots(figsize=(10, 6))
        labels = [r['label'] for r in rows]
        reported = [r['micro_f1_reported_full_test'] for r in rows]
        disjoint = [r['micro_f1_patient_disjoint_only'] for r in rows]
        x = np.arange(len(labels))
        width = 0.35
        bar_colors = [STRATEGY_COLORS.get(r['strategy'], '#B0B0B0') for r in rows]
        # Bake transparency into the FACE color only (not passed as `alpha=`), so the
        # black edge stays fully opaque and identical for both groups. `alpha=` on
        # ax.bar() dims the whole patch including its edge, which made the
        # "Reported" bars' outline render as a lighter gray than the "Patient-disjoint"
        # bars' solid black outline even though both used edgecolor='black'.
        reported_facecolors = [mcolors.to_rgba(c, alpha=0.55) for c in bar_colors]
        # Bar color encodes STRATEGY (matches the palette used elsewhere in the paper).
        # Group membership (full test set vs. unseen-patients-only) is encoded
        # separately via edge style (dotted vs. solid), so the legend can show that
        # distinction without implying a color meaning it doesn't have.
        ax.bar(x - width / 2, reported, width,
               color=reported_facecolors, edgecolor='black', linewidth=0.9, linestyle='-')
        ax.bar(x + width / 2, disjoint, width,
               color=bar_colors, edgecolor='black', linewidth=0.9, linestyle=':')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha='right')
        ax.set_ylabel('Micro-F1')
        ax.set_title('Full Test Set vs. Unseen-Patient Micro-F1 by Strategy')
        legend_handles = [
            Patch(facecolor='white', edgecolor='black', linewidth=1.2, linestyle='-',
                  label='Full test set'),
            Patch(facecolor='white', edgecolor='black', linewidth=1.2, linestyle=':',
                  label='Unseen patients only'),
        ]
        ax.legend(handles=legend_handles)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        fig_path = out_dir / 'patient_overlap_gap.png'
        plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\nFigure saved: {fig_path}")
        plt.close()

    print("\n" + "=" * 70)
    print(f"Done. Outputs written to: {out_dir}/")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    exit(main())
