#!/usr/bin/env python3
"""
9_ErrorAnalysis.py - Error Analysis and Confusion Matrix Generation

This script performs detailed error analysis on model predictions, including:
    1. False positive and false negative analysis
    2. Error categorization (perfect match, partial match, complete miss)
    3. Confusion matrix for top F-codes
    4. Common error patterns identification
    5. Performance by code-frequency tier (Reviewer 1, Comment 6): whether
       rare codes are systematically missed, broken down by how many
       ground-truth test instances each code has
    6. Chunk-count sensitivity (Reviewer 2, Comment 3): whether performance
       differs between single-chunk documents (no label noise possible)
       and multi-chunk documents (each chunk inherits the full
       document-level label set, so labels can be mismatched with a given
       chunk's content)

Output:
    - Error analysis report (JSON)
    - Confusion matrix visualization (PNG)
    - Error breakdown tables (CSV)
    - frequency_tier_breakdown.csv, chunk_sensitivity.csv

Usage:
    python 9_ErrorAnalysis.py --results-dir ./outputs --output-dir ./outputs/9_ErrorAnalysis

Author: Clinical Note Analysis Study
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from collections import Counter, defaultdict
import warnings

# Import centralized config for colors and style
from config import (
    COLORS, FIGURE_STYLE, apply_figure_style, get_category_color,
    OUTPUT_DIR, OUTPUT_DIR_6F, OUTPUT_DIR_6D, OUTPUT_DIR_9_ERROR
)

warnings.filterwarnings('ignore')

# Apply publication-ready figure style
apply_figure_style()
sns.set_style("white")  # Clean background, horizontal gridlines added manually


class ErrorAnalyzer:
    """Perform detailed error analysis on model predictions."""

    def __init__(self, results_dir: str, output_dir: str):
        """
        Initialize error analyzer.

        Args:
            results_dir: Directory containing evaluation results
            output_dir: Directory to save error analysis outputs
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = None
        self.sample_results = []
        self.predictions_df = None

    def load_results(self):
        """Load evaluation results."""
        print("=" * 70)
        print("LOADING EVALUATION RESULTS")
        print("=" * 70)

        # Find evaluation results - search in subdirectories if not found directly
        json_files = list(self.results_dir.glob("evaluation_results_*.json"))

        if not json_files:
            # Search in subdirectories (e.g., 6a_ZeroShotBaseline/, 6d_ChainOfThought/)
            json_files = list(self.results_dir.glob("*/evaluation_results_*.json"))

        if not json_files:
            raise FileNotFoundError(f"No evaluation results found in {self.results_dir}")

        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_json = json_files[0]

        print(f"\nLoading: {latest_json.name}")
        with open(latest_json, 'r') as f:
            self.results = json.load(f)

        self.sample_results = self.results.get('sample_results', [])
        print(f"Loaded {len(self.sample_results)} sample results")

        # Load predictions CSV if available (check same directory as JSON file)
        json_dir = latest_json.parent
        csv_files = list(json_dir.glob("predictions_*.csv"))
        if not csv_files:
            csv_files = list(self.results_dir.glob("*/predictions_*.csv"))
        if csv_files:
            csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            self.predictions_df = pd.read_csv(csv_files[0])
            print(f"Loaded predictions: {csv_files[0].name}")

    def categorize_errors(self) -> Dict:
        """
        Categorize predictions into error types.

        Returns:
            Dictionary with categorized samples
        """
        print("\n" + "=" * 70)
        print("ERROR CATEGORIZATION")
        print("=" * 70)

        categories = {
            'perfect_match': [],      # F1 = 1.0
            'partial_match': [],      # 0 < F1 < 1
            'complete_miss': [],      # F1 = 0, had actual codes
            'false_positive_only': [] # Predicted codes when none existed
        }

        for result in self.sample_results:
            f1 = result.get('f1', 0)
            actual = result.get('actual_codes', [])
            predicted = result.get('predicted_codes', [])

            if f1 == 1.0:
                categories['perfect_match'].append(result)
            elif f1 > 0:
                categories['partial_match'].append(result)
            elif len(actual) == 0 and len(predicted) > 0:
                categories['false_positive_only'].append(result)
            else:
                categories['complete_miss'].append(result)

        # Print summary
        total = len(self.sample_results)
        print(f"\nError Category Breakdown:")
        print(f"  Perfect Match:       {len(categories['perfect_match']):>4} "
              f"({len(categories['perfect_match']) / total * 100:>5.1f}%)")
        print(f"  Partial Match:       {len(categories['partial_match']):>4} "
              f"({len(categories['partial_match']) / total * 100:>5.1f}%)")
        print(f"  Complete Miss:       {len(categories['complete_miss']):>4} "
              f"({len(categories['complete_miss']) / total * 100:>5.1f}%)")
        print(f"  False Positive Only: {len(categories['false_positive_only']):>4} "
              f"({len(categories['false_positive_only']) / total * 100:>5.1f}%)")

        return categories

    def analyze_false_positives(self) -> Tuple[Counter, List[Dict]]:
        """
        Analyze false positive predictions.

        Returns:
            Tuple of (FP counter by code, detailed FP list)
        """
        print("\n" + "=" * 70)
        print("FALSE POSITIVE ANALYSIS")
        print("=" * 70)

        fp_counter = Counter()
        fp_details = []

        for result in self.sample_results:
            actual = set(result.get('actual_codes', []))
            predicted = set(result.get('predicted_codes', []))

            false_positives = predicted - actual

            for fp_code in false_positives:
                fp_counter[fp_code] += 1
                fp_details.append({
                    'sample_id': result.get('sample_id'),
                    'fp_code': fp_code,
                    'actual_codes': list(actual),
                    'predicted_codes': list(predicted)
                })

        # Print top false positives
        print(f"\nTop 15 Most Common False Positives:")
        print(f"{'Rank':<6} {'F-Code':<12} {'Count':<8} {'% of Samples'}")
        print("-" * 50)

        for rank, (code, count) in enumerate(fp_counter.most_common(15), 1):
            pct = count / len(self.sample_results) * 100
            print(f"{rank:<6} {code:<12} {count:<8} {pct:.1f}%")

        return fp_counter, fp_details

    def analyze_false_negatives(self) -> Tuple[Counter, List[Dict]]:
        """
        Analyze false negative predictions (missed codes).

        Returns:
            Tuple of (FN counter by code, detailed FN list)
        """
        print("\n" + "=" * 70)
        print("FALSE NEGATIVE ANALYSIS")
        print("=" * 70)

        fn_counter = Counter()
        fn_details = []

        for result in self.sample_results:
            actual = set(result.get('actual_codes', []))
            predicted = set(result.get('predicted_codes', []))

            false_negatives = actual - predicted

            for fn_code in false_negatives:
                fn_counter[fn_code] += 1
                fn_details.append({
                    'sample_id': result.get('sample_id'),
                    'fn_code': fn_code,
                    'actual_codes': list(actual),
                    'predicted_codes': list(predicted)
                })

        # Print top false negatives
        print(f"\nTop 15 Most Commonly Missed F-Codes:")
        print(f"{'Rank':<6} {'F-Code':<12} {'Count':<8} {'% of Samples'}")
        print("-" * 50)

        for rank, (code, count) in enumerate(fn_counter.most_common(15), 1):
            pct = count / len(self.sample_results) * 100
            print(f"{rank:<6} {code:<12} {count:<8} {pct:.1f}%")

        return fn_counter, fn_details

    def analyze_confusion_patterns(self) -> Dict:
        """
        Analyze confusion patterns between F-codes.

        Returns:
            Dictionary mapping actual codes to predicted substitutes
        """
        print("\n" + "=" * 70)
        print("CONFUSION PATTERN ANALYSIS")
        print("=" * 70)

        confusion = defaultdict(Counter)

        for result in self.sample_results:
            actual = set(result.get('actual_codes', []))
            predicted = set(result.get('predicted_codes', []))

            # For missed codes, see what was predicted instead
            missed = actual - predicted
            wrong = predicted - actual

            for missed_code in missed:
                for wrong_code in wrong:
                    confusion[missed_code][wrong_code] += 1

        # Find top confusions
        top_confusions = []
        for actual_code, predicted_codes in confusion.items():
            for predicted_code, count in predicted_codes.items():
                if count >= 2:  # At least 2 occurrences
                    top_confusions.append({
                        'actual': actual_code,
                        'predicted_instead': predicted_code,
                        'count': count
                    })

        top_confusions.sort(key=lambda x: x['count'], reverse=True)

        print(f"\nTop 15 Confusion Patterns (Actual -> Predicted Instead):")
        print(f"{'Rank':<6} {'Actual':<12} {'Predicted':<12} {'Count'}")
        print("-" * 50)

        for rank, conf in enumerate(top_confusions[:15], 1):
            print(f"{rank:<6} {conf['actual']:<12} {conf['predicted_instead']:<12} "
                  f"{conf['count']}")

        return dict(confusion)

    def analyze_by_frequency_tier(self) -> Dict:
        """
        Break down per-code performance by ground-truth frequency tier
        (Reviewer 1, Comment 6): are rare codes systematically missed, and
        does performance improve smoothly with frequency or drop off sharply
        below some threshold?

        The rarest tier boundary (1-5 occurrences) matches the "100 codes
        (74%) had five or fewer test instances" statement reported
        elsewhere in the manuscript for the same population.

        Returns:
            Dictionary with per-tier code counts, aggregate (micro) F1,
            mean per-code F1, and the codes scoring F1 = 0.000 in each tier.
        """
        print("\n" + "=" * 70)
        print("PERFORMANCE BY CODE-FREQUENCY TIER")
        print("=" * 70)

        per_code = self.results.get('performance_metrics', {}).get('per_code_metrics', {})
        if not per_code:
            print("No per-code metrics found in evaluation results - skipping.")
            return {}

        # Codes with occurrences == 0 are predicted but never actually true
        # (false-positive-only codes) and have no ground-truth frequency to
        # bucket, so they are excluded here entirely, matching the "135
        # codes with at least one true instance" population already used
        # elsewhere in the manuscript for macro-F1 and per-code analysis.
        tiers = [
            ('Rare (1-5)', 1, 5),
            ('Uncommon (6-20)', 6, 20),
            ('Moderate (21-100)', 21, 100),
            ('Common (>100)', 101, float('inf')),
        ]

        tier_results = []
        for tier_name, lo, hi in tiers:
            codes_in_tier = {
                code: m for code, m in per_code.items()
                if lo <= m.get('occurrences', 0) <= hi
            }
            if not codes_in_tier:
                continue

            tp = sum(m['tp'] for m in codes_in_tier.values())
            fp = sum(m['fp'] for m in codes_in_tier.values())
            fn = sum(m['fn'] for m in codes_in_tier.values())
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            micro_f1 = (2 * precision * recall / (precision + recall)
                        if (precision + recall) > 0 else 0.0)
            f1_scores = [m['f1'] for m in codes_in_tier.values()]
            mean_f1 = float(np.mean(f1_scores))
            zero_f1_codes = sorted(
                [code for code, m in codes_in_tier.items() if m['f1'] == 0.0]
            )

            tier_results.append({
                'tier': tier_name,
                'num_codes': len(codes_in_tier),
                'total_occurrences': sum(m['occurrences'] for m in codes_in_tier.values()),
                'micro_precision': round(precision, 4),
                'micro_recall': round(recall, 4),
                'micro_f1': round(micro_f1, 4),
                'mean_per_code_f1': round(mean_f1, 4),
                'num_zero_f1_codes': len(zero_f1_codes),
                'zero_f1_codes': zero_f1_codes,
            })

        print(f"\n{'Tier':<20} {'Codes':<8} {'Occurrences':<13} {'Micro F1':<10} {'Mean F1':<10} {'F1=0 codes'}")
        print("-" * 85)
        for t in tier_results:
            print(f"{t['tier']:<20} {t['num_codes']:<8} {t['total_occurrences']:<13} "
                  f"{t['micro_f1']:<10.4f} {t['mean_per_code_f1']:<10.4f} {t['num_zero_f1_codes']}")

        # Save CSV (without the full zero_f1_codes list, for a compact table)
        csv_rows = [{k: v for k, v in t.items() if k != 'zero_f1_codes'} for t in tier_results]
        csv_path = self.output_dir / 'frequency_tier_breakdown.csv'
        pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

        return {'tiers': tier_results}

    def analyze_chunk_sensitivity(self, results_dir: str = None) -> Dict:
        """
        Compare performance on single-chunk vs. multi-chunk samples
        (Reviewer 2, Comment 3): every training chunk inherits the full
        document-level label set, so a chunk may be trained against labels
        it contains no textual evidence for. This label noise can only
        occur in multi-chunk documents; a single-chunk document's one chunk
        is the whole note, so its labels are never mismatched with its
        content. A performance gap between the two groups is therefore
        consistent with (though not sole proof of) sensitivity to this
        label-assignment strategy.

        This check is only meaningful for a strategy that actually uses
        the overlapping-chunking text handling. It defaults to the
        Chain-of-Thought strategy (6d) rather than this script's usual
        default (Keyword + CoT, 6f), because Keyword + CoT uses keyword
        extraction to fit each document in a single pass and so has no
        multi-chunk samples at all.

        Args:
            results_dir: Directory to load predictions from. Defaults to
                the Chain-of-Thought (chunking) strategy, independent of
                whatever --results-dir this script was run with.

        Returns:
            Dictionary with per-group sample counts, micro-F1, and a
            bootstrap 95% CI for the single-chunk minus multi-chunk gap.
        """
        print("\n" + "=" * 70)
        print("CHUNK-COUNT SENSITIVITY (SINGLE- VS. MULTI-CHUNK SAMPLES)")
        print("=" * 70)

        results_dir = Path(results_dir) if results_dir else Path(OUTPUT_DIR_6D)
        csv_files = sorted(results_dir.glob("predictions_*.csv"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        if not csv_files:
            print(f"No predictions CSV found in {results_dir} - skipping.")
            return {}
        print(f"Loading: {csv_files[0]}")
        df = pd.read_csv(csv_files[0])

        if 'num_chunks' not in df.columns:
            print("Predictions CSV has no num_chunks column - skipping.")
            return {}
        if (df['num_chunks'] > 1).sum() == 0:
            print(f"No multi-chunk samples found in {csv_files[0].name} - "
                  "this strategy does not use overlapping chunking. Skipping.")
            return {}
        single = df[df['num_chunks'] == 1]
        multi = df[df['num_chunks'] > 1]
        if len(single) == 0:
            print(f"No single-chunk samples found in {csv_files[0].name} - skipping.")
            return {}

        def micro_f1_group(group: pd.DataFrame) -> Tuple[float, int, int, int]:
            tp, fp, fn = group['tp'].sum(), group['fp'].sum(), group['fn'].sum()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            return f1, int(tp), int(fp), int(fn)

        single_f1, s_tp, s_fp, s_fn = micro_f1_group(single)
        multi_f1, m_tp, m_fp, m_fn = micro_f1_group(multi)
        observed_gap = single_f1 - multi_f1

        # Independent-groups bootstrap (each group resampled from itself,
        # since single- and multi-chunk samples are different documents,
        # not a paired comparison).
        rng = np.random.default_rng(42)
        n_iter = 2000
        gaps = np.empty(n_iter)
        single_tp, single_fp, single_fn = single['tp'].values, single['fp'].values, single['fn'].values
        multi_tp, multi_fp, multi_fn = multi['tp'].values, multi['fp'].values, multi['fn'].values
        for i in range(n_iter):
            s_idx = rng.integers(0, len(single_tp), len(single_tp))
            m_idx = rng.integers(0, len(multi_tp), len(multi_tp))
            s_f1, _, _, _ = micro_f1_group(pd.DataFrame({
                'tp': single_tp[s_idx], 'fp': single_fp[s_idx], 'fn': single_fn[s_idx]
            }))
            m_f1, _, _, _ = micro_f1_group(pd.DataFrame({
                'tp': multi_tp[m_idx], 'fp': multi_fp[m_idx], 'fn': multi_fn[m_idx]
            }))
            gaps[i] = s_f1 - m_f1
        ci_lo, ci_hi = np.percentile(gaps, [2.5, 97.5])

        print(f"\nSingle-chunk samples: n={len(single)}, Micro F1={single_f1:.4f}")
        print(f"Multi-chunk samples:  n={len(multi)}, Micro F1={multi_f1:.4f}")
        print(f"Gap (single - multi): {observed_gap:+.4f}  95% CI: [{ci_lo:+.4f}, {ci_hi:+.4f}]")

        result = {
            'single_chunk': {'n_samples': len(single), 'micro_f1': round(single_f1, 4)},
            'multi_chunk': {'n_samples': len(multi), 'micro_f1': round(multi_f1, 4)},
            'gap_single_minus_multi': round(observed_gap, 4),
            'gap_95ci': [round(ci_lo, 4), round(ci_hi, 4)],
        }

        csv_path = self.output_dir / 'chunk_sensitivity.csv'
        pd.DataFrame([
            {'group': 'Single-chunk', 'n_samples': len(single), 'micro_f1': round(single_f1, 4)},
            {'group': 'Multi-chunk', 'n_samples': len(multi), 'micro_f1': round(multi_f1, 4)},
        ]).to_csv(csv_path, index=False)
        print(f"\nSaved: {csv_path}")

        return result

    def create_confusion_matrix(self, top_n: int = 10):
        """
        Create a multi-label prediction co-occurrence matrix for the top N F-codes.

        In a multi-label classification setting, a traditional confusion matrix
        is not strictly defined because each sample can have multiple true and
        predicted labels without a one-to-one correspondence. Instead, this
        method produces a co-occurrence matrix with the following semantics:

        - Diagonal cell (i, i): Count of samples where code i was correctly
          predicted (true positive).
        - Off-diagonal cell (i, j), i != j: Count of samples where code i was
          in the ground truth AND code j was a false positive (predicted but
          not in the ground truth) in the same sample.

        Off-diagonal cells do NOT represent pairwise misclassifications.
        They reflect co-occurrence patterns between true labels and false
        positives, providing clinically interpretable insight into which
        codes the model tends to over-predict when a given condition is present.

        Args:
            top_n: Number of top codes (by frequency in ground truth) to include.
        """
        print("\n" + "=" * 70)
        print("CREATING PREDICTION CO-OCCURRENCE MATRIX")
        print("=" * 70)

        # Count code occurrences
        all_codes = Counter()
        for result in self.sample_results:
            all_codes.update(result.get('actual_codes', []))

        top_codes = [code for code, _ in all_codes.most_common(top_n)]

        # Build co-occurrence matrix
        matrix = np.zeros((len(top_codes), len(top_codes)), dtype=int)
        code_to_idx = {code: i for i, code in enumerate(top_codes)}

        for result in self.sample_results:
            actual = set(result.get('actual_codes', []))
            predicted = set(result.get('predicted_codes', []))

            # Identify false positives in this sample (predicted but not actual)
            false_positives = predicted - actual

            for actual_code in actual:
                if actual_code in code_to_idx:
                    actual_idx = code_to_idx[actual_code]

                    # Diagonal: count true positives
                    if actual_code in predicted:
                        matrix[actual_idx, actual_idx] += 1

                    # Off-diagonal: count co-occurring false positives
                    # For every true label in this sample, record each false
                    # positive that appeared alongside it in the predictions.
                    for fp_code in false_positives:
                        if fp_code in code_to_idx:
                            fp_idx = code_to_idx[fp_code]
                            matrix[actual_idx, fp_idx] += 1

        # Create visualization (using colorblind-safe colormap)
        fig, ax = plt.subplots(figsize=(12, 10))

        im = ax.imshow(matrix, cmap=COLORS['sequential_cmap'], aspect='auto')

        ax.set_xticks(np.arange(len(top_codes)))
        ax.set_yticks(np.arange(len(top_codes)))
        ax.set_xticklabels(top_codes, rotation=45, ha='right')
        ax.set_yticklabels(top_codes)

        # Add text annotations
        for i in range(len(top_codes)):
            for j in range(len(top_codes)):
                if matrix[i, j] > 0:
                    text_color = 'white' if matrix[i, j] > matrix.max() / 2 else 'black'
                    ax.text(j, i, f'{matrix[i, j]}',
                            ha='center', va='center', color=text_color, fontsize=9)

        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title(f'Prediction Co-Occurrence Matrix (Top {top_n} F-Codes)\n'
                     f'Diagonal = True Positives; Off-Diagonal = False-Positive Co-Occurrences',
                     fontsize=14, fontweight='bold', pad=20)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Count', fontsize=11)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig1_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig1_confusion_matrix.pdf', bbox_inches='tight')
        print(f"\n  Saved: fig1_confusion_matrix.png/pdf")
        plt.close()

        return matrix, top_codes

    def create_error_visualizations(self, fp_counter: Counter, fn_counter: Counter):
        """
        Create visualizations for error analysis.

        Args:
            fp_counter: False positive counts by code
            fn_counter: False negative counts by code
        """
        print("\n  Creating error visualizations...")

        # Figure 2: FP vs FN comparison for top codes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Top 15 FP
        top_fp = fp_counter.most_common(15)
        if top_fp:
            codes = [c for c, _ in top_fp]
            counts = [cnt for _, cnt in top_fp]
            ax1.barh(range(len(codes)), counts, color=COLORS['false_positive'], alpha=0.8)
            ax1.set_yticks(range(len(codes)))
            ax1.set_yticklabels(codes)
            ax1.set_xlabel('Count', fontsize=11, fontweight='bold')
            ax1.set_title('Top 15 False Positives\n(Over-predicted)', fontsize=13, fontweight='bold')
            ax1.invert_yaxis()

            for i, count in enumerate(counts):
                ax1.text(count, i, f' {count}', va='center', fontsize=9)

        # Top 15 FN
        top_fn = fn_counter.most_common(15)
        if top_fn:
            codes = [c for c, _ in top_fn]
            counts = [cnt for _, cnt in top_fn]
            ax2.barh(range(len(codes)), counts, color=COLORS['false_negative'], alpha=0.8)
            ax2.set_yticks(range(len(codes)))
            ax2.set_yticklabels(codes)
            ax2.set_xlabel('Count', fontsize=11, fontweight='bold')
            ax2.set_title('Top 15 False Negatives\n(Under-predicted/Missed)', fontsize=13, fontweight='bold')
            ax2.invert_yaxis()

            for i, count in enumerate(counts):
                ax2.text(count, i, f' {count}', va='center', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig2_fp_fn_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig2_fp_fn_comparison.pdf', bbox_inches='tight')
        print("    Saved: fig2_fp_fn_comparison.png/pdf")
        plt.close()

    def save_error_report(
        self,
        categories: Dict,
        fp_counter: Counter,
        fn_counter: Counter,
        confusion_patterns: Dict,
        frequency_tiers: Dict = None,
        chunk_sensitivity: Dict = None,
    ):
        """
        Save comprehensive error analysis report.

        Args:
            categories: Error categories
            fp_counter: False positive counts
            fn_counter: False negative counts
            confusion_patterns: Confusion pattern dictionary
            frequency_tiers: Output of analyze_by_frequency_tier() (R1-6)
            chunk_sensitivity: Output of analyze_chunk_sensitivity() (R2-3)
        """
        print("\n" + "=" * 70)
        print("SAVING ERROR ANALYSIS REPORT")
        print("=" * 70)

        # Create summary report
        total = len(self.sample_results)
        report = {
            'analysis_info': {
                'timestamp': datetime.now().isoformat(),
                'total_samples': total,
                'source_file': str(self.results_dir)
            },
            'error_categories': {
                'perfect_match': {
                    'count': len(categories['perfect_match']),
                    'percentage': round(len(categories['perfect_match']) / total * 100, 1)
                },
                'partial_match': {
                    'count': len(categories['partial_match']),
                    'percentage': round(len(categories['partial_match']) / total * 100, 1)
                },
                'complete_miss': {
                    'count': len(categories['complete_miss']),
                    'percentage': round(len(categories['complete_miss']) / total * 100, 1)
                },
                'false_positive_only': {
                    'count': len(categories['false_positive_only']),
                    'percentage': round(len(categories['false_positive_only']) / total * 100, 1)
                }
            },
            'top_false_positives': [
                {'code': code, 'count': count, 'percentage': round(count / total * 100, 1)}
                for code, count in fp_counter.most_common(20)
            ],
            'top_false_negatives': [
                {'code': code, 'count': count, 'percentage': round(count / total * 100, 1)}
                for code, count in fn_counter.most_common(20)
            ],
            'frequency_tier_breakdown': frequency_tiers or {},
            'chunk_sensitivity': chunk_sensitivity or {},
        }

        # Save JSON report
        report_path = self.output_dir / 'error_analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n  Saved: {report_path}")

        # Save FP/FN tables as CSV
        fp_df = pd.DataFrame([
            {'f_code': code, 'fp_count': count, 'percentage': round(count / total * 100, 1)}
            for code, count in fp_counter.most_common()
        ])
        fp_path = self.output_dir / 'false_positives.csv'
        fp_df.to_csv(fp_path, index=False)
        print(f"  Saved: {fp_path}")

        fn_df = pd.DataFrame([
            {'f_code': code, 'fn_count': count, 'percentage': round(count / total * 100, 1)}
            for code, count in fn_counter.most_common()
        ])
        fn_path = self.output_dir / 'false_negatives.csv'
        fn_df.to_csv(fn_path, index=False)
        print(f"  Saved: {fn_path}")

        # Save error category samples
        for cat_name, samples in categories.items():
            if samples:
                cat_df = pd.DataFrame([
                    {
                        'sample_id': s.get('sample_id'),
                        'actual_codes': '; '.join(s.get('actual_codes', [])),
                        'predicted_codes': '; '.join(s.get('predicted_codes', [])),
                        'f1_score': s.get('f1', 0)
                    }
                    for s in samples[:50]  # Limit to 50 samples
                ])
                cat_path = self.output_dir / f'samples_{cat_name}.csv'
                cat_df.to_csv(cat_path, index=False)
                print(f"  Saved: {cat_path}")

    def run(self):
        """Run complete error analysis pipeline."""
        print("\n" + "=" * 70)
        print("ERROR ANALYSIS")
        print("=" * 70 + "\n")

        self.load_results()

        # Categorize errors
        categories = self.categorize_errors()

        # Analyze false positives and negatives
        fp_counter, fp_details = self.analyze_false_positives()
        fn_counter, fn_details = self.analyze_false_negatives()

        # Analyze confusion patterns
        confusion_patterns = self.analyze_confusion_patterns()

        # Performance by code-frequency tier (Reviewer 1, Comment 6)
        frequency_tiers = self.analyze_by_frequency_tier()

        # Chunk-count sensitivity (Reviewer 2, Comment 3)
        chunk_sensitivity = self.analyze_chunk_sensitivity()

        # Create confusion matrix
        self.create_confusion_matrix(top_n=10)

        # Create visualizations
        self.create_error_visualizations(fp_counter, fn_counter)

        # Save report
        self.save_error_report(categories, fp_counter, fn_counter, confusion_patterns,
                                frequency_tiers, chunk_sensitivity)

        print("\n" + "=" * 70)
        print("ERROR ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\nOutput directory: {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob('*')):
            print(f"  - {file.name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Perform error analysis on evaluation results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=OUTPUT_DIR_6F,
        help="Directory containing evaluation results (default: best strategy 6f)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR_9_ERROR,
        help="Output directory for error analysis"
    )

    args = parser.parse_args()

    analyzer = ErrorAnalyzer(
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )
    analyzer.run()

    return 0


if __name__ == "__main__":
    exit(main())
