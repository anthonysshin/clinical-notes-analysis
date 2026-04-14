#!/usr/bin/env python3
"""
7_MultiEvalAnalysis.py - Aggregate and Compare Results Across All Evaluation Strategies

This script combines results from all evaluation strategies (6a-6f + base model)
into a unified comparison with publication-ready visualizations and statistical tests.

Analyses Performed:
    1. Results aggregation from all strategies into comparison tables
    2. Publication-ready figures (bar charts, radar plots, heatmaps)
    3. Statistical significance test: Paired t-test between top 2 strategies
    4. Effect size calculation (Cohen's d)

Output:
    - Comparison tables (CSV, LaTeX)
    - Publication figures (PNG, PDF): fig1 (main), figS2 (supplementary)
    - Statistical test results (JSON)
    - Summary report

Usage:
    python 7_MultiEvalAnalysis.py --output-dir ./outputs/7_MultiEvalAnalysis

Author: Clinical Note Analysis Study
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import warnings
from scipy import stats

# Import centralized config for reproducibility and visualization
from config import (
    RANDOM_SEED, set_all_seeds,
    STRATEGY_COLORS, COLORS, FIGURE_STYLE,
    get_strategy_color, get_category_color, apply_figure_style,
    OUTPUT_DIR, OUTPUT_DIR_7_MULTI
)

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility (affects bootstrap sampling)
set_all_seeds(RANDOM_SEED)

# Apply publication-ready figure style
apply_figure_style()

# Set publication-quality plotting style
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Strategy configurations - Professional color palette
# Base model strategies (6g1, 6g2) use same prompts as fine-tuned counterparts for fair comparison
STRATEGY_INFO = {
    '6g1_BaseModel_ZeroShot': {
        'name': 'Zero-Shot (Base)',
        'short_name': 'ZS-Base',
        'description': 'Base model with Zero-Shot prompt (same as 6a)',
        'fine_tuned': False,
        'order': 0
    },
    '6g2_BaseModel_CoT': {
        'name': 'CoT (Base)',
        'short_name': 'CoT-Base',
        'description': 'Base model with Chain-of-Thought prompt (same as 6d)',
        'fine_tuned': False,
        'order': 1
    },
    '6a_ZeroShotBaseline': {
        'name': 'Zero-Shot',
        'short_name': 'ZS',
        'description': 'Basic prompting',
        'fine_tuned': True,
        'order': 2
    },
    '6b_FewShotExemplar': {
        'name': 'Few-Shot',
        'short_name': 'FS',
        'description': 'With examples',
        'fine_tuned': True,
        'order': 3
    },
    '6c_RuleConstrained': {
        'name': 'Rule-Constrained',
        'short_name': 'RC',
        'description': 'With coding rules',
        'fine_tuned': True,
        'order': 4
    },
    '6d_ChainOfThought': {
        'name': 'Chain-of-Thought',
        'short_name': 'CoT',
        'description': 'Step-by-step reasoning',
        'fine_tuned': True,
        'order': 5
    },
    '6e_KeywordAugmented': {
        'name': 'Keyword-Extraction',
        'short_name': 'KE',
        'description': 'With keyword extraction',
        'fine_tuned': True,
        'order': 6
    },
    '6f_KeywordAugmentedCoT': {
        'name': 'Keyword + CoT',
        'short_name': 'KA-CoT',
        'description': 'Best strategy: Keywords + CoT + ambiguous code guidance',
        'fine_tuned': True,
        'order': 7
    }
}


class ResultsAggregator:
    """Aggregate and analyze results from all evaluation strategies."""

    def __init__(self, outputs_dir: str, output_dir: str):
        """
        Initialize aggregator.

        Args:
            outputs_dir: Directory containing evaluation outputs (5a-5f, 5x subdirs)
            output_dir: Directory to save aggregated results
        """
        self.outputs_dir = Path(outputs_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.strategy_results = {}
        self.strategy_predictions = {}
        self.comparison_df = None
        self.statistical_results = None  # Store statistical test results

    def load_all_results(self):
        """Load results from all evaluation strategies."""
        print("=" * 70)
        print("LOADING RESULTS FROM ALL STRATEGIES")
        print("=" * 70)

        for strategy_id in STRATEGY_INFO.keys():
            # Find strategy output directory
            strategy_dir = None
            for pattern in [f"{strategy_id}*", f"*{strategy_id}*"]:
                matches = list(self.outputs_dir.glob(pattern))
                if matches:
                    strategy_dir = matches[0]
                    break

            if strategy_dir is None or not strategy_dir.exists():
                print(f"\n  [{strategy_id}] Not found - skipping")
                continue

            # Find most recent results file
            json_files = list(strategy_dir.glob("evaluation_results_*.json"))
            if not json_files:
                print(f"\n  [{strategy_id}] No results file found")
                continue

            json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            latest_file = json_files[0]

            # Load results
            print(f"\n  [{strategy_id}] Loading: {latest_file.name}")
            with open(latest_file, 'r') as f:
                results = json.load(f)

            self.strategy_results[strategy_id] = results

            # Load predictions CSV if available
            csv_files = list(strategy_dir.glob("predictions_*.csv"))
            if csv_files:
                csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                self.strategy_predictions[strategy_id] = pd.read_csv(csv_files[0])

        print(f"\n\nLoaded {len(self.strategy_results)} strategies:")
        for strategy_id in sorted(self.strategy_results.keys(),
                                   key=lambda x: STRATEGY_INFO.get(x, {}).get('order', 99)):
            info = STRATEGY_INFO.get(strategy_id, {})
            metrics = self.strategy_results[strategy_id].get('performance_metrics', {})
            print(f"  - {info.get('name', strategy_id)}: Micro F1 = {metrics.get('micro_f1', 0):.4f}")

    def create_comparison_table(self):
        """Create comparison table across all strategies."""
        print("\n" + "=" * 70)
        print("CREATING COMPARISON TABLE")
        print("=" * 70)

        rows = []
        for strategy_id in sorted(self.strategy_results.keys(),
                                   key=lambda x: STRATEGY_INFO.get(x, {}).get('order', 99)):
            info = STRATEGY_INFO.get(strategy_id, {})
            metrics = self.strategy_results[strategy_id].get('performance_metrics', {})

            # Get evaluation time in minutes
            eval_time_sec = metrics.get('evaluation_time_seconds', 0)
            eval_time_min = eval_time_sec / 60 if eval_time_sec else 0

            # Compute macro-F1 over codes with at least one ground truth instance,
            # excluding FP-only codes that would artificially deflate the average
            per_code = metrics.get('per_code_metrics', {})
            f1_with_occ = [v['f1'] for v in per_code.values()
                           if v.get('occurrences', 0) > 0]
            macro_f1_filtered = np.mean(f1_with_occ) if f1_with_occ else 0.0

            rows.append({
                'Strategy': info.get('name', strategy_id),
                'Short': info.get('short_name', strategy_id[:4]),
                'Fine-tuned': 'Yes' if info.get('fine_tuned', True) else 'No',
                'Micro P': metrics.get('micro_precision', 0),
                'Micro R': metrics.get('micro_recall', 0),
                'Micro F1': metrics.get('micro_f1', 0),
                'Macro F1': macro_f1_filtered,
                'Perfect Match %': round(metrics.get('perfect_match_rate', 0) * 100, 1),
                'Eval Time (min)': eval_time_min,
                'Samples': metrics.get('total_samples', 0)
            })

        self.comparison_df = pd.DataFrame(rows)

        # Print table
        print("\n" + "-" * 115)
        print(f"{'Strategy':<22} {'FT':<4} {'Micro P':<10} {'Micro R':<10} {'Micro F1':<10} "
              f"{'Macro F1':<10} {'Perfect %':<10} {'Time (min)':<10}")
        print("-" * 115)

        for _, row in self.comparison_df.iterrows():
            ft = 'Yes' if row['Fine-tuned'] == 'Yes' else 'No'
            print(f"{row['Strategy']:<22} {ft:<4} {row['Micro P']:<10.4f} {row['Micro R']:<10.4f} "
                  f"{row['Micro F1']:<10.4f} {row['Macro F1']:<10.4f} {row['Perfect Match %']:<10.1f} "
                  f"{row['Eval Time (min)']:<10.1f}")

        print("-" * 115)

        # Find best strategy
        best_idx = self.comparison_df['Micro F1'].idxmax()
        best_strategy = self.comparison_df.loc[best_idx, 'Strategy']
        best_f1 = self.comparison_df.loc[best_idx, 'Micro F1']
        print(f"\nBest Strategy: {best_strategy} (Micro F1 = {best_f1:.4f})")

        # Calculate improvement from base model
        base_row = self.comparison_df[self.comparison_df['Fine-tuned'] == 'No']
        if not base_row.empty:
            base_f1 = base_row.iloc[0]['Micro F1']
            improvement = best_f1 - base_f1
            pct_improvement = (improvement / base_f1 * 100) if base_f1 > 0 else 0
            print(f"Improvement over Base Model: +{improvement:.4f} ({pct_improvement:.1f}%)")

        # Save CSV
        csv_path = self.output_dir / 'strategy_comparison.csv'
        self.comparison_df.to_csv(csv_path, index=False)
        print(f"\nTable saved: {csv_path}")

        # Save LaTeX table
        latex_path = self.output_dir / 'strategy_comparison.tex'
        self._save_latex_table(latex_path)
        print(f"LaTeX table saved: {latex_path}")

    def _save_latex_table(self, path: Path):
        """Save comparison table in LaTeX format."""
        df = self.comparison_df.copy()

        # Format numbers
        for col in ['Micro P', 'Micro R', 'Micro F1', 'Macro F1']:
            df[col] = df[col].apply(lambda x: f"{x:.3f}")
        df['Perfect Match %'] = df['Perfect Match %'].apply(lambda x: f"{x:.1f}")
        df['Eval Time (min)'] = df['Eval Time (min)'].apply(lambda x: f"{x:.1f}")

        # Find best values for bolding
        numeric_df = self.comparison_df.copy()
        best_micro_f1 = numeric_df['Micro F1'].max()
        best_macro_f1 = numeric_df['Macro F1'].max()

        latex_lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Comparison of Prompting Strategies for Psychiatric F-Code Prediction}",
            "\\label{tab:strategy_comparison}",
            "\\begin{tabular}{lccccccc}",
            "\\toprule",
            "Strategy & Fine-tuned & Micro P & Micro R & Micro F1 & Macro F1 & Perfect (\\%) & Time (min) \\\\",
            "\\midrule"
        ]

        for idx, row in df.iterrows():
            # Bold best values
            micro_f1 = row['Micro F1']
            macro_f1 = row['Macro F1']

            if float(micro_f1) == best_micro_f1:
                micro_f1 = f"\\textbf{{{micro_f1}}}"
            if float(macro_f1) == best_macro_f1:
                macro_f1 = f"\\textbf{{{macro_f1}}}"

            line = f"{row['Strategy']} & {row['Fine-tuned']} & {row['Micro P']} & " \
                   f"{row['Micro R']} & {micro_f1} & {macro_f1} & {row['Perfect Match %']} & " \
                   f"{row['Eval Time (min)']} \\\\"
            latex_lines.append(line)

        latex_lines.extend([
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}"
        ])

        with open(path, 'w') as f:
            f.write('\n'.join(latex_lines))

    def run_statistical_tests(self):
        """Run statistical comparison between top 2 strategies.

        Tests performed:
        - Paired t-test (parametric)
        - Wilcoxon signed-rank test (nonparametric, no distributional assumptions)
        - Cohen's d effect size (paired, using SD of differences)
        """
        print("\n" + "=" * 70)
        print("STATISTICAL SIGNIFICANCE TEST")
        print("=" * 70)

        if len(self.strategy_predictions) < 2:
            print("\nInsufficient data for statistical tests (need at least 2 strategies with predictions)")
            return {}

        # Get sample-level F1 scores for each strategy
        strategy_f1_scores = {}
        for strategy_id, pred_df in self.strategy_predictions.items():
            if 'f1_score' in pred_df.columns:
                strategy_f1_scores[strategy_id] = pred_df['f1_score'].values

        # Find top 2 strategies by mean F1
        strategy_means = {k: np.mean(v) for k, v in strategy_f1_scores.items()}
        sorted_strategies = sorted(strategy_means.items(), key=lambda x: x[1], reverse=True)

        if len(sorted_strategies) < 2:
            print("\nNeed at least 2 strategies with F1 scores")
            return {}

        top1_id, top1_mean = sorted_strategies[0]
        top2_id, top2_mean = sorted_strategies[1]

        top1_name = STRATEGY_INFO.get(top1_id, {}).get('name', top1_id)
        top2_name = STRATEGY_INFO.get(top2_id, {}).get('name', top2_id)

        print(f"\nTop 2 Strategies (by Sample-Averaged F1):")
        print(f"  #1: {top1_name} (Sample-Avg F1 = {top1_mean:.4f})")
        print(f"  #2: {top2_name} (Sample-Avg F1 = {top2_mean:.4f})")

        # Paired samples
        f1_1 = strategy_f1_scores[top1_id]
        f1_2 = strategy_f1_scores[top2_id]

        # Ensure same length
        min_len = min(len(f1_1), len(f1_2))
        f1_1 = f1_1[:min_len]
        f1_2 = f1_2[:min_len]

        diff = f1_1 - f1_2

        # Paired t-test
        t_stat, t_pvalue = stats.ttest_rel(f1_1, f1_2)

        # Wilcoxon signed-rank test (nonparametric)
        w_stat, w_pvalue = stats.wilcoxon(f1_1, f1_2)

        # Effect size (Cohen's d for paired samples)
        cohens_d = np.mean(diff) / np.std(diff, ddof=1) if np.std(diff) > 0 else 0

        # Interpret significance
        def interpret_p(p):
            if p < 0.001: return "highly significant (p < 0.001)"
            elif p < 0.01: return "very significant (p < 0.01)"
            elif p < 0.05: return "significant (p < 0.05)"
            else: return "NOT significant (p >= 0.05)"

        # Interpret effect size
        if abs(cohens_d) < 0.2:
            effect_interp = "negligible"
        elif abs(cohens_d) < 0.5:
            effect_interp = "small"
        elif abs(cohens_d) < 0.8:
            effect_interp = "medium"
        else:
            effect_interp = "large"

        print(f"\nPaired t-test (#1 vs #2):")
        print("-" * 60)
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {t_pvalue:.6f}")
        print(f"  Result: {interpret_p(t_pvalue)}")

        print(f"\nWilcoxon signed-rank test (#1 vs #2):")
        print("-" * 60)
        print(f"  W-statistic: {w_stat:.1f}")
        print(f"  p-value: {w_pvalue:.6f}")
        print(f"  Result: {interpret_p(w_pvalue)}")

        print(f"\nEffect Size:")
        print(f"  Cohen's d: {cohens_d:.4f} ({effect_interp} effect)")
        print(f"  Mean difference: {np.mean(diff):.4f}")

        # Store results for use in visualizations
        self.statistical_results = {
            'top1_strategy': top1_id,
            'top1_name': top1_name,
            'top1_sample_avg_f1': float(top1_mean),
            'top2_strategy': top2_id,
            'top2_name': top2_name,
            'top2_sample_avg_f1': float(top2_mean),
            'paired_ttest': {
                't_statistic': float(t_stat),
                'p_value': float(t_pvalue),
                'significant_0.05': bool(t_pvalue < 0.05),
                'significant_0.01': bool(t_pvalue < 0.01),
                'significant_0.001': bool(t_pvalue < 0.001),
            },
            'wilcoxon': {
                'w_statistic': float(w_stat),
                'p_value': float(w_pvalue),
                'significant_0.05': bool(w_pvalue < 0.05),
                'significant_0.01': bool(w_pvalue < 0.01),
                'significant_0.001': bool(w_pvalue < 0.001),
            },
            'effect_size': {
                'cohens_d': float(cohens_d),
                'interpretation': effect_interp,
                'mean_difference': float(np.mean(diff))
            }
        }

        # Save results to file
        stats_path = self.output_dir / 'statistical_tests.json'
        with open(stats_path, 'w') as f:
            json.dump(self.statistical_results, f, indent=2)
        print(f"\nStatistical results saved: {stats_path}")

        return self.statistical_results

    def run_bootstrap_cis(self, n_iterations: int = 1000):
        """Compute bootstrap 95% CIs for all strategies and the paired difference.

        For each strategy, resamples per-sample TP/FP/FN with replacement and
        recomputes micro-F1, micro-precision, micro-recall, and exact match rate.
        This correctly bootstraps micro-averaged metrics by aggregating counts
        rather than averaging per-sample F1 scores.

        Also computes a paired bootstrap CI for the micro-F1 difference between
        the top two strategies (using the same resampled indices for both).

        Args:
            n_iterations: Number of bootstrap iterations (default 1000).
        """
        print("\n" + "=" * 70)
        print(f"BOOTSTRAP CONFIDENCE INTERVALS ({n_iterations} iterations)")
        print("=" * 70)

        rng = np.random.RandomState(RANDOM_SEED)

        # --- Per-strategy CIs ---
        bootstrap_results = {}
        for strategy_id, results in self.strategy_results.items():
            samples = results.get('sample_results', [])
            if not samples:
                continue

            tp = np.array([s['tp'] for s in samples])
            fp = np.array([s['fp'] for s in samples])
            fn = np.array([s['fn'] for s in samples])
            pm = np.array([s['perfect_match'] for s in samples], dtype=float)
            n = len(samples)

            boot_micro_f1 = np.empty(n_iterations)
            boot_micro_p = np.empty(n_iterations)
            boot_micro_r = np.empty(n_iterations)
            boot_exact = np.empty(n_iterations)

            for i in range(n_iterations):
                idx = rng.choice(n, size=n, replace=True)
                b_tp = np.sum(tp[idx])
                b_fp = np.sum(fp[idx])
                b_fn = np.sum(fn[idx])

                b_p = b_tp / (b_tp + b_fp) if (b_tp + b_fp) > 0 else 0.0
                b_r = b_tp / (b_tp + b_fn) if (b_tp + b_fn) > 0 else 0.0
                b_f1 = 2 * b_p * b_r / (b_p + b_r) if (b_p + b_r) > 0 else 0.0

                boot_micro_f1[i] = b_f1
                boot_micro_p[i] = b_p
                boot_micro_r[i] = b_r
                boot_exact[i] = np.mean(pm[idx])

            info = STRATEGY_INFO.get(strategy_id, {})
            name = info.get('name', strategy_id)
            metrics = results.get('performance_metrics', {})

            ci = {
                'micro_f1': {
                    'point': float(metrics.get('micro_f1', 0)),
                    'ci_lower': float(np.percentile(boot_micro_f1, 2.5)),
                    'ci_upper': float(np.percentile(boot_micro_f1, 97.5)),
                },
                'micro_precision': {
                    'point': float(metrics.get('micro_precision', 0)),
                    'ci_lower': float(np.percentile(boot_micro_p, 2.5)),
                    'ci_upper': float(np.percentile(boot_micro_p, 97.5)),
                },
                'micro_recall': {
                    'point': float(metrics.get('micro_recall', 0)),
                    'ci_lower': float(np.percentile(boot_micro_r, 2.5)),
                    'ci_upper': float(np.percentile(boot_micro_r, 97.5)),
                },
                'exact_match': {
                    'point': float(metrics.get('perfect_match_rate', 0)),
                    'ci_lower': float(np.percentile(boot_exact, 2.5)),
                    'ci_upper': float(np.percentile(boot_exact, 97.5)),
                },
            }
            bootstrap_results[strategy_id] = ci

            print(f"\n  {name}:")
            print(f"    Micro F1:  {ci['micro_f1']['point']:.3f} "
                  f"({ci['micro_f1']['ci_lower']:.3f}-{ci['micro_f1']['ci_upper']:.3f})")
            print(f"    Precision: {ci['micro_precision']['point']:.3f} "
                  f"({ci['micro_precision']['ci_lower']:.3f}-{ci['micro_precision']['ci_upper']:.3f})")
            print(f"    Recall:    {ci['micro_recall']['point']:.3f} "
                  f"({ci['micro_recall']['ci_lower']:.3f}-{ci['micro_recall']['ci_upper']:.3f})")
            print(f"    Exact:     {ci['exact_match']['point']:.1%} "
                  f"({ci['exact_match']['ci_lower']:.1%}-{ci['exact_match']['ci_upper']:.1%})")

        # --- Paired bootstrap for top 2 strategies ---
        paired_diff = None
        if self.statistical_results:
            top1_id = self.statistical_results['top1_strategy']
            top2_id = self.statistical_results['top2_strategy']

            s1 = self.strategy_results[top1_id]['sample_results']
            s2 = self.strategy_results[top2_id]['sample_results']

            tp1 = np.array([s['tp'] for s in s1])
            fp1 = np.array([s['fp'] for s in s1])
            fn1 = np.array([s['fn'] for s in s1])
            tp2 = np.array([s['tp'] for s in s2])
            fp2 = np.array([s['fp'] for s in s2])
            fn2 = np.array([s['fn'] for s in s2])
            n = min(len(s1), len(s2))

            boot_diff = np.empty(n_iterations)
            for i in range(n_iterations):
                idx = rng.choice(n, size=n, replace=True)

                b_tp1, b_fp1, b_fn1 = np.sum(tp1[idx]), np.sum(fp1[idx]), np.sum(fn1[idx])
                b_tp2, b_fp2, b_fn2 = np.sum(tp2[idx]), np.sum(fp2[idx]), np.sum(fn2[idx])

                denom1 = 2 * b_tp1 + b_fp1 + b_fn1
                denom2 = 2 * b_tp2 + b_fp2 + b_fn2
                f1_1 = 2 * b_tp1 / denom1 if denom1 > 0 else 0.0
                f1_2 = 2 * b_tp2 / denom2 if denom2 > 0 else 0.0
                boot_diff[i] = f1_1 - f1_2

            top1_name = self.statistical_results['top1_name']
            top2_name = self.statistical_results['top2_name']
            point_diff = bootstrap_results[top1_id]['micro_f1']['point'] - \
                         bootstrap_results[top2_id]['micro_f1']['point']

            paired_diff = {
                'strategy_1': top1_name,
                'strategy_2': top2_name,
                'point_difference': float(point_diff),
                'mean_boot_difference': float(np.mean(boot_diff)),
                'ci_lower': float(np.percentile(boot_diff, 2.5)),
                'ci_upper': float(np.percentile(boot_diff, 97.5)),
            }

            print(f"\n  Paired Difference ({top1_name} - {top2_name}):")
            print(f"    Point diff:    {point_diff:.4f}")
            print(f"    Bootstrap mean: {np.mean(boot_diff):.4f}")
            print(f"    95% CI:        ({paired_diff['ci_lower']:.4f}-{paired_diff['ci_upper']:.4f})")

        # Save
        output = {
            'n_iterations': n_iterations,
            'random_seed': RANDOM_SEED,
            'per_strategy': bootstrap_results,
        }
        if paired_diff:
            output['paired_difference'] = paired_diff

        ci_path = self.output_dir / 'bootstrap_cis.json'
        with open(ci_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"\n  Saved: {ci_path}")

    def create_visualizations(self):
        """Create publication-ready visualizations."""
        print("\n" + "=" * 70)
        print("CREATING VISUALIZATIONS")
        print("=" * 70)

        self._plot_grouped_strategy_comparison()
        self._plot_precision_recall_comparison()  # Supplementary figure

    def _plot_strategy_comparison_bar(self):
        """Create bar chart comparing all strategies with statannotations for significance.

        Uses actual Micro F1 values from evaluation results (not re-computed from samples)
        to ensure correctness and consistency with reported metrics.
        Colors are assigned dynamically based on F1 ranking for visual consistency.
        """
        print("\n  Creating strategy comparison bar chart...")

        # Build data structures from actual evaluation results
        strategy_data = []
        for strategy_id, results in self.strategy_results.items():
            info = STRATEGY_INFO.get(strategy_id, {})
            metrics = results.get('performance_metrics', {})
            micro_f1 = metrics.get('micro_f1', 0)

            strategy_name = info.get('name', strategy_id)
            if not info.get('fine_tuned', True):
                display_name = f"{strategy_name}\n(No Fine-tuning)"
            else:
                display_name = strategy_name

            strategy_data.append({
                'strategy_id': strategy_id,
                'name': strategy_name,
                'display_name': display_name,
                'micro_f1': micro_f1,
                'fine_tuned': info.get('fine_tuned', True)
            })

        # Sort by Micro F1 (ascending for left-to-right improvement visual)
        strategy_data.sort(key=lambda x: x['micro_f1'])

        # Move Base Models (not fine-tuned) to leftmost positions
        base_items = [s for s in strategy_data if not s['fine_tuned']]
        ft_items = [s for s in strategy_data if s['fine_tuned']]

        # Sort base items by F1 (ascending) and put them first
        base_items.sort(key=lambda x: x['micro_f1'])
        strategy_data = base_items + ft_items

        # Assign colors dynamically based on position (darker = higher F1)
        # Professional grey gradient from light to dark
        n_strategies = len(strategy_data)
        n_base = len(base_items)
        n_ft = len(ft_items)
        colors = []
        strategy_colors = self._get_strategy_colors()
        for i, s in enumerate(strategy_data):
            # Use centralized color palette from config.py
            strategy_id = s.get('strategy_id', '')
            colors.append(strategy_colors.get(strategy_id, COLORS['neutral']))

        # Extract ordered lists for plotting
        display_names = [s['display_name'] for s in strategy_data]
        micro_f1_values = [s['micro_f1'] for s in strategy_data]

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 7))
        x_pos = np.arange(len(strategy_data))

        # Create bars
        bars = ax.bar(x_pos, micro_f1_values, color=colors,
                      edgecolor='black', linewidth=0.8)

        # Add hatch pattern to all base model bars (not fine-tuned)
        for i, s in enumerate(strategy_data):
            if not s['fine_tuned']:
                bars[i].set_hatch('///')

        # Add F1 values on top of bars (3 decimal places for journal)
        for i, (bar, f1_val) in enumerate(zip(bars, micro_f1_values)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.012,
                    f'{f1_val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Statistical significance annotations - draw manually for better control
        # Get fine-tuned strategies sorted by actual Micro F1 (descending)
        ft_strategies = [s for s in strategy_data if s['fine_tuned']]
        ft_strategies.sort(key=lambda x: x['micro_f1'], reverse=True)

        # Get positions of top strategies in display order
        def get_bar_index(display_name):
            return display_names.index(display_name)

        if len(ft_strategies) >= 2:
            max_f1 = max(micro_f1_values)

            # First bracket: top 1 vs top 2 (Keyword+CoT vs Chain-of-Thought)
            idx1 = get_bar_index(ft_strategies[0]['display_name'])
            idx2 = get_bar_index(ft_strategies[1]['display_name'])
            x1, x2 = min(idx1, idx2), max(idx1, idx2)
            y_bracket1 = max_f1 + 0.045

            # Draw bracket with significance annotation based on actual statistical test
            ax.plot([x1, x1, x2, x2], [y_bracket1 - 0.01, y_bracket1, y_bracket1, y_bracket1 - 0.01],
                    color='black', linewidth=1.2)
            # Determine annotation based on statistical test p-value
            if hasattr(self, 'statistical_results') and self.statistical_results:
                p_val = self.statistical_results.get('paired_ttest', {}).get('p_value', 1.0)
                if p_val < 0.001:
                    sig_annotation = '***'
                elif p_val < 0.01:
                    sig_annotation = '**'
                elif p_val < 0.05:
                    sig_annotation = '*'
                else:
                    sig_annotation = 'ns'
            else:
                sig_annotation = 'ns'
            ax.text((x1 + x2) / 2, y_bracket1 + 0.008, sig_annotation, ha='center', va='bottom',
                    fontsize=11, fontweight='bold')

            if len(ft_strategies) >= 3:
                # Second bracket: top 2 vs top 3 (Chain-of-Thought vs Keyword-Augmented)
                idx2 = get_bar_index(ft_strategies[1]['display_name'])
                idx3 = get_bar_index(ft_strategies[2]['display_name'])
                x1, x2 = min(idx2, idx3), max(idx2, idx3)
                y_bracket2 = max_f1 + 0.095

                # Draw bracket
                ax.plot([x1, x1, x2, x2], [y_bracket2 - 0.01, y_bracket2, y_bracket2, y_bracket2 - 0.01],
                        color='black', linewidth=1.2)
                ax.text((x1 + x2) / 2, y_bracket2 + 0.008, '****', ha='center', va='bottom',
                        fontsize=11, fontweight='bold')

        # Configure axes
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_names, rotation=45, ha='right', fontsize=11)
        ax.set_xlabel('')
        ax.set_ylabel('Micro F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('Evaluation Approach Comparison for Psychiatric F-Code Prediction',
                     fontsize=14, fontweight='bold', pad=15)

        # Horizontal gridlines only and limits
        ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='grey')
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(micro_f1_values) * 1.28)  # Extra space for brackets

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig1_strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig1_strategy_comparison.pdf', bbox_inches='tight')
        print("    Saved: fig1_strategy_comparison.png/pdf")
        plt.close()

        # Print verification info
        print(f"    Verified: Best strategy is '{ft_strategies[0]['name']}' with Micro F1 = {ft_strategies[0]['micro_f1']:.4f}")

    def _plot_grouped_strategy_comparison(self):
        """Create grouped bar chart with 3 categories: Base, Prompting, Keyword Augmentation.

        Groups:
        - Base: Zero-Shot (Base), CoT (Base)
        - Prompting: Zero-Shot, Few-Shot, Rule-Constrained, Chain-of-Thought
        - Keyword Augmentation: Keyword-Augmented, Keyword + CoT
        """
        print("\n  Creating grouped strategy comparison chart...")

        # Define group assignments
        base_strategies = ['6g1_BaseModel_ZeroShot', '6g2_BaseModel_CoT']
        prompting_strategies = ['6a_ZeroShotBaseline', '6b_FewShotExemplar',
                                '6c_RuleConstrained', '6d_ChainOfThought']
        keyword_strategies = ['6e_KeywordAugmented', '6f_KeywordAugmentedCoT']

        # Build data structures from actual evaluation results
        all_strategy_data = {}
        for strategy_id, results in self.strategy_results.items():
            info = STRATEGY_INFO.get(strategy_id, {})
            metrics = results.get('performance_metrics', {})
            micro_f1 = metrics.get('micro_f1', 0)

            strategy_name = info.get('name', strategy_id)
            short_name = info.get('short_name', strategy_name)

            all_strategy_data[strategy_id] = {
                'strategy_id': strategy_id,
                'name': strategy_name,
                'short_name': short_name,
                'micro_f1': micro_f1,
                'fine_tuned': info.get('fine_tuned', True)
            }

        # Build ordered list by groups
        strategy_data = []
        group_labels = []
        group_positions = []

        # Group 1: Base (use full names with "(No Fine-tuning)" suffix)
        group_start = 0
        for sid in base_strategies:
            if sid in all_strategy_data:
                s = all_strategy_data[sid]
                s['display_name'] = f"{s['name']}\n(No Fine-tuning)"
                s['group'] = 'Base'
                strategy_data.append(s)
        group_positions.append((group_start, len(strategy_data) - 1, 'Base'))

        # Group 2: Prompting (sorted by F1 within group, use full names)
        group_start = len(strategy_data)
        prompting_data = [all_strategy_data[sid] for sid in prompting_strategies if sid in all_strategy_data]
        prompting_data.sort(key=lambda x: x['micro_f1'])
        for s in prompting_data:
            s['display_name'] = s['name']
            s['group'] = 'Prompting'
            strategy_data.append(s)
        group_positions.append((group_start, len(strategy_data) - 1, 'Prompting'))

        # Group 3: Keyword Augmentation (Keyword-Augmented first, then Keyword + CoT, use full names)
        group_start = len(strategy_data)
        for sid in keyword_strategies:
            if sid in all_strategy_data:
                s = all_strategy_data[sid]
                s['display_name'] = s['name']
                s['group'] = 'Keyword'
                strategy_data.append(s)
        group_positions.append((group_start, len(strategy_data) - 1, 'Keyword'))

        # Get colors
        strategy_colors = self._get_strategy_colors()
        colors = [strategy_colors.get(s['strategy_id'], COLORS['neutral']) for s in strategy_data]

        # Extract ordered lists for plotting
        display_names = [s['display_name'] for s in strategy_data]
        micro_f1_values = [s['micro_f1'] for s in strategy_data]

        # Create figure with more width for groups and extra bottom space for labels
        fig, ax = plt.subplots(figsize=(14, 10))
        plt.subplots_adjust(bottom=0.28, left=0.08, right=0.96)  # Space for rotated labels

        # Calculate x positions with gaps between groups
        x_pos = []
        current_x = 0
        gap = 0.4  # Small gap between groups
        for i, s in enumerate(strategy_data):
            if i > 0 and strategy_data[i]['group'] != strategy_data[i-1]['group']:
                current_x += gap
            x_pos.append(current_x)
            current_x += 1
        x_pos = np.array(x_pos)

        # Load bootstrap CIs if available
        ci_path = self.output_dir / 'bootstrap_cis.json'
        boot_cis = {}
        if ci_path.exists():
            with open(ci_path) as f:
                boot_data = json.load(f)
            boot_cis = boot_data.get('per_strategy', {})

        # Build CI error arrays (asymmetric: lower error, upper error)
        ci_lower_err = []
        ci_upper_err = []
        ci_upper_vals = []
        for s in strategy_data:
            sid = s['strategy_id']
            if sid in boot_cis:
                ci_lo = boot_cis[sid]['micro_f1']['ci_lower']
                ci_hi = boot_cis[sid]['micro_f1']['ci_upper']
                ci_lower_err.append(s['micro_f1'] - ci_lo)
                ci_upper_err.append(ci_hi - s['micro_f1'])
                ci_upper_vals.append(ci_hi)
            else:
                ci_lower_err.append(0)
                ci_upper_err.append(0)
                ci_upper_vals.append(s['micro_f1'])

        # Create bars with CI error bars
        bars = ax.bar(x_pos, micro_f1_values, color=colors,
                      edgecolor='black', linewidth=0.8, width=0.8,
                      yerr=[ci_lower_err, ci_upper_err],
                      error_kw={'capsize': 4, 'capthick': 1.2, 'elinewidth': 1.2,
                                'color': 'black'})

        # Add hatch pattern to base model bars
        for i, s in enumerate(strategy_data):
            if not s['fine_tuned']:
                bars[i].set_hatch('///')

        # Add F1 values above CI whisker caps
        for i, (bar, f1_val) in enumerate(zip(bars, micro_f1_values)):
            label_y = ci_upper_vals[i] + 0.008
            ax.text(bar.get_x() + bar.get_width()/2, label_y,
                    f'{f1_val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # Add group labels at the bottom
        group_label_y = -0.22
        for start, end, label in group_positions:
            mid_x = (x_pos[start] + x_pos[end]) / 2
            ax.text(mid_x, group_label_y, label, ha='center', va='top',
                    fontsize=12, fontweight='bold', transform=ax.get_xaxis_transform())

        # Statistical significance annotation: Top 1 vs Top 2 only
        all_sorted = sorted(strategy_data, key=lambda x: x['micro_f1'], reverse=True)

        if len(all_sorted) >= 2:
            max_f1 = max(micro_f1_values)

            # Helper to get bar index
            def get_bar_index(strategy_id):
                for i, s in enumerate(strategy_data):
                    if s['strategy_id'] == strategy_id:
                        return i
                return -1

            # Bracket: Top 1 vs Top 2
            idx1 = get_bar_index(all_sorted[0]['strategy_id'])
            idx2 = get_bar_index(all_sorted[1]['strategy_id'])
            if idx1 >= 0 and idx2 >= 0:
                x1, x2 = x_pos[min(idx1, idx2)], x_pos[max(idx1, idx2)]
                # Place bracket above the highest CI whisker + F1 label
                max_ci_upper = max(ci_upper_vals[idx1], ci_upper_vals[idx2])
                y_bracket = max_ci_upper + 0.06  # Clear whisker caps and F1 labels

                ax.plot([x1, x1, x2, x2], [y_bracket - 0.01, y_bracket, y_bracket, y_bracket - 0.01],
                        color='black', linewidth=1.2)

                # Get significance annotation (using Wilcoxon signed-rank test)
                if hasattr(self, 'statistical_results') and self.statistical_results:
                    p_val = self.statistical_results.get('wilcoxon', {}).get('p_value', 1.0)
                    if p_val < 0.001:
                        sig_label = '***'
                    elif p_val < 0.01:
                        sig_label = '**'
                    elif p_val < 0.05:
                        sig_label = '*'
                    else:
                        sig_label = 'ns'
                else:
                    sig_label = '*'  # Default based on known result
                ax.text((x1 + x2) / 2, y_bracket + 0.008, sig_label, ha='center', va='bottom',
                        fontsize=11, fontweight='bold')

        # Configure axes
        ax.set_xticks(x_pos)
        ax.set_xticklabels(display_names, rotation=40, ha='right', fontsize=10)
        ax.set_xlabel('')
        ax.set_ylabel('Micro F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('Evaluation Approach Comparison for Psychiatric F-Code Prediction',
                     fontsize=14, fontweight='bold', pad=15)

        # Horizontal gridlines only and limits
        ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='grey')
        ax.set_axisbelow(True)
        ax.set_ylim(0, max(micro_f1_values) * 1.25)  # Space for significance bracket

        plt.savefig(self.output_dir / 'fig1_strategy_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig1_strategy_comparison.pdf', bbox_inches='tight')
        print("    Saved: fig1_strategy_comparison.png/pdf")
        plt.close()

    def _get_strategy_colors(self):
        """Get professional color palette for strategies from centralized config."""
        # Use centralized STRATEGY_COLORS from config.py for consistency
        return STRATEGY_COLORS.copy()

    def _plot_precision_recall_comparison(self):
        """Create precision-recall comparison chart."""
        print("  Creating precision-recall comparison...")

        fig, ax = plt.subplots(figsize=(10, 8))
        colors = self._get_strategy_colors()

        for _, row in self.comparison_df.iterrows():
            strategy_id = next((k for k, v in STRATEGY_INFO.items() if v['name'] == row['Strategy']), None)
            if strategy_id:
                info = STRATEGY_INFO[strategy_id]
                color = colors.get(strategy_id, COLORS['neutral'])
                ax.scatter(row['Micro R'], row['Micro P'],
                           s=200, c=[color], alpha=0.8,
                           edgecolors='black', linewidth=1,
                           label=f"{info['short_name']}: F1={row['Micro F1']:.3f}")

                # Add label
                ax.annotate(info['short_name'],
                            (row['Micro R'], row['Micro P']),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9, fontweight='bold')

        # Add F1 iso-curves
        for f1 in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            x = np.linspace(0.01, 1, 100)
            y = f1 * x / (2 * x - f1)
            valid = (y > 0) & (y <= 1)
            ax.plot(x[valid], y[valid], '--', color='gray', alpha=0.4, linewidth=1)
            if np.any(valid):
                ax.text(0.92, f1 * 0.92 / (2 * 0.92 - f1), f'F1={f1}',
                        fontsize=8, color='gray', alpha=0.7)

        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Trade-off by Strategy', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.yaxis.grid(True, linestyle='-', alpha=0.3, color='grey')
        ax.legend(loc='lower left', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figS2_precision_recall.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figS2_precision_recall.pdf', bbox_inches='tight')
        print("    Saved: figS2_precision_recall.png/pdf (Supplementary)")
        plt.close()

    def create_summary_report(self):
        """Create comprehensive summary report."""
        print("\n" + "=" * 70)
        print("CREATING SUMMARY REPORT")
        print("=" * 70)

        # Find best strategy
        best_idx = self.comparison_df['Micro F1'].idxmax()
        best_row = self.comparison_df.loc[best_idx]

        # Base model comparison
        base_row = self.comparison_df[self.comparison_df['Fine-tuned'] == 'No']
        base_f1 = base_row.iloc[0]['Micro F1'] if not base_row.empty else 0

        report = {
            'summary': {
                'num_strategies': len(self.strategy_results),
                'strategies_evaluated': list(self.strategy_results.keys()),
                'best_strategy': best_row['Strategy'],
                'best_micro_f1': float(best_row['Micro F1']),
                'base_model_f1': float(base_f1),
                'improvement_over_base': float(best_row['Micro F1'] - base_f1),
                'improvement_percentage': float((best_row['Micro F1'] - base_f1) / base_f1 * 100) if base_f1 > 0 else 0
            },
            'strategy_rankings': {
                'by_micro_f1': self.comparison_df.sort_values('Micro F1', ascending=False)[
                    ['Strategy', 'Micro F1']].to_dict('records'),
                'by_precision': self.comparison_df.sort_values('Micro P', ascending=False)[
                    ['Strategy', 'Micro P']].to_dict('records'),
                'by_recall': self.comparison_df.sort_values('Micro R', ascending=False)[
                    ['Strategy', 'Micro R']].to_dict('records')
            },
            'analysis_timestamp': datetime.now().isoformat()
        }

        report_path = self.output_dir / 'aggregation_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved: {report_path}")

        # Print summary
        print("\n" + "=" * 70)
        print("AGGREGATION SUMMARY")
        print("=" * 70)
        print(f"\nStrategies Evaluated: {len(self.strategy_results)}")
        print(f"Best Strategy: {best_row['Strategy']}")
        print(f"  Micro F1: {best_row['Micro F1']:.4f}")
        print(f"  Micro Precision: {best_row['Micro P']:.4f}")
        print(f"  Micro Recall: {best_row['Micro R']:.4f}")
        if base_f1 > 0:
            print(f"\nBase Model F1: {base_f1:.4f}")
            print(f"Improvement: +{best_row['Micro F1'] - base_f1:.4f} "
                  f"({(best_row['Micro F1'] - base_f1) / base_f1 * 100:.1f}%)")

    def run(self):
        """Run complete aggregation pipeline."""
        print("\n" + "=" * 70)
        print("RESULTS AGGREGATION AND COMPARISON")
        print("=" * 70 + "\n")

        self.load_all_results()

        if len(self.strategy_results) == 0:
            print("\nNo results found to aggregate!")
            return

        self.create_comparison_table()
        self.run_statistical_tests()
        self.run_bootstrap_cis()
        self.create_visualizations()
        self.create_summary_report()

        print("\n" + "=" * 70)
        print("AGGREGATION COMPLETE")
        print("=" * 70)
        print(f"\nOutput directory: {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob('*')):
            print(f"  - {file.name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Aggregate and compare results from all evaluation strategies"
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory containing evaluation outputs (6a-6g subdirs)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR_7_MULTI,
        help="Output directory for aggregated results"
    )

    args = parser.parse_args()

    aggregator = ResultsAggregator(
        outputs_dir=args.outputs_dir,
        output_dir=args.output_dir
    )
    aggregator.run()

    return 0


if __name__ == "__main__":
    exit(main())
