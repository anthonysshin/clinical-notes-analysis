#!/usr/bin/env python3
"""
8_BestEvalAnalysis.py - Detailed Analysis of Best Evaluation Strategy (Keyword + CoT)

This script performs comprehensive statistical analysis on the best evaluation
strategy results, including per-code analysis, category breakdowns, and
detailed performance visualizations.

Analyses Performed:
    1. Overall performance summary
    2. Per-code performance analysis
    3. Performance visualizations (3 main + 1 supplementary)
    4. Category-wise performance breakdown

Output:
    - Performance visualization figures (PNG, PDF): fig1-fig3 (main), figS1 (supplementary)
    - Statistical analysis tables (CSV)
    - Summary report (JSON)

Usage:
    python 8_BestEvalAnalysis.py --results-dir ./outputs/6f_KeywordAugmentedCoT --output-dir ./outputs/8_BestEvalAnalysis

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
from typing import Dict
import warnings

from config import OUTPUT_DIR_8_BEST, OUTPUT_DIR, OUTPUT_DIR_6F
from collections import defaultdict

warnings.filterwarnings('ignore')

# F-code categories for grouping analysis
FCODE_CATEGORIES = {
    'Substance Use': ['F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19'],
    'Mood Disorders': ['F30', 'F31', 'F32', 'F33', 'F34', 'F39'],
    'Anxiety Disorders': ['F40', 'F41', 'F42', 'F43', 'F44', 'F45', 'F48'],
    'Cognitive Disorders': ['F00', 'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F09'],
    'Psychotic Disorders': ['F20', 'F21', 'F22', 'F23', 'F24', 'F25', 'F28', 'F29'],
    'Other': ['F50', 'F51', 'F60', 'F63', 'F70', 'F79', 'F80', 'F84', 'F90', 'F91', 'F99']
}


def get_code_category(code):
    """Get category for an F-code."""
    prefix = code[:3] if len(code) >= 3 else code
    for category, prefixes in FCODE_CATEGORIES.items():
        if prefix in prefixes:
            return category
    return 'Other'


# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11


class ResultsAnalyzer:
    """Analyze and visualize evaluation results."""

    def __init__(self, results_dir: str, output_dir: str):
        """
        Initialize analyzer.

        Args:
            results_dir: Directory containing evaluation results
            output_dir: Directory to save analysis outputs
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results = []
        self.predictions_df = None
        self.per_code_df = None

    def load_results(self):
        """Load evaluation results from JSON and CSV files."""
        print("=" * 70)
        print("LOADING EVALUATION RESULTS")
        print("=" * 70)

        # Find evaluation result files
        json_files = list(self.results_dir.glob("evaluation_results_*.json"))
        csv_files = list(self.results_dir.glob("predictions_*.csv"))
        per_code_files = list(self.results_dir.glob("per_code_performance_*.csv"))

        if not json_files:
            raise FileNotFoundError(f"No evaluation results found in {self.results_dir}")

        # Load most recent JSON
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_json = json_files[0]

        print(f"\nLoading: {latest_json.name}")
        with open(latest_json, 'r') as f:
            self.results = json.load(f)

        # Load predictions CSV if available
        if csv_files:
            csv_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            self.predictions_df = pd.read_csv(csv_files[0])
            print(f"Loaded predictions: {csv_files[0].name}")

        # Load per-code performance if available
        if per_code_files:
            per_code_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            self.per_code_df = pd.read_csv(per_code_files[0])
            print(f"Loaded per-code performance: {per_code_files[0].name}")

        print(f"\nLoaded {len(self.results.get('sample_results', []))} sample results")

    def summarize_performance(self):
        """Print and save performance summary."""
        print("\n" + "=" * 70)
        print("PERFORMANCE SUMMARY")
        print("=" * 70)

        metrics = self.results.get('performance_metrics', {})

        print(f"\nOverall Metrics:")
        print(f"  Total Samples:     {metrics.get('total_samples', 'N/A')}")
        print(f"  Success Rate:      {metrics.get('success_rate', 0):.1%}")

        print(f"\nMicro-averaged Metrics:")
        print(f"  Precision:         {metrics.get('micro_precision', 0):.4f}")
        print(f"  Recall:            {metrics.get('micro_recall', 0):.4f}")
        print(f"  F1 Score:          {metrics.get('micro_f1', 0):.4f}")

        print(f"\nMacro-averaged Metrics:")
        print(f"  Precision:         {metrics.get('macro_precision', 0):.4f}")
        print(f"  Recall:            {metrics.get('macro_recall', 0):.4f}")
        print(f"  F1 Score:          {metrics.get('macro_f1', 0):.4f}")

        print(f"\nPerfect Match:")
        print(f"  Count:             {metrics.get('perfect_matches', 0)}")
        print(f"  Rate:              {metrics.get('perfect_match_rate', 0):.1%}")

        # Save summary
        summary = {
            'evaluation_info': self.results.get('evaluation_info', {}),
            'metrics': metrics,
            'analysis_timestamp': datetime.now().isoformat()
        }

        summary_path = self.output_dir / 'performance_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nSummary saved: {summary_path}")

    def analyze_per_code_performance(self):
        """Analyze and visualize per-code performance."""
        print("\n" + "=" * 70)
        print("PER-CODE PERFORMANCE ANALYSIS")
        print("=" * 70)

        if self.per_code_df is None:
            print("No per-code performance data available")
            return

        # Sort by occurrences
        df = self.per_code_df.sort_values('occurrences', ascending=False)

        # Top performers
        print("\nTop 10 Best Performing F-Codes (by F1):")
        top_f1 = df[df['occurrences'] >= 5].nlargest(10, 'f1_score')
        for _, row in top_f1.iterrows():
            print(f"  {row['f_code']:<12} F1: {row['f1_score']:.3f} "
                  f"(n={row['occurrences']})")

        # Worst performers
        print("\nTop 10 Challenging F-Codes (by F1, min 5 occurrences):")
        bottom_f1 = df[df['occurrences'] >= 5].nsmallest(10, 'f1_score')
        for _, row in bottom_f1.iterrows():
            print(f"  {row['f_code']:<12} F1: {row['f1_score']:.3f} "
                  f"(n={row['occurrences']})")

        # Save detailed analysis
        analysis_path = self.output_dir / 'per_code_analysis.csv'
        df.to_csv(analysis_path, index=False)
        print(f"\nPer-code analysis saved: {analysis_path}")

    def create_visualizations(self):
        """Create all analysis visualizations."""
        print("\n" + "=" * 70)
        print("CREATING VISUALIZATIONS")
        print("=" * 70)

        self._plot_sample_f1_distribution()
        self._plot_per_code_heatmap()
        self._plot_category_performance()
        self._plot_frequency_vs_performance()  # Supplementary figure

    def _plot_sample_f1_distribution(self):
        """Plot distribution of F1 scores across samples."""
        print("  Creating F1 distribution chart...")

        if self.predictions_df is None:
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Histogram
        ax1.hist(self.predictions_df['f1_score'], bins=20, color='#3498db',
                 alpha=0.7, edgecolor='black')
        ax1.axvline(self.predictions_df['f1_score'].mean(), color='red',
                    linestyle='--', linewidth=2,
                    label=f"Mean: {self.predictions_df['f1_score'].mean():.3f}")
        ax1.axvline(self.predictions_df['f1_score'].median(), color='green',
                    linestyle='--', linewidth=2,
                    label=f"Median: {self.predictions_df['f1_score'].median():.3f}")
        ax1.set_xlabel('F1 Score', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax1.set_title('Distribution of Sample F1 Scores', fontsize=13, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(axis='y', alpha=0.3)

        # Perfect match breakdown
        perfect = (self.predictions_df['perfect_match'] == 1).sum()
        partial = ((self.predictions_df['f1_score'] > 0) &
                   (self.predictions_df['f1_score'] < 1)).sum()
        complete_miss = (self.predictions_df['f1_score'] == 0).sum()

        categories = ['Perfect Match\n(F1=1.0)', 'Partial Match\n(0<F1<1)', 'Complete Miss\n(F1=0)']
        counts = [perfect, partial, complete_miss]
        colors = ['#2ecc71', '#f39c12', '#e74c3c']

        bars = ax2.bar(categories, counts, color=colors, alpha=0.8)
        ax2.set_ylabel('Number of Samples', fontsize=11, fontweight='bold')
        ax2.set_title('Prediction Quality Breakdown', fontsize=13, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        for bar, count in zip(bars, counts):
            pct = count / len(self.predictions_df) * 100
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{count}\n({pct:.1f}%)', ha='center', va='bottom',
                     fontsize=10, fontweight='bold')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig1_f1_distribution.png', dpi=300, bbox_inches='tight')
        print("    Saved: fig1_f1_distribution.png")
        plt.close()

    def _plot_frequency_vs_performance(self):
        """Plot relationship between code frequency and performance."""
        print("  Creating frequency vs performance chart...")

        if self.per_code_df is None:
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        scatter = ax.scatter(
            self.per_code_df['occurrences'],
            self.per_code_df['f1_score'],
            c=self.per_code_df['f1_score'],
            cmap='RdYlGn',
            s=60,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

        ax.set_xscale('log')
        ax.set_xlabel('Code Frequency (log scale)', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
        ax.set_title('F-Code Frequency vs Performance', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add trend line
        log_freq = np.log10(self.per_code_df['occurrences'] + 1)
        z = np.polyfit(log_freq, self.per_code_df['f1_score'], 1)
        p = np.poly1d(z)
        x_trend = np.logspace(0, np.log10(self.per_code_df['occurrences'].max()), 100)
        ax.plot(x_trend, p(np.log10(x_trend)), 'r--', alpha=0.8, linewidth=2,
                label=f'Trend (slope={z[0]:.3f})')
        ax.legend(fontsize=10)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('F1 Score', fontsize=11)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figS1_frequency_vs_performance.png',
                    dpi=300, bbox_inches='tight')
        print("    Saved: figS1_frequency_vs_performance.png (Supplementary)")
        plt.close()

    def _plot_per_code_heatmap(self):
        """Create heatmap showing per-code performance (top 20 codes by frequency)."""
        print("  Creating per-code performance heatmap...")

        # Load per-code performance from best strategy
        per_code_files = list(Path(OUTPUT_DIR_6F).glob('per_code_performance_*.csv'))
        if not per_code_files:
            print("    Per-code file not found, skipping...")
            return

        per_code_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        df = pd.read_csv(per_code_files[0])

        # Get top 20 codes by occurrences (frequency)
        df_sorted = df.sort_values('occurrences', ascending=False).head(20)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        heatmap_data = df_sorted[['precision', 'recall', 'f1_score']].values

        sns.heatmap(heatmap_data,
                    annot=True,
                    fmt='.2f',
                    cmap='RdYlGn',
                    vmin=0, vmax=1,
                    xticklabels=['Precision', 'Recall', 'F1'],
                    yticklabels=[f"{row['f_code']} (n={int(row['occurrences'])})"
                                for _, row in df_sorted.iterrows()],
                    ax=ax)

        ax.set_title('Per-Code Performance (Top 20 by Frequency)\nKeyword + CoT Strategy',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Metric', fontsize=12)
        ax.set_ylabel('F-Code (Support)', fontsize=12)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig2_per_code_heatmap.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig2_per_code_heatmap.pdf', bbox_inches='tight')
        print("    Saved: fig2_per_code_heatmap.png/pdf")
        plt.close()

    def _plot_category_performance(self):
        """Create performance breakdown by F-code category."""
        print("  Creating category performance chart...")

        # Load predictions from best strategy
        pred_files = list(Path(OUTPUT_DIR_6F).glob('predictions_*.csv'))
        if not pred_files:
            print("    Predictions file not found, skipping...")
            return

        pred_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        df = pd.read_csv(pred_files[0])

        # Calculate per-category metrics
        category_metrics = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})

        for _, row in df.iterrows():
            actual = set(str(row['actual_f_codes']).split('; ')) if pd.notna(row['actual_f_codes']) else set()
            predicted = set(str(row['predicted_f_codes']).split('; ')) if pd.notna(row['predicted_f_codes']) else set()

            actual = {c.strip() for c in actual if c.strip()}
            predicted = {c.strip() for c in predicted if c.strip()}

            for code in actual | predicted:
                category = get_code_category(code)
                if code in actual and code in predicted:
                    category_metrics[category]['tp'] += 1
                elif code in predicted and code not in actual:
                    category_metrics[category]['fp'] += 1
                elif code in actual and code not in predicted:
                    category_metrics[category]['fn'] += 1

        # Calculate F1 for each category
        categories = []
        f1_scores = []
        precisions = []
        recalls = []

        for category in ['Substance Use', 'Mood Disorders', 'Anxiety Disorders',
                         'Cognitive Disorders', 'Psychotic Disorders', 'Other']:
            m = category_metrics[category]
            precision = m['tp'] / (m['tp'] + m['fp']) if (m['tp'] + m['fp']) > 0 else 0
            recall = m['tp'] / (m['tp'] + m['fn']) if (m['tp'] + m['fn']) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            categories.append(category)
            f1_scores.append(f1)
            precisions.append(precision)
            recalls.append(recall)

        # Create grouped bar chart
        fig, ax = plt.subplots(figsize=(12, 6))

        x = np.arange(len(categories))
        width = 0.25

        ax.bar(x - width, precisions, width, label='Precision', color='#3498db', alpha=0.8)
        ax.bar(x, recalls, width, label='Recall', color='#e74c3c', alpha=0.8)
        ax.bar(x + width, f1_scores, width, label='F1', color='#27ae60', alpha=0.8)

        ax.set_xlabel('F-Code Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance by F-Code Category\nKeyword + CoT Strategy', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend(loc='upper right')
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig3_category_performance.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig3_category_performance.pdf', bbox_inches='tight')
        print("    Saved: fig3_category_performance.png/pdf")
        plt.close()

    def create_summary_tables(self):
        """Create summary tables for the analysis."""
        print("\n" + "=" * 70)
        print("CREATING SUMMARY TABLES")
        print("=" * 70)

        metrics = self.results.get('performance_metrics', {})

        # Overall metrics table
        overall_table = pd.DataFrame({
            'Metric': [
                'Total Samples',
                'Success Rate',
                'Micro Precision',
                'Micro Recall',
                'Micro F1',
                'Macro Precision',
                'Macro Recall',
                'Macro F1',
                'Perfect Matches',
                'Perfect Match Rate'
            ],
            'Value': [
                metrics.get('total_samples', 'N/A'),
                f"{metrics.get('success_rate', 0):.1%}",
                f"{metrics.get('micro_precision', 0):.4f}",
                f"{metrics.get('micro_recall', 0):.4f}",
                f"{metrics.get('micro_f1', 0):.4f}",
                f"{metrics.get('macro_precision', 0):.4f}",
                f"{metrics.get('macro_recall', 0):.4f}",
                f"{metrics.get('macro_f1', 0):.4f}",
                metrics.get('perfect_matches', 'N/A'),
                f"{metrics.get('perfect_match_rate', 0):.1%}"
            ]
        })

        overall_path = self.output_dir / 'overall_metrics_table.csv'
        overall_table.to_csv(overall_path, index=False)
        print(f"\n  Saved: {overall_path}")

        # Top codes table
        if self.per_code_df is not None:
            top_codes = self.per_code_df.nlargest(20, 'occurrences')[
                ['f_code', 'occurrences', 'precision', 'recall', 'f1_score', 'tp', 'fp', 'fn']
            ]
            top_codes_path = self.output_dir / 'top_codes_performance.csv'
            top_codes.to_csv(top_codes_path, index=False)
            print(f"  Saved: {top_codes_path}")

    def run(self):
        """Run complete analysis pipeline."""
        print("\n" + "=" * 70)
        print("EVALUATION RESULTS ANALYSIS")
        print("=" * 70 + "\n")

        self.load_results()
        self.summarize_performance()
        self.analyze_per_code_performance()
        self.create_visualizations()
        self.create_summary_tables()

        print("\n" + "=" * 70)
        print("ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\nOutput directory: {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob('*')):
            print(f"  - {file.name}")
        print("\nNext: Run 9_ErrorAnalysis.py for detailed error analysis")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze and visualize evaluation results"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=OUTPUT_DIR,
        help="Directory containing evaluation results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR_8_BEST,
        help="Output directory for analysis results"
    )

    args = parser.parse_args()

    analyzer = ResultsAnalyzer(
        results_dir=args.results_dir,
        output_dir=args.output_dir
    )
    analyzer.run()

    return 0


if __name__ == "__main__":
    exit(main())
