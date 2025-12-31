#!/usr/bin/env python3
"""
7_SingleEvalAnalysis.py - Statistical Analysis and Visualization of Evaluation Results

This script performs comprehensive statistical analysis on evaluation results,
including significance testing, performance visualizations, and comparison tables.

Analyses Performed:
    1. Overall performance summary
    2. Per-code performance analysis
    3. Performance visualizations
    4. Statistical significance tests (if multiple evaluations)

Output:
    - Performance visualization figures (PNG)
    - Statistical analysis tables (CSV)
    - Summary report (JSON)

Usage:
    python 6_Analysis.py --results-dir ./evaluation_results --output-dir ./analysis_outputs

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

from config import OUTPUT_DIR_7_ANALYSIS, OUTPUT_DIR

warnings.filterwarnings('ignore')

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

        self._plot_overall_metrics()
        self._plot_sample_f1_distribution()
        self._plot_per_code_performance()
        self._plot_frequency_vs_performance()
        self._plot_precision_recall_tradeoff()

    def _plot_overall_metrics(self):
        """Plot overall performance metrics."""
        print("\n  Creating overall metrics chart...")

        metrics = self.results.get('performance_metrics', {})

        fig, ax = plt.subplots(figsize=(10, 6))

        categories = ['Micro', 'Macro']
        precision = [metrics.get('micro_precision', 0), metrics.get('macro_precision', 0)]
        recall = [metrics.get('micro_recall', 0), metrics.get('macro_recall', 0)]
        f1 = [metrics.get('micro_f1', 0), metrics.get('macro_f1', 0)]

        x = np.arange(len(categories))
        width = 0.25

        bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x, recall, width, label='Recall', color='#2ecc71', alpha=0.8)
        bars3 = ax.bar(x + width, f1, width, label='F1 Score', color='#e74c3c', alpha=0.8)

        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Overall Performance Metrics', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=11)
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)

        # Add value labels
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig1_overall_metrics.png', dpi=300, bbox_inches='tight')
        print("    Saved: fig1_overall_metrics.png")
        plt.close()

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
        plt.savefig(self.output_dir / 'fig2_f1_distribution.png', dpi=300, bbox_inches='tight')
        print("    Saved: fig2_f1_distribution.png")
        plt.close()

    def _plot_per_code_performance(self):
        """Plot per-code performance for top codes."""
        print("  Creating per-code performance chart...")

        if self.per_code_df is None:
            return

        # Get top 15 codes by occurrence
        top_codes = self.per_code_df.nlargest(15, 'occurrences')

        fig, ax = plt.subplots(figsize=(14, 8))

        x = np.arange(len(top_codes))
        width = 0.25

        bars1 = ax.bar(x - width, top_codes['precision'], width,
                       label='Precision', color='#3498db', alpha=0.8)
        bars2 = ax.bar(x, top_codes['recall'], width,
                       label='Recall', color='#2ecc71', alpha=0.8)
        bars3 = ax.bar(x + width, top_codes['f1_score'], width,
                       label='F1 Score', color='#e74c3c', alpha=0.8)

        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_xlabel('F-Code', fontsize=12, fontweight='bold')
        ax.set_title('Performance by F-Code (Top 15 by Frequency)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(top_codes['f_code'], rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3)

        # Add occurrence counts
        for i, (_, row) in enumerate(top_codes.iterrows()):
            ax.text(i, 1.02, f'n={row["occurrences"]}', ha='center', fontsize=8)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig3_per_code_performance.png', dpi=300, bbox_inches='tight')
        print("    Saved: fig3_per_code_performance.png")
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
        plt.savefig(self.output_dir / 'fig4_frequency_vs_performance.png',
                    dpi=300, bbox_inches='tight')
        print("    Saved: fig4_frequency_vs_performance.png")
        plt.close()

    def _plot_precision_recall_tradeoff(self):
        """Plot precision-recall tradeoff for each code."""
        print("  Creating precision-recall tradeoff chart...")

        if self.per_code_df is None:
            return

        fig, ax = plt.subplots(figsize=(10, 10))

        # Filter codes with sufficient samples
        df = self.per_code_df[self.per_code_df['occurrences'] >= 3]

        scatter = ax.scatter(
            df['recall'],
            df['precision'],
            c=df['f1_score'],
            cmap='RdYlGn',
            s=df['occurrences'] * 3,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )

        # Add diagonal lines for F1 iso-curves
        for f1 in [0.2, 0.4, 0.6, 0.8]:
            x = np.linspace(0.01, 1, 100)
            y = f1 * x / (2 * x - f1)
            valid = (y > 0) & (y <= 1)
            ax.plot(x[valid], y[valid], '--', color='gray', alpha=0.5, linewidth=1)
            if np.any(valid):
                ax.text(0.95, f1 * 0.95 / (2 * 0.95 - f1), f'F1={f1}',
                        fontsize=8, color='gray')

        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Trade-off by F-Code', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.05)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('F1 Score', fontsize=11)

        ax.text(0.02, 0.98, 'Size = Code Frequency', transform=ax.transAxes,
                fontsize=9, verticalalignment='top')

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig5_precision_recall_tradeoff.png',
                    dpi=300, bbox_inches='tight')
        print("    Saved: fig5_precision_recall_tradeoff.png")
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
        default=OUTPUT_DIR_7_ANALYSIS,
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
