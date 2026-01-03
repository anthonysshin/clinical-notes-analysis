#!/usr/bin/env python3
"""
9_ErrorAnalysis.py - Error Analysis and Confusion Matrix Generation

This script performs detailed error analysis on model predictions, including:
    1. False positive and false negative analysis
    2. Error categorization (perfect match, partial match, complete miss)
    3. Confusion matrix for top F-codes
    4. Common error patterns identification

Output:
    - Error analysis report (JSON)
    - Confusion matrix visualization (PNG)
    - Error breakdown tables (CSV)

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
    OUTPUT_DIR, OUTPUT_DIR_6F, OUTPUT_DIR_9_ERROR
)

warnings.filterwarnings('ignore')

# Apply publication-ready figure style
apply_figure_style()
sns.set_style("whitegrid")


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

    def create_confusion_matrix(self, top_n: int = 10):
        """
        Create confusion matrix for top N F-codes.

        Args:
            top_n: Number of top codes to include
        """
        print("\n" + "=" * 70)
        print("CREATING CONFUSION MATRIX")
        print("=" * 70)

        # Count code occurrences
        all_codes = Counter()
        for result in self.sample_results:
            all_codes.update(result.get('actual_codes', []))

        top_codes = [code for code, _ in all_codes.most_common(top_n)]

        # Build confusion matrix
        matrix = np.zeros((len(top_codes), len(top_codes)))
        code_to_idx = {code: i for i, code in enumerate(top_codes)}

        for result in self.sample_results:
            actual = set(result.get('actual_codes', []))
            predicted = set(result.get('predicted_codes', []))

            for actual_code in actual:
                if actual_code in code_to_idx:
                    actual_idx = code_to_idx[actual_code]

                    if actual_code in predicted:
                        # True positive
                        matrix[actual_idx, actual_idx] += 1
                    else:
                        # False negative - check if confused with another code
                        for pred_code in predicted:
                            if pred_code in code_to_idx:
                                pred_idx = code_to_idx[pred_code]
                                matrix[actual_idx, pred_idx] += 0.5

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 10))

        im = ax.imshow(matrix, cmap='Blues', aspect='auto')

        ax.set_xticks(np.arange(len(top_codes)))
        ax.set_yticks(np.arange(len(top_codes)))
        ax.set_xticklabels(top_codes, rotation=45, ha='right')
        ax.set_yticklabels(top_codes)

        # Add text annotations
        for i in range(len(top_codes)):
            for j in range(len(top_codes)):
                if matrix[i, j] > 0:
                    text_color = 'white' if matrix[i, j] > matrix.max() / 2 else 'black'
                    ax.text(j, i, f'{int(matrix[i, j])}',
                            ha='center', va='center', color=text_color, fontsize=9)

        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix (Top {top_n} F-Codes)\nDiagonal = True Positives',
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
            ax1.grid(axis='x', alpha=0.3)

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
            ax2.grid(axis='x', alpha=0.3)

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
        confusion_patterns: Dict
    ):
        """
        Save comprehensive error analysis report.

        Args:
            categories: Error categories
            fp_counter: False positive counts
            fn_counter: False negative counts
            confusion_patterns: Confusion pattern dictionary
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
            ]
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

        # Create confusion matrix
        self.create_confusion_matrix(top_n=10)

        # Create visualizations
        self.create_error_visualizations(fp_counter, fn_counter)

        # Save report
        self.save_error_report(categories, fp_counter, fn_counter, confusion_patterns)

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
