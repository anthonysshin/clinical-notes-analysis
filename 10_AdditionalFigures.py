#!/usr/bin/env python3
"""
10_AdditionalFigures.py - Generate Additional Publication Figures

This script creates additional publication-ready figures:
1. Per-code performance heatmap (top 20 codes by frequency)
2. Strategy comparison grouped by approach type
3. Precision-Recall trade-off with detailed annotations
4. Performance by code category (substance, mood, anxiety, cognitive)

Usage:
    python 10_AdditionalFigures.py
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import warnings

from config import (
    OUTPUT_DIR, OUTPUT_DIR_10_FIGURES, OUTPUT_DIR_6F, OUTPUT_DIR_9_ERROR,
    set_all_seeds, RANDOM_SEED, apply_figure_style
)

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
set_all_seeds(RANDOM_SEED)

# Apply publication-quality plotting style
apply_figure_style()

# F-code categories for grouping
FCODE_CATEGORIES = {
    'Substance Use': ['F10', 'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19'],
    'Mood Disorders': ['F30', 'F31', 'F32', 'F33', 'F34', 'F39'],
    'Anxiety Disorders': ['F40', 'F41', 'F42', 'F43', 'F44', 'F45', 'F48'],
    'Cognitive Disorders': ['F00', 'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F09'],
    'Psychotic Disorders': ['F20', 'F21', 'F22', 'F23', 'F24', 'F25', 'F28', 'F29'],
    'Other': ['F50', 'F51', 'F60', 'F63', 'F70', 'F79', 'F80', 'F84', 'F90', 'F91', 'F99']
}

# Strategy info (must match actual folder names in outputs/)
STRATEGY_INFO = {
    '6g1_BaseModel_ZeroShot': {'name': 'Base ZS', 'approach': 'Base', 'color': '#95a5a6'},
    '6g2_BaseModel_CoT': {'name': 'Base CoT', 'approach': 'Base', 'color': '#7f8c8d'},
    '6a_ZeroShotBaseline': {'name': 'Zero-Shot', 'approach': 'Chunking', 'color': '#3498db'},
    '6b_FewShotExemplar': {'name': 'Few-Shot', 'approach': 'Chunking', 'color': '#2ecc71'},
    '6c_RuleConstrained': {'name': 'Rule-Constrained', 'approach': 'Chunking', 'color': '#9b59b6'},
    '6d_ChainOfThought': {'name': 'Chain-of-Thought', 'approach': 'Chunking+CoT', 'color': '#e74c3c'},
    '6e_KeywordAugmented': {'name': 'Keyword-Aug', 'approach': 'Keyword', 'color': '#f39c12'},
    '6f_KeywordAugmentedCoT': {'name': 'Keyword+CoT', 'approach': 'Keyword+CoT', 'color': '#27ae60'},
}

# Output directory for figures (separate from strategy results directory)
FIGURES_OUTPUT_DIR = Path(OUTPUT_DIR_10_FIGURES)


def get_code_category(code):
    """Get category for an F-code."""
    prefix = code[:3] if len(code) >= 3 else code
    for category, prefixes in FCODE_CATEGORIES.items():
        if prefix in prefixes:
            return category
    return 'Other'


def load_all_results():
    """Load results from all strategies."""
    outputs_dir = Path(OUTPUT_DIR)  # Use OUTPUT_DIR from config (not FIGURES_OUTPUT_DIR)
    results = {}

    for strategy_id in STRATEGY_INFO.keys():
        # Find directory matching strategy ID
        matches = list(outputs_dir.glob(f"{strategy_id}*"))
        if not matches:
            continue
        strategy_dir = matches[0]

        # Find latest JSON
        json_files = list(strategy_dir.glob("evaluation_results_*.json"))
        if not json_files:
            continue
        json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

        with open(json_files[0], 'r') as f:
            results[strategy_id] = json.load(f)

    return results


def create_per_code_heatmap(results):
    """Create heatmap showing per-code performance across strategies."""
    print("Creating per-code performance heatmap...")

    # Get best strategy's per-code performance
    best_strategy = '6f_KeywordAugmentedCoT'
    if best_strategy not in results:
        print("  Best strategy not found, skipping...")
        return

    # Load per-code performance (find most recent file)
    per_code_files = list(Path(OUTPUT_DIR_6F).glob('per_code_performance_*.csv'))
    if not per_code_files:
        print("  Per-code file not found, skipping...")
        return
    per_code_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    per_code_file = per_code_files[0]

    df = pd.read_csv(per_code_file)

    # Get top 20 codes by occurrences (frequency)
    df_sorted = df.sort_values('occurrences', ascending=False).head(20)

    # Create heatmap data
    fig, ax = plt.subplots(figsize=(10, 8))

    # Prepare data for heatmap
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
    plt.savefig(FIGURES_OUTPUT_DIR / 'fig6_per_code_heatmap.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_OUTPUT_DIR / 'fig6_per_code_heatmap.pdf', bbox_inches='tight')
    print("  Saved: fig6_per_code_heatmap.png/pdf")
    plt.close()


def create_approach_comparison(results):
    """Create bar chart comparing approaches (Chunking vs Keyword)."""
    print("Creating approach comparison chart...")

    # Group strategies by approach
    approach_data = defaultdict(list)

    for strategy_id, data in results.items():
        if strategy_id not in STRATEGY_INFO:
            continue
        info = STRATEGY_INFO[strategy_id]
        metrics = data.get('performance_metrics', {})
        approach_data[info['approach']].append({
            'name': info['name'],
            'f1': metrics.get('micro_f1', 0),
            'precision': metrics.get('micro_precision', 0),
            'recall': metrics.get('micro_recall', 0),
            'color': info['color']
        })

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 6))

    approaches = ['Base', 'Chunking', 'Chunking+CoT', 'Keyword', 'Keyword+CoT']
    x_positions = []
    labels = []
    colors = []
    f1_scores = []

    current_x = 0
    approach_positions = {}

    for approach in approaches:
        if approach not in approach_data:
            continue
        approach_positions[approach] = current_x
        strategies = approach_data[approach]
        for strat in strategies:
            x_positions.append(current_x)
            labels.append(strat['name'])
            colors.append(strat['color'])
            f1_scores.append(strat['f1'])
            current_x += 1
        current_x += 0.5  # Gap between approach groups

    bars = ax.bar(x_positions, f1_scores, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)

    # Add value labels
    for bar, val in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Micro F1 Score', fontsize=12, fontweight='bold')
    ax.set_title('Strategy Performance by Approach Type', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(f1_scores) * 1.15)

    # Add approach group labels using a secondary x-axis at the bottom
    # First, get the figure and adjust layout
    fig = plt.gcf()
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.28)

    # Add approach labels at fixed y position in figure coordinates
    for approach, start_x in approach_positions.items():
        strategies = approach_data[approach]
        mid_x = start_x + (len(strategies) - 1) / 2
        # Convert data x to axes fraction, then to figure coordinates
        x_frac = (mid_x - ax.get_xlim()[0]) / (ax.get_xlim()[1] - ax.get_xlim()[0])
        # Get axes position in figure
        bbox = ax.get_position()
        x_fig = bbox.x0 + x_frac * bbox.width
        fig.text(x_fig, 0.02, approach, ha='center', va='bottom', fontsize=10,
                 style='italic', fontweight='bold')

    plt.savefig(FIGURES_OUTPUT_DIR / 'fig7_approach_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_OUTPUT_DIR / 'fig7_approach_comparison.pdf', bbox_inches='tight')
    print("  Saved: fig7_approach_comparison.png/pdf")
    plt.close()


def create_category_performance(results):
    """Create performance breakdown by F-code category."""
    print("Creating category performance chart...")

    # Load predictions from best strategy (find most recent file)
    pred_files = list(Path(OUTPUT_DIR_6F).glob('predictions_*.csv'))
    if not pred_files:
        print("  Predictions file not found, skipping...")
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

    bars1 = ax.bar(x - width, precisions, width, label='Precision', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x, recalls, width, label='Recall', color='#e74c3c', alpha=0.8)
    bars3 = ax.bar(x + width, f1_scores, width, label='F1', color='#27ae60', alpha=0.8)

    ax.set_xlabel('F-Code Category', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Performance by F-Code Category\nKeyword + CoT Strategy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_OUTPUT_DIR / 'fig8_category_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_OUTPUT_DIR / 'fig8_category_performance.pdf', bbox_inches='tight')
    print("  Saved: fig8_category_performance.png/pdf")
    plt.close()


def create_detailed_pr_curve(results):
    """Create detailed precision-recall plot with annotations."""
    print("Creating detailed precision-recall plot...")

    fig, ax = plt.subplots(figsize=(10, 8))

    for strategy_id, data in results.items():
        if strategy_id not in STRATEGY_INFO:
            continue

        info = STRATEGY_INFO[strategy_id]
        metrics = data.get('performance_metrics', {})

        precision = metrics.get('micro_precision', 0)
        recall = metrics.get('micro_recall', 0)
        f1 = metrics.get('micro_f1', 0)

        # Plot point
        ax.scatter(recall, precision, s=200, c=info['color'],
                   alpha=0.8, edgecolors='black', linewidth=1.5, zorder=5)

        # Add label with F1
        offset = (10, 10) if precision > 0.4 else (10, -15)
        ax.annotate(f"{info['name']}\nF1={f1:.3f}",
                    (recall, precision),
                    xytext=offset, textcoords='offset points',
                    fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Add F1 iso-curves
    for f1_val in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
        x = np.linspace(0.01, 1, 100)
        y = f1_val * x / (2 * x - f1_val)
        valid = (y > 0) & (y <= 1)
        ax.plot(x[valid], y[valid], '--', color='gray', alpha=0.4, linewidth=1)
        # Label the curve
        if np.any(valid):
            label_x = 0.95
            label_y = f1_val * label_x / (2 * label_x - f1_val)
            if 0 < label_y < 1:
                ax.text(label_x, label_y, f'F1={f1_val}', fontsize=8, color='gray', alpha=0.7)

    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Trade-off Across Strategies', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # Add legend for approach types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#95a5a6', label='Base Model'),
        Patch(facecolor='#3498db', label='Chunking'),
        Patch(facecolor='#e74c3c', label='Chunking+CoT'),
        Patch(facecolor='#f39c12', label='Keyword'),
        Patch(facecolor='#27ae60', label='Keyword+CoT'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', title='Approach')

    plt.tight_layout()
    plt.savefig(FIGURES_OUTPUT_DIR / 'fig9_detailed_pr.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_OUTPUT_DIR / 'fig9_detailed_pr.pdf', bbox_inches='tight')
    print("  Saved: fig9_detailed_pr.png/pdf")
    plt.close()


def create_error_summary_figure():
    """Create summary figure of error patterns."""
    print("Creating error summary figure...")

    # Load error analysis
    error_file = Path(OUTPUT_DIR_9_ERROR) / 'error_analysis_report.json'
    if not error_file.exists():
        print("  Error analysis not found, skipping...")
        return

    with open(error_file, 'r') as f:
        error_data = json.load(f)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 1. Error categories pie chart
    ax1 = axes[0]
    categories = error_data['error_categories']
    sizes = [categories['perfect_match']['count'],
             categories['partial_match']['count'],
             categories['complete_miss']['count']]
    # Calculate percentages dynamically
    total = sum(sizes)
    pcts = [s / total * 100 for s in sizes]
    labels = [f'Perfect Match\n({pcts[0]:.1f}%)',
              f'Partial Match\n({pcts[1]:.1f}%)',
              f'Complete Miss\n({pcts[2]:.1f}%)']
    colors = ['#27ae60', '#f39c12', '#e74c3c']

    ax1.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90)
    ax1.set_title('Error Category Distribution', fontsize=12, fontweight='bold')

    # 2. Top False Positives
    ax2 = axes[1]
    fp_data = error_data['top_false_positives'][:10]
    codes = [item['code'] for item in fp_data]
    counts = [item['count'] for item in fp_data]

    ax2.barh(range(len(codes)), counts, color='#e74c3c', alpha=0.7)
    ax2.set_yticks(range(len(codes)))
    ax2.set_yticklabels(codes)
    ax2.set_xlabel('Count')
    ax2.set_title('Top 10 False Positives', fontsize=12, fontweight='bold')
    ax2.invert_yaxis()

    # 3. Top False Negatives
    ax3 = axes[2]
    fn_data = error_data['top_false_negatives'][:10]
    codes = [item['code'] for item in fn_data]
    counts = [item['count'] for item in fn_data]

    ax3.barh(range(len(codes)), counts, color='#3498db', alpha=0.7)
    ax3.set_yticks(range(len(codes)))
    ax3.set_yticklabels(codes)
    ax3.set_xlabel('Count')
    ax3.set_title('Top 10 False Negatives', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()

    plt.suptitle('Error Analysis Summary - Keyword + CoT Strategy', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(FIGURES_OUTPUT_DIR / 'fig10_error_summary.png', dpi=300, bbox_inches='tight')
    plt.savefig(FIGURES_OUTPUT_DIR / 'fig10_error_summary.pdf', bbox_inches='tight')
    print("  Saved: fig10_error_summary.png/pdf")
    plt.close()


def main():
    print("=" * 70)
    print("GENERATING ADDITIONAL PUBLICATION FIGURES")
    print("=" * 70)

    # Create output directory for figures
    FIGURES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load results
    print("\nLoading results from all strategies...")
    results = load_all_results()
    print(f"Loaded {len(results)} strategies")

    # Generate figures
    print("\n" + "=" * 70)
    print("CREATING FIGURES")
    print("=" * 70 + "\n")

    create_per_code_heatmap(results)
    create_approach_comparison(results)
    create_category_performance(results)
    create_detailed_pr_curve(results)
    create_error_summary_figure()

    print("\n" + "=" * 70)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {FIGURES_OUTPUT_DIR}")
    print("\nGenerated files:")
    for f in sorted(FIGURES_OUTPUT_DIR.glob('*')):
        print(f"  - {f.name}")

    return 0


if __name__ == "__main__":
    exit(main())
