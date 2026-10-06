#!/usr/bin/env python3
"""
3_EDA.py - Exploratory Data Analysis for MIMIC-IV Psychiatric F-Code Dataset

This script performs comprehensive exploratory data analysis on the prepared
MIMIC-IV dataset, generating statistics, visualizations, and summary reports.

Analyses Performed:
    1. Basic dataset statistics (splits, patients, admissions)
    2. F-code distribution analysis (frequency, coverage)
    3. Clinical note text characteristics (length, word count, token count)
    4. Label co-occurrence patterns
    5. Clinical note section structure analysis
    6. Substance use severity level analysis (F10-F19)
    7. Chunking statistics analysis (training data)

Output:
    - 5 visualization figures (PNG): 2 main + 3 supplementary
    - 4 summary CSV files
    - Console report with detailed statistics

Figures Generated:
    - fig1: F-code distribution (top 20 + long-tail)
    - fig2: Text length distribution (words & tokens)
    - figS1: Co-occurrence heatmap (supplementary)
    - figS2: Section structure analysis (supplementary)
    - figS3: Substance use severity analysis (supplementary)

Usage:
    python 3_EDA.py --data-dir ./data --output-dir ./outputs/3_EDA

Author: Clinical Note Analysis Study
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
from collections import Counter
import warnings

# Import centralized config for reproducibility, colors, and figure style
from config import (
    RANDOM_SEED, set_all_seeds, COLORS, apply_figure_style,
    DATA_DIR, OUTPUT_DIR_3_EDA, CHUNK_SIZE, CHUNK_OVERLAP, BASE_MODEL_NAME
)

# Import tokenizer for token counting
from transformers import AutoTokenizer

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
set_all_seeds(RANDOM_SEED)


def format_f_code(code: str) -> str:
    """
    Format F-code with period at 3rd position if not present.

    Args:
        code: Raw F-code string (e.g., "F329" or "F32.9")

    Returns:
        Formatted F-code (e.g., "F32.9")
    """
    code = code.strip()
    if len(code) > 3 and '.' not in code:
        return f"{code[:3]}.{code[3:]}"
    return code


# Set plotting style
sns.set_style("whitegrid")
apply_figure_style()
plt.rcParams['figure.figsize'] = (14, 8)


class MIMICDatasetEDA:
    """Comprehensive EDA for MIMIC-IV Psychiatric F-Code Dataset."""

    def __init__(self, data_dir: str, output_dir: str):
        """
        Initialize EDA analyzer.

        Args:
            data_dir: Directory containing processed MIMIC-IV CSV files
            output_dir: Directory to save EDA outputs
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.df = None
        self.stats = {}

    def load_data(self):
        """Load train, validation, and test datasets."""
        print("=" * 70)
        print("LOADING MIMIC-IV PSYCHIATRIC F-CODE DATASET")
        print("=" * 70 + "\n")

        all_data = []

        for split in ['train', 'val', 'test']:
            file_path = self.data_dir / f'mimic_iv_{split}_data.csv'

            if file_path.exists():
                df = pd.read_csv(file_path)
                df['split'] = split
                all_data.append(df)
                print(f"  Loaded {split}: {len(df):,} samples")
            else:
                print(f"  Warning: {file_path} not found")

        if all_data:
            self.df = pd.concat(all_data, ignore_index=True)
            print(f"\n  Total dataset: {len(self.df):,} samples")
        else:
            raise FileNotFoundError("No data files found in the specified directory")

    def load_chunked_data(self):
        """Load chunked JSON data for analysis."""
        print("\n" + "=" * 70)
        print("LOADING CHUNKED TRAINING DATA")
        print("=" * 70 + "\n")

        self.chunked_data = {'train': [], 'val': [], 'test': []}

        for split in ['train', 'val', 'test']:
            file_path = self.data_dir / f'{split}_chunked.json'

            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    self.chunked_data[split] = data
                    print(f"  Loaded {split}_chunked.json: {len(data):,} samples")
            else:
                print(f"  Warning: {file_path} not found")

        # Load dataset_info.json if available
        info_path = self.data_dir / 'dataset_info.json'
        if info_path.exists():
            with open(info_path, 'r') as f:
                self.dataset_info = json.load(f)
            print(f"  Loaded dataset_info.json")
        else:
            self.dataset_info = None

    def basic_statistics(self):
        """Calculate and display basic dataset statistics."""
        print("\n" + "=" * 70)
        print("1. BASIC DATASET STATISTICS")
        print("=" * 70 + "\n")

        # Split sizes
        print("Dataset Split Sizes:")
        split_counts = self.df['split'].value_counts()
        for split in ['train', 'val', 'test']:
            if split in split_counts.index:
                count = split_counts[split]
                pct = count / len(self.df) * 100
                print(f"  {split.capitalize():<12}: {count:>6,} samples ({pct:5.1f}%)")

        print(f"  {'Total':<12}: {len(self.df):>6,} samples")

        # Unique patients and admissions
        unique_patients = self.df['subject_id'].nunique()
        unique_admissions = self.df['hadm_id'].nunique()

        print(f"\nUnique Patients (subject_id): {unique_patients:,}")
        print(f"Unique Admissions (hadm_id):  {unique_admissions:,}")

        # Patients with multiple admissions
        multi_admit = self.df.groupby('subject_id')['hadm_id'].nunique()
        multi_admit_count = (multi_admit > 1).sum()
        print(f"Patients with >1 admission:   {multi_admit_count:,} "
              f"({multi_admit_count / unique_patients * 100:.1f}%)")

        self.stats['basic'] = {
            'total_samples': len(self.df),
            'train_samples': int(split_counts.get('train', 0)),
            'val_samples': int(split_counts.get('val', 0)),
            'test_samples': int(split_counts.get('test', 0)),
            'unique_patients': unique_patients,
            'unique_admissions': unique_admissions,
            'patients_multi_admission': multi_admit_count
        }

    def analyze_fcode_distribution(self):
        """Analyze F-code distribution across the dataset."""
        print("\n" + "=" * 70)
        print("2. F-CODE DISTRIBUTION ANALYSIS")
        print("=" * 70 + "\n")

        # Extract all F-codes
        all_fcodes = []
        labels_per_sample = []

        for _, row in self.df.iterrows():
            codes = [format_f_code(c.strip()) for c in str(row['f_codes_str']).split(',')]
            all_fcodes.extend(codes)
            labels_per_sample.append(len(codes))

        self.df['label_count'] = labels_per_sample

        # Count occurrences
        fcode_counter = Counter(all_fcodes)
        total_codes = len(all_fcodes)
        unique_codes = len(fcode_counter)

        print(f"Total F-code instances:  {total_codes:,}")
        print(f"Unique F-codes:          {unique_codes}")
        print(f"Average codes/sample:    {np.mean(labels_per_sample):.2f} "
              f"± {np.std(labels_per_sample):.2f}")
        print(f"Median codes/sample:     {np.median(labels_per_sample):.0f}")
        print(f"Min codes/sample:        {min(labels_per_sample)}")
        print(f"Max codes/sample:        {max(labels_per_sample)}")

        # Top codes
        print(f"\nTop 20 Most Frequent F-Codes:")
        print(f"{'Rank':<6} {'Code':<12} {'Count':<8} {'% of Instances':<16} {'% of Samples'}")
        print("-" * 70)

        for rank, (code, count) in enumerate(fcode_counter.most_common(20), 1):
            pct_instances = count / total_codes * 100
            pct_samples = count / len(self.df) * 100
            print(f"{rank:<6} {code:<12} {count:<8,} {pct_instances:>6.2f}%          "
                  f"{pct_samples:>6.2f}%")

        # Coverage analysis
        print("\nCoverage Analysis:")
        for k in [3, 5, 10, 20, 50]:
            top_k_codes = fcode_counter.most_common(k)
            cumulative_count = sum(count for _, count in top_k_codes)
            coverage = cumulative_count / total_codes * 100
            print(f"  Top-{k:<2} codes cover: {coverage:.1f}% of all F-code instances")

        self.stats['fcode'] = {
            'total_instances': total_codes,
            'unique_codes': unique_codes,
            'avg_codes_per_sample': round(np.mean(labels_per_sample), 2),
            'median_codes_per_sample': int(np.median(labels_per_sample)),
            'min_codes_per_sample': min(labels_per_sample),
            'max_codes_per_sample': max(labels_per_sample)
        }

        return fcode_counter, labels_per_sample

    def analyze_text_characteristics(self):
        """Analyze clinical note text characteristics including token counts."""
        print("\n" + "=" * 70)
        print("3. CLINICAL NOTE TEXT CHARACTERISTICS")
        print("=" * 70 + "\n")

        # Calculate text lengths
        self.df['text_length'] = self.df['text'].str.len()
        self.df['word_count'] = self.df['text'].str.split().str.len()

        # Calculate token counts using model tokenizer
        print("Loading tokenizer for token counting...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
            print("  Counting tokens (this may take a moment)...")

            # Sample for faster processing if dataset is large
            if len(self.df) > 10000:
                sample_df = self.df.sample(n=10000, random_state=RANDOM_SEED)
                print(f"  Using sample of 10,000 for token statistics")
            else:
                sample_df = self.df

            token_counts = []
            for idx, text in enumerate(sample_df['text']):
                tokens = tokenizer.encode(text, add_special_tokens=False)
                token_counts.append(len(tokens))
                if (idx + 1) % 2000 == 0:
                    print(f"    Processed {idx + 1}/{len(sample_df)} samples...")

            self.token_counts = token_counts
            self.has_token_data = True
            print(f"  Token counting complete.")
        except Exception as e:
            print(f"  Warning: Could not load tokenizer: {e}")
            print(f"  Skipping token analysis.")
            self.has_token_data = False
            self.token_counts = []

        print("\nWord Count Statistics:")
        print(f"  Mean:    {self.df['word_count'].mean():>10,.1f} words")
        print(f"  Median:  {self.df['word_count'].median():>10,.1f} words")
        print(f"  Std:     {self.df['word_count'].std():>10,.1f} words")
        print(f"  Min:     {self.df['word_count'].min():>10,} words")
        print(f"  Max:     {self.df['word_count'].max():>10,} words")

        if self.has_token_data:
            print("\nToken Count Statistics:")
            print(f"  Mean:    {np.mean(self.token_counts):>10,.1f} tokens")
            print(f"  Median:  {np.median(self.token_counts):>10,.1f} tokens")
            print(f"  Std:     {np.std(self.token_counts):>10,.1f} tokens")
            print(f"  Min:     {min(self.token_counts):>10,} tokens")
            print(f"  Max:     {max(self.token_counts):>10,} tokens")

            # Chunking analysis
            exceed_limit = sum(1 for t in self.token_counts if t > CHUNK_SIZE)
            print(f"\nChunking Analysis (chunk size: {CHUNK_SIZE} tokens, overlap: {CHUNK_OVERLAP}):")
            print(f"  Samples exceeding chunk: {exceed_limit:,} ({exceed_limit/len(self.token_counts)*100:.1f}%) - will be split")
            print(f"  Samples within chunk:    {len(self.token_counts) - exceed_limit:,} ({(len(self.token_counts) - exceed_limit)/len(self.token_counts)*100:.1f}%) - single chunk")

        # Percentiles
        print("\nToken Count Percentiles:" if self.has_token_data else "\nWord Count Percentiles:")
        data_for_pct = self.token_counts if self.has_token_data else self.df['word_count']
        for pct in [25, 50, 75, 90, 95, 99]:
            val = np.percentile(data_for_pct, pct)
            unit = "tokens" if self.has_token_data else "words"
            print(f"  {pct}th percentile: {val:>10,.0f} {unit}")

        self.stats['text'] = {
            'mean_chars': round(self.df['text_length'].mean(), 1),
            'median_chars': round(self.df['text_length'].median(), 1),
            'mean_words': round(self.df['word_count'].mean(), 1),
            'median_words': round(self.df['word_count'].median(), 1),
            'mean_tokens': round(np.mean(self.token_counts), 1) if self.has_token_data else None,
            'median_tokens': round(np.median(self.token_counts), 1) if self.has_token_data else None
        }

    def analyze_information_location(self):
        """Analyze clinical note section structure with explicit token-based calculations."""
        print("\n" + "=" * 70)
        print("5. CLINICAL NOTE SECTION STRUCTURE ANALYSIS")
        print("=" * 70 + "\n")

        import re
        from collections import defaultdict

        # Load tokenizer for token-based position calculation
        print("Loading tokenizer for token position analysis...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
        except Exception as e:
            print(f"  Warning: Could not load tokenizer: {e}")
            self.section_analysis = None
            return None

        # Clinical note section patterns with detailed descriptions and importance levels
        # Importance: High = directly contains F-code relevant info, Medium = contextual, Low = not relevant
        section_info = {
            'Chief Complaint': {
                'pattern': r'chief complaint[s]?:?',
                'contents': 'Reason for hospital visit, presenting symptoms',
                'importance': 'High',
                'importance_reason': 'Often mentions psychiatric symptoms'
            },
            'History of Present Illness': {
                'pattern': r'history of present illness:?',
                'contents': 'Detailed current illness narrative, symptom progression',
                'importance': 'High',
                'importance_reason': 'Contains psychiatric symptoms and history'
            },
            'Past Medical History': {
                'pattern': r'past medical history:?',
                'contents': 'Prior diagnoses, chronic conditions, past psychiatric diagnoses',
                'importance': 'High',
                'importance_reason': 'Lists prior F-code diagnoses'
            },
            'Social History': {
                'pattern': r'social history:?',
                'contents': 'Substance use (alcohol, tobacco, drugs), occupation, living situation',
                'importance': 'High',
                'importance_reason': 'Critical for substance use F-codes (F10-F19)'
            },
            'Family History': {
                'pattern': r'family history:?',
                'contents': 'Family medical and psychiatric history',
                'importance': 'Medium',
                'importance_reason': 'Contextual for psychiatric conditions'
            },
            'Physical Exam': {
                'pattern': r'physical exam[ination]?:?',
                'contents': 'Physical examination findings, vital signs',
                'importance': 'Low',
                'importance_reason': 'Rarely contains psychiatric diagnostic info'
            },
            'Pertinent Results': {
                'pattern': r'pertinent results:?',
                'contents': 'Laboratory values, imaging results, diagnostic tests',
                'importance': 'Low',
                'importance_reason': 'Lab values not used for F-code classification'
            },
            'Brief Hospital Course': {
                'pattern': r'brief hospital course:?',
                'contents': 'Summary of hospital stay, treatments, clinical events',
                'importance': 'Medium',
                'importance_reason': 'May mention psychiatric consultations'
            },
            'Discharge Diagnosis': {
                'pattern': r'discharge diagnosis:?|discharge diagnos[ie]s:?',
                'contents': 'Final diagnoses including ICD-10 codes (F-codes)',
                'importance': 'High',
                'importance_reason': 'Contains EXPLICIT F-code diagnoses'
            },
        }

        print(f"Analyzing section positions in tokens (chunk size: {CHUNK_SIZE})...")

        # Sample for analysis - use 10,000 for robust statistics
        sample_size = min(10000, len(self.df))
        sample_df = self.df.sample(n=sample_size, random_state=RANDOM_SEED)

        # Track section statistics
        section_stats = defaultdict(lambda: {
            'found_count': 0,
            'token_positions': [],
            'within_first_chunk': 0,
            'beyond_first_chunk': 0,
        })

        for idx, (_, row) in enumerate(sample_df.iterrows()):
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{sample_size}...")

            text = row['text']
            text_lower = text.lower()

            if len(text) == 0:
                continue

            for section, info in section_info.items():
                match = re.search(info['pattern'], text_lower)
                if match:
                    char_pos = match.start()
                    # Calculate token position
                    text_before = text[:char_pos]
                    tokens_before = tokenizer.encode(text_before, add_special_tokens=False)
                    token_pos = len(tokens_before)

                    section_stats[section]['found_count'] += 1
                    section_stats[section]['token_positions'].append(token_pos)

                    if token_pos <= CHUNK_SIZE:
                        section_stats[section]['within_first_chunk'] += 1
                    else:
                        section_stats[section]['beyond_first_chunk'] += 1

        print(f"\n  Analyzed {sample_size:,} samples")

        # Print results
        print("\n" + "-" * 80)
        print(f"{'Section':<30} {'Found':<8} {'Median':<10} {'In 1st Chunk':<12} {'Later Chunks':<12}")
        print(f"{'':30} {'':8} {'(tokens)':<10} {'(≤' + str(CHUNK_SIZE) + ')':<12} {'(>' + str(CHUNK_SIZE) + ')':<12}")
        print("-" * 80)

        results = []
        for section in section_info.keys():
            stats = section_stats[section]
            found = stats['found_count']
            found_pct = found / sample_size * 100

            if found > 0:
                median_token = np.median(stats['token_positions'])
                within_pct = stats['within_first_chunk'] / found * 100
                beyond_pct = stats['beyond_first_chunk'] / found * 100

                info = section_info[section]
                importance = info['importance']
                marker = "★" if importance == 'High' else ("◆" if importance == 'Medium' else " ")

                print(f"{marker}{section:<29} {found_pct:>5.1f}%  {median_token:>8.0f}   "
                      f"{within_pct:>8.1f}%    {beyond_pct:>8.1f}%")

                results.append({
                    'section': section,
                    'contents': info['contents'],
                    'importance': importance,
                    'importance_reason': info['importance_reason'],
                    'found_pct': found_pct,
                    'median_token_pos': median_token,
                    'in_first_chunk_pct': within_pct,
                    'in_later_chunks_pct': beyond_pct,
                    'token_positions': stats['token_positions'],  # Store all positions for box plot
                })

        print("-" * 80)
        print("★ = High importance, ◆ = Medium importance")

        # Key insights
        print(f"\n{'=' * 60}")
        print("KEY FINDINGS (Chunking Context):")
        print(f"{'=' * 60}")

        # Find Discharge Diagnosis stats
        dx_result = next((r for r in results if r['section'] == 'Discharge Diagnosis'), None)
        if dx_result:
            print(f"\n• Discharge Diagnosis (contains explicit F-codes):")
            print(f"    Median position: {dx_result['median_token_pos']:.0f} tokens")
            print(f"    In first chunk: {dx_result['in_first_chunk_pct']:.1f}%")
            print(f"    In later chunks: {dx_result['in_later_chunks_pct']:.1f}%")
            print(f"    → Chunking with overlap ensures this section is captured")

        # Summary of high-importance sections
        high_importance = [r for r in results if r['importance'] == 'High']
        early_high = [r for r in high_importance if r['in_first_chunk_pct'] >= 99]
        late_high = [r for r in high_importance if r['in_first_chunk_pct'] < 99]

        print(f"\n• High-importance sections in first chunk: {len(early_high)}")
        for r in early_high:
            print(f"    - {r['section']}: {r['in_first_chunk_pct']:.1f}% in first chunk")

        print(f"\n• High-importance sections needing later chunks: {len(late_high)}")
        for r in late_high:
            print(f"    - {r['section']}: {r['in_later_chunks_pct']:.1f}% in later chunks")

        # Store for visualization
        self.section_analysis = {
            'results': results,
            'sample_size': sample_size,
            'chunk_size': CHUNK_SIZE,
        }

        # Save to CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(self.output_dir / 'section_position_analysis.csv', index=False)
        print(f"\n  Saved: section_position_analysis.csv")

        return results

    def analyze_severity_levels(self, fcode_counter):
        """Analyze severity level distribution in substance use codes (F10-F19)."""
        print("\n" + "=" * 70)
        print("6. SUBSTANCE USE SEVERITY LEVEL ANALYSIS")
        print("=" * 70 + "\n")

        # Substance use code prefixes
        substance_prefixes = {
            'F10': 'Alcohol',
            'F11': 'Opioid',
            'F12': 'Cannabis',
            'F13': 'Sedative',
            'F14': 'Cocaine',
            'F15': 'Stimulant',
            'F16': 'Hallucinogen',
            'F17': 'Nicotine',
            'F18': 'Inhalant',
            'F19': 'Multiple/Other'
        }

        # Collect substance use codes
        substance_codes = {}
        for code, count in fcode_counter.items():
            for prefix, name in substance_prefixes.items():
                if code.startswith(prefix):
                    if prefix not in substance_codes:
                        substance_codes[prefix] = {'name': name, 'codes': {}}
                    substance_codes[prefix]['codes'][code] = count
                    break

        print("Substance Use Disorder Codes (F10-F19):")
        print("-" * 60)

        total_substance_instances = 0
        severity_counts = {'Abuse': 0, 'Dependence': 0, 'Unspecified': 0, 'Other': 0}
        remission_counts = {'Active': 0, 'In Remission': 0, 'Unknown': 0}

        for prefix in sorted(substance_codes.keys()):
            data = substance_codes[prefix]
            prefix_total = sum(data['codes'].values())
            total_substance_instances += prefix_total
            print(f"\n{prefix} ({data['name']}): {prefix_total:,} instances")

            # Sort codes by frequency
            for code, count in sorted(data['codes'].items(), key=lambda x: -x[1])[:5]:
                print(f"    {code}: {count:,}")

                # Classify severity
                if len(code) >= 5:  # e.g., F10.10
                    fourth_char = code[4] if len(code) > 4 else ''
                    if fourth_char == '1':
                        severity_counts['Abuse'] += count
                    elif fourth_char == '2':
                        severity_counts['Dependence'] += count
                    elif fourth_char == '9':
                        severity_counts['Unspecified'] += count
                    else:
                        severity_counts['Other'] += count

                    # Classify remission
                    if len(code) >= 6:  # e.g., F10.10 or F10.21
                        fifth_char = code[5] if len(code) > 5 else '0'
                        if fifth_char == '1':
                            remission_counts['In Remission'] += count
                        elif fifth_char == '0':
                            remission_counts['Active'] += count
                        else:
                            remission_counts['Unknown'] += count
                    else:
                        remission_counts['Unknown'] += count
                else:
                    severity_counts['Other'] += count
                    remission_counts['Unknown'] += count

        print(f"\n{'=' * 60}")
        print(f"Total Substance Use Instances: {total_substance_instances:,}")
        print(f"  ({total_substance_instances / sum(fcode_counter.values()) * 100:.1f}% of all F-code instances)")

        print(f"\nSeverity Distribution:")
        for severity, count in sorted(severity_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = count / total_substance_instances * 100
                print(f"  {severity:<15}: {count:>8,} ({pct:5.1f}%)")

        print(f"\nRemission Status Distribution:")
        for status, count in sorted(remission_counts.items(), key=lambda x: -x[1]):
            if count > 0:
                pct = count / total_substance_instances * 100
                print(f"  {status:<15}: {count:>8,} ({pct:5.1f}%)")

        self.substance_analysis = {
            'total_instances': total_substance_instances,
            'severity_counts': severity_counts,
            'remission_counts': remission_counts,
            'by_substance': substance_codes
        }

        return substance_codes, severity_counts, remission_counts

    def analyze_label_cooccurrence(self, fcode_counter):
        """Analyze label co-occurrence patterns."""
        print("\n" + "=" * 70)
        print("4. LABEL CO-OCCURRENCE ANALYSIS")
        print("=" * 70 + "\n")

        # Get top 10 codes
        top_codes = [code for code, _ in fcode_counter.most_common(10)]

        # Build co-occurrence matrix
        cooccur_matrix = np.zeros((len(top_codes), len(top_codes)))

        for _, row in self.df.iterrows():
            codes = [format_f_code(c.strip()) for c in str(row['f_codes_str']).split(',')]
            for i, code1 in enumerate(top_codes):
                for j, code2 in enumerate(top_codes):
                    if code1 in codes and code2 in codes:
                        cooccur_matrix[i, j] += 1

        # Print top co-occurrences (excluding diagonal)
        cooccur_pairs = []
        for i in range(len(top_codes)):
            for j in range(i + 1, len(top_codes)):
                if cooccur_matrix[i, j] > 0:
                    cooccur_pairs.append(
                        (top_codes[i], top_codes[j], int(cooccur_matrix[i, j]))
                    )

        cooccur_pairs.sort(key=lambda x: x[2], reverse=True)

        print("Top 15 F-Code Co-Occurrences:")
        print(f"{'Rank':<6} {'Code 1':<12} {'Code 2':<12} {'Count':<10} {'% of Samples'}")
        print("-" * 60)

        for rank, (code1, code2, count) in enumerate(cooccur_pairs[:15], 1):
            pct = count / len(self.df) * 100
            print(f"{rank:<6} {code1:<12} {code2:<12} {count:<10,} {pct:.1f}%")

        return cooccur_matrix, top_codes

    def analyze_chunking_statistics(self, full_fcode_counter):
        """Analyze chunking statistics on training data."""
        print("\n" + "=" * 70)
        print("7. CHUNKING STATISTICS ANALYSIS")
        print("=" * 70 + "\n")

        if not self.chunked_data.get('train'):
            print("  Chunked data not available. Skipping analysis.")
            return None, None, None

        # Get full training data
        train_df = self.df[self.df['split'] == 'train'].copy()
        full_train_count = len(train_df)
        chunked_train = self.chunked_data['train']
        chunked_train_count = len(chunked_train)

        print("--- Sample Count ---")
        print(f"  Full CSV (train):     {full_train_count:>10,} samples")
        print(f"  Chunked JSON (train): {chunked_train_count:>10,} samples")
        print(f"  Status:               ALL samples processed (1:1 mapping)")

        # Extract F-codes from chunked data
        chunked_fcodes = []
        for sample in chunked_train:
            output = sample.get('output', '[]')
            try:
                codes = json.loads(output)
                chunked_fcodes.extend(codes)
            except json.JSONDecodeError:
                pass

        chunked_fcode_counter = Counter(chunked_fcodes)

        # Extract full CSV F-codes for comparison
        full_train_fcodes = []
        for _, row in train_df.iterrows():
            codes = [format_f_code(c.strip()) for c in str(row['f_codes_str']).split(',')]
            full_train_fcodes.extend(codes)
        full_train_fcode_counter = Counter(full_train_fcodes)

        print("\n--- F-Code Instance Comparison ---")
        print(f"  Full CSV F-code instances:    {len(full_train_fcodes):>10,}")
        print(f"  Chunked F-code instances:     {len(chunked_fcodes):>10,}")
        print(f"  Ratio:                        {len(chunked_fcodes) / len(full_train_fcodes) * 100:>10.1f}%")

        print(f"\n  Unique F-codes in full CSV:   {len(full_train_fcode_counter):>10}")
        print(f"  Unique F-codes in chunked:    {len(chunked_fcode_counter):>10}")

        # Find missing codes (should be none with chunking)
        missing_codes = set(full_train_fcode_counter.keys()) - set(chunked_fcode_counter.keys())
        print(f"  F-codes MISSING from training:{len(missing_codes):>10}")

        if missing_codes:
            missing_with_counts = [(c, full_train_fcode_counter[c]) for c in missing_codes]
            missing_with_counts.sort(key=lambda x: -x[1])
            print("\n  Top 10 most frequent missing codes:")
            for code, count in missing_with_counts[:10]:
                print(f"    {code}: {count} instances in full CSV")

        # Analyze rare codes in training set
        print("\n--- Rare Code Analysis (Training Set) ---")
        freq_counts = list(chunked_fcode_counter.values())
        print(f"  Mean frequency:   {sum(freq_counts) / len(freq_counts):>8.1f}")
        print(f"  Median frequency: {sorted(freq_counts)[len(freq_counts) // 2]:>8}")

        ranges = [
            (1, 1, 'Singleton (1)'),
            (2, 5, 'Very rare (2-5)'),
            (6, 10, 'Rare (6-10)'),
            (11, 50, 'Uncommon (11-50)'),
            (51, 100, 'Moderate (51-100)'),
            (101, float('inf'), 'Common (>100)')
        ]

        print(f"\n  {'Range':<22} {'Count':>8} {'% of codes':>12}")
        print("  " + "-" * 45)
        for low, high, label in ranges:
            count = sum(1 for f in freq_counts if low <= f <= high)
            pct = count / len(freq_counts) * 100
            print(f"  {label:<22} {count:>8} {pct:>11.1f}%")

        # Singleton codes warning
        singletons = [code for code, count in chunked_fcode_counter.items() if count == 1]
        print(f"\n  WARNING: {len(singletons)} codes appear only ONCE (no learning signal)")

        # Chunking statistics from dataset_info
        if self.dataset_info and 'chunking_statistics' in self.dataset_info:
            print("\n--- Chunking Statistics (from preparation) ---")
            chunk_stats = self.dataset_info['chunking_statistics']['train']
            print(f"  Total chunks:         {chunk_stats.get('total_chunks', 'N/A'):>10}")
            print(f"  Avg chunks/sample:    {chunk_stats.get('chunks_per_doc', 0):>10.2f}")
            # Derive max chunks from chunk_distribution
            chunk_dist = chunk_stats.get('chunk_distribution', {})
            max_chunks = max((int(k) for k in chunk_dist.keys()), default=0) if chunk_dist else 'N/A'
            print(f"  Max chunks/sample:    {max_chunks:>10}")
            print(f"  Multi-chunk samples:  {chunk_stats.get('multi_chunk_docs', 'N/A'):>10} ({chunk_stats.get('multi_chunk_pct', 0):.1f}%)")
            print(f"  Single-chunk samples: {chunk_stats.get('single_chunk_docs', 'N/A'):>10}")

        self.stats['chunking'] = {
            'full_train_samples': full_train_count,
            'chunked_train_samples': chunked_train_count,
            'full_fcode_instances': len(full_train_fcodes),
            'chunked_fcode_instances': len(chunked_fcodes),
            'full_unique_fcodes': len(full_train_fcode_counter),
            'chunked_unique_fcodes': len(chunked_fcode_counter),
            'missing_fcodes': len(missing_codes),
            'singleton_codes': len(singletons)
        }

        return chunked_fcode_counter, missing_codes, full_train_fcode_counter

    def create_visualizations(self, fcode_counter, labels_per_sample,
                              cooccur_matrix, top_codes, chunking_data=None):
        """Create EDA visualizations for journal submission.

        Figures:
            Main: fig1 (F-code distribution), fig2 (text length)
            Supplementary: figS1 (co-occurrence), figS2 (section structure), figS3 (severity)
        """
        print("\n" + "=" * 70)
        print("8. CREATING VISUALIZATIONS")
        print("=" * 70 + "\n")

        # Main figures
        self._plot_fcode_distribution(fcode_counter)  # fig1
        self._plot_text_length_distribution()  # fig2

        # Supplementary figures
        self._plot_cooccurrence_heatmap(cooccur_matrix, top_codes)  # figS1
        self._plot_section_structure()  # figS2
        self._plot_severity_analysis()  # figS3

    def _plot_fcode_distribution(self, fcode_counter):
        """Plot comprehensive F-code distribution (top codes + long-tail)."""
        print("  Creating F-code distribution figure...")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        # Left: Top 20 F-codes (horizontal bar chart)
        top_20 = fcode_counter.most_common(20)
        codes = [code for code, _ in top_20]
        counts = [count for _, count in top_20]

        bars = ax1.barh(range(len(codes)), counts, color=COLORS['primary'], alpha=0.8)

        # Color top 3 differently
        for i in range(min(3, len(bars))):
            bars[i].set_color(COLORS['negative'])

        ax1.set_yticks(range(len(codes)))
        ax1.set_yticklabels(codes)
        ax1.set_xlabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_ylabel('F-Code', fontsize=12, fontweight='bold')
        ax1.set_title('Top 20 Most Frequent F-Codes', fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)

        # Add count labels with margin
        ax1.margins(x=0.15)
        for i, (bar, count) in enumerate(zip(bars, counts)):
            ax1.text(count, i, f' {count:,}', va='center', fontsize=9)

        # Right: Long-tail distribution
        all_codes = sorted(fcode_counter.items(), key=lambda x: x[1], reverse=True)
        all_counts = [count for _, count in all_codes]

        ax2.plot(range(len(all_counts)), all_counts, color=COLORS['primary'], linewidth=2)
        ax2.fill_between(range(len(all_counts)), all_counts, alpha=0.3, color=COLORS['primary'])
        ax2.set_xlabel('F-Code Rank', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Long-tail Distribution (All Codes)', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Statistics annotation using axes fraction coordinates
        stats_text = (f'Total unique: {len(all_counts)}\n'
                      f'Max: {max(all_counts):,}\n'
                      f'Median: {np.median(all_counts):.0f}\n'
                      f'Min: {min(all_counts)}')
        ax2.text(0.95, 0.95, stats_text, transform=ax2.transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.suptitle('F-Code Frequency Distribution', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig1_fcode_distribution.png',
                    dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig1_fcode_distribution.pdf',
                    bbox_inches='tight')
        print("    Saved: fig1_fcode_distribution.png/pdf")
        plt.close()

    def _plot_text_length_distribution(self):
        """Plot text length distribution (words and tokens)."""
        print("  Creating text length distribution...")

        if self.has_token_data:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            # Word count
            ax1.hist(self.df['word_count'], bins=50, color=COLORS['positive'],
                     alpha=0.7, edgecolor='black')
            ax1.axvline(self.df['word_count'].mean(), color='red', linestyle='--',
                        linewidth=2, label=f'Mean: {self.df["word_count"].mean():,.0f}')
            ax1.axvline(self.df['word_count'].median(), color='blue', linestyle='--',
                        linewidth=2, label=f'Median: {self.df["word_count"].median():,.0f}')
            ax1.set_xlabel('Word Count', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax1.set_title('Clinical Note Length (Words)', fontsize=13, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(axis='y', alpha=0.3)

            # Token count (with chunk size line)
            ax2.hist(self.token_counts, bins=50, color=COLORS['primary'],
                     alpha=0.7, edgecolor='black')
            ax2.axvline(np.mean(self.token_counts), color='red', linestyle='--',
                        linewidth=2, label=f'Mean: {np.mean(self.token_counts):,.0f}')
            ax2.axvline(np.median(self.token_counts), color='blue', linestyle='--',
                        linewidth=2, label=f'Median: {np.median(self.token_counts):,.0f}')
            ax2.axvline(CHUNK_SIZE, color='green', linestyle='-',
                        linewidth=3, label=f'Chunk Size: {CHUNK_SIZE:,}')
            ax2.set_xlabel('Token Count', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax2.set_title('Clinical Note Length (Tokens)', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(axis='y', alpha=0.3)

            # Add annotation about chunking
            exceed_pct = sum(1 for t in self.token_counts if t > CHUNK_SIZE) / len(self.token_counts) * 100
            ax2.text(0.95, 0.95, f'{exceed_pct:.1f}% exceed chunk size\n(will be split)',
                     transform=ax2.transAxes, fontsize=10, verticalalignment='top',
                     horizontalalignment='right',
                     bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        else:
            # Fallback: word count and character count
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            ax1.hist(self.df['word_count'], bins=50, color=COLORS['positive'],
                     alpha=0.7, edgecolor='black')
            ax1.axvline(self.df['word_count'].mean(), color='red', linestyle='--',
                        linewidth=2, label=f'Mean: {self.df["word_count"].mean():,.0f}')
            ax1.set_xlabel('Word Count', fontsize=11, fontweight='bold')
            ax1.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax1.set_title('Clinical Note Length (Words)', fontsize=13, fontweight='bold')
            ax1.legend(fontsize=10)
            ax1.grid(axis='y', alpha=0.3)

            ax2.hist(self.df['text_length'], bins=50, color=COLORS['quaternary'],
                     alpha=0.7, edgecolor='black')
            ax2.axvline(self.df['text_length'].mean(), color='red', linestyle='--',
                        linewidth=2, label=f'Mean: {self.df["text_length"].mean():,.0f}')
            ax2.set_xlabel('Character Count', fontsize=11, fontweight='bold')
            ax2.set_ylabel('Frequency', fontsize=11, fontweight='bold')
            ax2.set_title('Clinical Note Length (Characters)', fontsize=13, fontweight='bold')
            ax2.legend(fontsize=10)
            ax2.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'fig2_text_length_distribution.png',
                    dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'fig2_text_length_distribution.pdf',
                    bbox_inches='tight')
        print("    Saved: fig2_text_length_distribution.png/pdf")
        plt.close()

    def _plot_cooccurrence_heatmap(self, cooccur_matrix, top_codes):
        """Plot co-occurrence heatmap."""
        print("  Creating co-occurrence heatmap...")

        fig, ax = plt.subplots(figsize=(12, 10))

        im = ax.imshow(cooccur_matrix, cmap='YlOrRd', aspect='auto')

        ax.set_xticks(np.arange(len(top_codes)))
        ax.set_yticks(np.arange(len(top_codes)))
        ax.set_xticklabels(top_codes, rotation=45, ha='right')
        ax.set_yticklabels(top_codes)

        for i in range(len(top_codes)):
            for j in range(len(top_codes)):
                if cooccur_matrix[i, j] > 0:
                    text_color = ('white' if cooccur_matrix[i, j] > cooccur_matrix.max() / 2
                                  else 'black')
                    ax.text(j, i, int(cooccur_matrix[i, j]),
                            ha="center", va="center", color=text_color, fontsize=9)

        ax.set_title('F-Code Co-Occurrence Matrix (Top 10 Codes)',
                     fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('F-Code', fontsize=12, fontweight='bold')
        ax.set_ylabel('F-Code', fontsize=12, fontweight='bold')

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Co-occurrence Count', fontsize=11)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'figS1_cooccurrence_heatmap.png',
                    dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figS1_cooccurrence_heatmap.pdf',
                    bbox_inches='tight')
        print("    Saved: figS1_cooccurrence_heatmap.png/pdf (Supplementary)")
        plt.close()

    def _plot_section_structure(self):
        """Plot clinical note section structure with comprehensive table and box plot visualization."""
        print("  Creating section structure figure...")

        if not hasattr(self, 'section_analysis') or self.section_analysis is None:
            print("    Skipping: No section analysis data available")
            return

        results = self.section_analysis['results']
        chunk_size = self.section_analysis['chunk_size']
        sample_size = self.section_analysis['sample_size']

        # Sort by median position (order of appearance) - early sections first in table,
        # but reversed for visualization (Chief Complaint on top, Discharge Diagnosis on bottom)
        sorted_results = sorted(results, key=lambda x: x['median_token_pos'])

        # Use constrained_layout for automatic spacing
        fig = plt.figure(figsize=(18, 16), constrained_layout=True)

        # Create grid with proper spacing - larger table area for full contents
        gs = fig.add_gridspec(2, 1, height_ratios=[1.2, 1.0])

        # ============ TOP: Comprehensive Table ============
        ax_table = fig.add_subplot(gs[0])
        ax_table.axis('off')

        # Create table data with full contents (allow two-line wrapping)
        table_data = []
        for r in sorted_results:
            importance_marker = "★" if r['importance'] == 'High' else ("◆" if r['importance'] == 'Medium' else "")
            chunk_status = "1st Chunk" if r['in_first_chunk_pct'] >= 99 else \
                          ("Mixed" if r['in_first_chunk_pct'] >= 50 else "Later Chunks")

            # Show full contents (will wrap in cell)
            contents = r['contents']

            table_data.append([
                f"{importance_marker} {r['section']}",
                f"{r['median_token_pos']:.0f}",
                contents,
                r['importance'],
                f"{r['in_first_chunk_pct']:.1f}%",
                chunk_status
            ])

        columns = ['Section', 'Position\n(tokens)', 'Contents', 'Importance', f'In 1st Chunk\n(≤{chunk_size})', 'Status']

        # Create table with adjusted column widths for full contents
        table = ax_table.table(
            cellText=table_data,
            colLabels=columns,
            cellLoc='left',
            loc='upper center',
            colWidths=[0.18, 0.07, 0.38, 0.09, 0.10, 0.10]
        )

        # Style the table with larger row height for two-line contents
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.0, 2.0)  # Increased row height for content wrapping

        # Color header
        for j in range(len(columns)):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', fontweight='bold')

        # Color rows by chunk status and importance
        for i, r in enumerate(sorted_results, start=1):
            if r['in_first_chunk_pct'] >= 99:
                bg_color = '#E2EFDA'  # Light green - in first chunk
            elif r['in_first_chunk_pct'] >= 50:
                bg_color = '#FFF2CC'  # Light yellow - mixed
            else:
                bg_color = '#FCE4D6'  # Light orange - mostly in later chunks

            for j in range(len(columns)):
                table[(i, j)].set_facecolor(bg_color)

            if r['importance'] == 'High':
                for j in range(len(columns)):
                    table[(i, j)].set_text_props(fontweight='bold')

        ax_table.set_title(f'Section Analysis (n={sample_size:,}, chunk size: {chunk_size:,} tokens)',
                          fontsize=13, fontweight='bold', pad=10, loc='center')

        # ============ BOTTOM: Box Plot Visualization ============
        ax_viz = fig.add_subplot(gs[1])

        # Prepare box plot data - keep order consistent (early sections at bottom)
        box_data = [r['token_positions'] for r in sorted_results]
        section_labels = []
        box_colors = []

        for r in sorted_results:
            marker = "★" if r['importance'] == 'High' else ("◆" if r['importance'] == 'Medium' else "")
            section_labels.append(f"{marker} {r['section']}")

            # Color based on importance
            if r['importance'] == 'High':
                box_colors.append(COLORS['positive'] if r['in_first_chunk_pct'] >= 99 else 'orange')
            elif r['importance'] == 'Medium':
                box_colors.append('#6BAED6')
            else:
                box_colors.append('#BDBDBD')

        # Add shaded regions first (behind everything)
        max_x = max(max(positions) for positions in box_data if positions) * 1.1
        ax_viz.axvspan(0, chunk_size, alpha=0.15, color='green', zorder=1, label='First chunk region')
        ax_viz.axvspan(chunk_size, max_x, alpha=0.15, color='blue', zorder=1, label='Later chunks region')

        # Add chunk boundary line
        ax_viz.axvline(chunk_size, color='blue', linestyle='-', linewidth=2.5, zorder=10)

        # Create horizontal box plot
        bp = ax_viz.boxplot(box_data, positions=range(len(sorted_results)), vert=False,
                           patch_artist=True, widths=0.6,
                           flierprops=dict(marker='o', markersize=3, alpha=0.5),
                           medianprops=dict(color='black', linewidth=1.5),
                           whiskerprops=dict(linewidth=1.2),
                           capprops=dict(linewidth=1.2))

        # Color boxes based on importance and retention
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.2)

        # Set y-axis labels (sections)
        ax_viz.set_yticks(range(len(sorted_results)))
        ax_viz.set_yticklabels(section_labels, fontsize=10)

        # Invert y-axis so Chief Complaint is on top, Discharge Diagnosis on bottom
        ax_viz.invert_yaxis()

        # Set x-axis limits
        ax_viz.set_xlim(0, max_x)

        ax_viz.set_xlabel('Token Position', fontsize=12, fontweight='bold')
        ax_viz.set_title('Section Position Distribution (Box Plot)', fontsize=12, fontweight='bold')
        ax_viz.grid(axis='x', alpha=0.3, zorder=0)

        # Add chunk boundary annotation (positioned at bottom after y-axis inversion)
        ax_viz.annotate(f'Chunk Boundary\n({chunk_size:,} tokens)',
                       xy=(chunk_size, len(sorted_results) - 1),
                       xytext=(chunk_size + 200, len(sorted_results) - 1.5),
                       fontsize=10, color='blue', fontweight='bold',
                       va='center',
                       arrowprops=dict(arrowstyle='->', color='blue', linewidth=1.5))

        # Legend using built-in positioning (upper right)
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['positive'], edgecolor='black', alpha=0.7, label='High importance (in 1st chunk)'),
            Patch(facecolor='orange', edgecolor='black', alpha=0.7, label='High importance (needs later chunks)'),
            Patch(facecolor='#6BAED6', edgecolor='black', alpha=0.7, label='Medium importance'),
            Patch(facecolor='#BDBDBD', edgecolor='black', alpha=0.7, label='Low importance'),
        ]
        ax_viz.legend(handles=legend_elements, loc='upper right', fontsize=9,
                     framealpha=0.95, edgecolor='gray')

        fig.suptitle('Clinical Note Section Structure Analysis', fontsize=15, fontweight='bold')

        plt.savefig(self.output_dir / 'figS2_section_structure.png',
                    dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figS2_section_structure.pdf',
                    bbox_inches='tight')
        print("    Saved: figS2_section_structure.png/pdf (Supplementary)")
        plt.close()

    def _plot_severity_analysis(self):
        """Plot substance use severity level analysis."""
        print("  Creating severity level analysis...")

        if not hasattr(self, 'substance_analysis'):
            print("    Skipping: No substance analysis data available")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Severity distribution (pie chart)
        ax1 = axes[0, 0]
        severity_data = self.substance_analysis['severity_counts']
        labels = [k for k, v in severity_data.items() if v > 0]
        sizes = [v for v in severity_data.values() if v > 0]
        colors = [COLORS['negative'], COLORS['primary'], COLORS['quaternary'], COLORS['positive']][:len(labels)]

        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, colors=colors,
                                            autopct='%1.1f%%', startangle=90,
                                            textprops={'fontsize': 10})
        ax1.set_title('Severity Level Distribution\n(Substance Use Codes)', fontsize=12, fontweight='bold')

        # Plot 2: Remission status (pie chart)
        ax2 = axes[0, 1]
        remission_data = self.substance_analysis['remission_counts']
        labels2 = [k for k, v in remission_data.items() if v > 0]
        sizes2 = [v for v in remission_data.values() if v > 0]
        colors2 = [COLORS['positive'], COLORS['negative'], COLORS['quaternary']][:len(labels2)]

        wedges2, texts2, autotexts2 = ax2.pie(sizes2, labels=labels2, colors=colors2,
                                               autopct='%1.1f%%', startangle=90,
                                               textprops={'fontsize': 10})
        ax2.set_title('Remission Status Distribution\n(Substance Use Codes)', fontsize=12, fontweight='bold')

        # Plot 3: By substance type (bar chart)
        ax3 = axes[1, 0]
        substance_data = self.substance_analysis['by_substance']
        substances = []
        counts = []
        for prefix in sorted(substance_data.keys()):
            data = substance_data[prefix]
            substances.append(f"{prefix}\n({data['name'][:4]})")
            counts.append(sum(data['codes'].values()))

        bars = ax3.bar(substances, counts, color=COLORS['primary'], alpha=0.8, edgecolor='black')
        ax3.set_xlabel('Substance Type', fontsize=11, fontweight='bold')
        ax3.set_ylabel('Instance Count', fontsize=11, fontweight='bold')
        ax3.set_title('Distribution by Substance Type', fontsize=12, fontweight='bold')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(axis='y', alpha=0.3)

        # Add count labels
        for bar, count in zip(bars, counts):
            if count > 0:
                ax3.text(bar.get_x() + bar.get_width() / 2., bar.get_height(),
                         f'{count:,}', ha='center', va='bottom', fontsize=8)

        # Plot 4: Key insights text
        ax4 = axes[1, 1]
        ax4.axis('off')

        total = self.substance_analysis['total_instances']
        sev = self.substance_analysis['severity_counts']
        rem = self.substance_analysis['remission_counts']

        insights_text = f"""
SUBSTANCE USE DISORDER ANALYSIS
{'=' * 40}

Total Instances: {total:,}

SEVERITY CLASSIFICATION CHALLENGE:
• Abuse (.10): {sev['Abuse']:,} ({sev['Abuse']/total*100:.1f}%)
• Dependence (.20): {sev['Dependence']:,} ({sev['Dependence']/total*100:.1f}%)
• Unspecified (.90): {sev['Unspecified']:,} ({sev['Unspecified']/total*100:.1f}%)

REMISSION STATUS CHALLENGE:
• Active (.x0): {rem['Active']:,} ({rem['Active']/total*100:.1f}%)
• In Remission (.x1): {rem['In Remission']:,} ({rem['In Remission']/total*100:.1f}%)
• Unknown: {rem['Unknown']:,} ({rem['Unknown']/total*100:.1f}%)

KEY CHALLENGES FOR MODEL:
1. Distinguishing Abuse vs Dependence
   requires clinical severity assessment
2. Remission requires explicit documentation
   ("former", "quit", "in recovery")
3. Many codes use Unspecified (.90)
   when severity unclear
"""
        ax4.text(0.05, 0.95, insights_text, transform=ax4.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

        plt.suptitle('Substance Use Disorder Severity Analysis (F10-F19)',
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figS3_severity_analysis.png',
                    dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'figS3_severity_analysis.pdf',
                    bbox_inches='tight')
        print("    Saved: figS3_severity_analysis.png/pdf (Supplementary)")
        plt.close()

    def create_summary_csv(self, fcode_counter, chunking_data=None):
        """Create summary statistics CSV files."""
        print("\n  Creating summary CSV files...")

        # Dataset summary
        summary_data = {
            'Metric': [
                'Total Samples',
                'Train Samples',
                'Validation Samples',
                'Test Samples',
                'Unique Patients',
                'Unique Admissions',
                'Total F-Code Instances',
                'Unique F-Codes',
                'Avg Labels/Sample',
                'Median Labels/Sample',
                'Avg Text Length (chars)',
                'Median Text Length (chars)',
                'Avg Word Count',
                'Median Word Count'
            ],
            'Value': [
                len(self.df),
                len(self.df[self.df['split'] == 'train']),
                len(self.df[self.df['split'] == 'val']),
                len(self.df[self.df['split'] == 'test']),
                self.df['subject_id'].nunique(),
                self.df['hadm_id'].nunique(),
                sum(fcode_counter.values()),
                len(fcode_counter),
                f"{self.df['label_count'].mean():.2f}",
                f"{self.df['label_count'].median():.0f}",
                f"{self.df['text_length'].mean():,.0f}",
                f"{self.df['text_length'].median():,.0f}",
                f"{self.df['word_count'].mean():,.0f}",
                f"{self.df['word_count'].median():,.0f}"
            ]
        }

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.output_dir / 'dataset_summary.csv', index=False)
        print("    Saved: dataset_summary.csv")

        # Top F-codes
        total_instances = sum(fcode_counter.values())
        top_codes_data = []

        for rank, (code, count) in enumerate(fcode_counter.most_common(50), 1):
            top_codes_data.append({
                'Rank': rank,
                'F-Code': code,
                'Count': count,
                '% of Instances': f"{count / total_instances * 100:.2f}",
                '% of Samples': f"{count / len(self.df) * 100:.2f}"
            })

        top_codes_df = pd.DataFrame(top_codes_data)
        top_codes_df.to_csv(self.output_dir / 'top_50_fcodes.csv', index=False)
        print("    Saved: top_50_fcodes.csv")

        # Chunking summary CSV (if available)
        if chunking_data and chunking_data[0] is not None:
            chunked_counter, missing_codes, full_counter = chunking_data

            # Chunking summary
            chunk_summary = {
                'Metric': [
                    'Full CSV Train Samples',
                    'Chunked Train Samples',
                    'Full CSV F-Code Instances',
                    'Chunked F-Code Instances',
                    'Instance Retention Rate (%)',
                    'Full CSV Unique F-Codes',
                    'Chunked Unique F-Codes',
                    'Missing F-Codes',
                    'Singleton Codes (freq=1)',
                    'Very Rare Codes (freq≤5)',
                    'Chunk Size (config)',
                    'Chunk Overlap (config)'
                ],
                'Value': [
                    len(self.df[self.df['split'] == 'train']),
                    len(self.chunked_data['train']),
                    sum(full_counter.values()),
                    sum(chunked_counter.values()),
                    f"{sum(chunked_counter.values()) / sum(full_counter.values()) * 100:.1f}",
                    len(full_counter),
                    len(chunked_counter),
                    len(missing_codes),
                    sum(1 for c in chunked_counter.values() if c == 1),
                    sum(1 for c in chunked_counter.values() if c <= 5),
                    CHUNK_SIZE,
                    CHUNK_OVERLAP
                ]
            }
            chunk_df = pd.DataFrame(chunk_summary)
            chunk_df.to_csv(self.output_dir / 'chunking_summary.csv', index=False)
            print("    Saved: chunking_summary.csv")

            # Missing codes CSV
            if missing_codes:
                missing_data = []
                for code in sorted(missing_codes, key=lambda x: -full_counter.get(x, 0)):
                    missing_data.append({
                        'F-Code': code,
                        'Full CSV Count': full_counter.get(code, 0),
                        'Note': 'NOT in training data'
                    })
                missing_df = pd.DataFrame(missing_data)
                missing_df.to_csv(self.output_dir / 'missing_fcodes.csv', index=False)
                print(f"    Saved: missing_fcodes.csv ({len(missing_codes)} codes)")

    def run(self):
        """Run complete EDA pipeline."""
        print("\n" + "=" * 70)
        print("MIMIC-IV PSYCHIATRIC F-CODE DATASET")
        print("EXPLORATORY DATA ANALYSIS (CHUNKING APPROACH)")
        print("=" * 70 + "\n")

        # Load data
        self.load_data()
        self.load_chunked_data()

        # Original dataset analysis (Sections 1-4)
        self.basic_statistics()
        fcode_counter, labels_per_sample = self.analyze_fcode_distribution()
        self.analyze_text_characteristics()
        cooccur_matrix, top_codes = self.analyze_label_cooccurrence(fcode_counter)

        # New analyses (Sections 5-6)
        self.analyze_information_location()
        self.analyze_severity_levels(fcode_counter)

        # Chunking statistics (Section 7)
        chunking_data = self.analyze_chunking_statistics(fcode_counter)

        # Create visualizations (Section 8)
        self.create_visualizations(fcode_counter, labels_per_sample,
                                   cooccur_matrix, top_codes, chunking_data)

        # Create summary CSVs
        self.create_summary_csv(fcode_counter, chunking_data)

        print("\n" + "=" * 70)
        print("EDA COMPLETE")
        print("=" * 70)
        print(f"\nOutput directory: {self.output_dir}")
        print("\nGenerated files:")
        for file in sorted(self.output_dir.glob('*')):
            print(f"  - {file.name}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Exploratory Data Analysis for MIMIC-IV Psychiatric F-Code Dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DATA_DIR,
        help="Directory containing MIMIC-IV CSV files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=OUTPUT_DIR_3_EDA,
        help="Output directory for EDA results"
    )

    args = parser.parse_args()

    eda = MIMICDatasetEDA(data_dir=args.data_dir, output_dir=args.output_dir)
    eda.run()

    return 0


if __name__ == "__main__":
    exit(main())
