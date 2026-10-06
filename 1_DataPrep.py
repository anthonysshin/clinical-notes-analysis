#!/usr/bin/env python3
"""
1_DataPrep.py - MIMIC-IV Dataset Preparation for Psychiatric F-Code Prediction

This script creates train/validation/test splits from MIMIC-IV discharge notes
with ICD-10 F-codes (psychiatric diagnoses). No primary_seq filter is applied --
all admissions with any F-code diagnosis are included, not just those where a
psychiatric code is the primary diagnosis.

Pipeline:
    Step 1: Load and merge discharge notes with F-code diagnoses
    Step 2: Remove temporal conflicts (F32+F33, F30+F31)
    Step 3: Create stratified train/validation/test splits (70-15-15)
    Step 4: Validate data structure

Data Sources:
    - MIMIC-IV-Note 2.2: Discharge summaries
    - MIMIC-IV 3.1: ICD diagnoses (filtered to F-codes)

Output Files:
    - mimic_iv_train_data.csv
    - mimic_iv_val_data.csv
    - mimic_iv_test_data.csv
    - mimic_iv_dataset_stats.json

Usage:
    python 1_DataPrep.py --mimic-path /path/to/mimic --output-dir ./data
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import json
import time
import argparse
import warnings
from pathlib import Path
from datetime import datetime

# Import centralized config for reproducibility
from config import RANDOM_SEED, set_all_seeds, MIMIC_RAW_DATA_DIR, DATA_DIR

warnings.filterwarnings('ignore')

# Set random seeds at module load for reproducibility
set_all_seeds(RANDOM_SEED)


def format_f_code(code: str) -> str:
    """Format F-code with period at 3rd position if not present."""
    code = str(code).strip()
    if len(code) > 3 and '.' not in code:
        return f"{code[:3]}.{code[3:]}"
    return code


class MIMICDatasetCreator:
    """Create MIMIC-IV dataset for psychiatric F-code prediction."""

    def __init__(self, mimic_base_path: str = "."):
        self.mimic_base_path = Path(mimic_base_path)
        self.stats = {}

    def step1_create_merged_dataset(self) -> pd.DataFrame:
        """
        Step 1: Merge discharge summaries with F-code diagnoses.

        NO primary_seq filter - uses ALL admissions with ANY F-code.
        """
        print("=" * 70)
        print("STEP 1: CREATE MERGED DATASET")
        print("=" * 70)

        start_time = time.time()

        # Define file paths
        discharge_path = self.mimic_base_path / "mimic-iv-note/2.2/note/discharge.csv"
        diagnoses_path = self.mimic_base_path / "mimiciv/3.1/hosp/diagnoses_icd.csv"

        # Load discharge summaries
        print("\n[1/5] Loading discharge summaries...")
        discharge_df = pd.read_csv(
            discharge_path,
            dtype={'subject_id': str, 'hadm_id': str}
        )
        print(f"      Loaded: {len(discharge_df):,} discharge notes")

        # Load diagnoses
        print("\n[2/5] Loading diagnoses...")
        diagnoses_df = pd.read_csv(
            diagnoses_path,
            dtype={'subject_id': str, 'hadm_id': str}
        )
        print(f"      Loaded: {len(diagnoses_df):,} diagnoses")

        # Filter F-codes only (ICD-10 psychiatric diagnoses: F00-F99)
        print("\n[3/5] Filtering F-codes (psychiatric diagnoses)...")
        f_diagnoses = diagnoses_df[
            diagnoses_df['icd_code'].str.startswith('F', na=False)
        ].copy()
        print(f"      Found: {len(f_diagnoses):,} F-code diagnoses")

        # Sort by hadm_id and seq_num to maintain clinical priority
        f_diagnoses_sorted = f_diagnoses.sort_values(['hadm_id', 'seq_num'])

        # Group F-codes by admission (preserve seq_num order)
        print("\n[4/5] Grouping F-codes by admission...")
        grouped_f_codes = f_diagnoses_sorted.groupby('hadm_id').agg({
            'subject_id': 'first',
            'icd_code': lambda x: ', '.join([format_f_code(c) for c in x.tolist()]),
            'seq_num': ['count', 'min']
        }).reset_index()

        grouped_f_codes.columns = ['hadm_id', 'subject_id', 'f_codes_str', 'f_code_count', 'primary_seq']
        print(f"      Admissions with F-codes: {len(grouped_f_codes):,}")

        # Merge discharge summaries with F-codes (INNER JOIN)
        print("\n[5/5] Merging discharge summaries with F-codes...")
        merged_dataset = pd.merge(
            discharge_df[['hadm_id', 'subject_id', 'text']],
            grouped_f_codes,
            on=['hadm_id', 'subject_id'],
            how='inner'
        )

        print(f"      Final merged dataset: {len(merged_dataset):,} records")

        # Add basic statistics
        merged_dataset['text_length'] = merged_dataset['text'].str.len()
        merged_dataset['primary_f_code'] = merged_dataset['f_codes_str'].str.split(',').str[0].str.strip()
        merged_dataset['primary_category'] = merged_dataset['primary_f_code'].str[:3]

        # Collect unique F-codes
        all_f_codes = []
        for codes in merged_dataset['f_codes_str']:
            all_f_codes.extend([c.strip() for c in str(codes).split(',')])

        self.stats['step1'] = {
            'original_discharge_notes': len(discharge_df),
            'original_all_diagnoses': len(diagnoses_df),
            'f_code_diagnoses': len(f_diagnoses),
            'admissions_with_f_codes': len(grouped_f_codes),
            'final_merged_records': len(merged_dataset),
            'unique_patients': merged_dataset['subject_id'].nunique(),
            'unique_f_codes': len(set(all_f_codes)),
            'filter_criteria': 'NO primary_seq filter - ALL F-codes included',
            'merge_rate_percent': round(len(merged_dataset) / len(discharge_df) * 100, 2),
            'processing_time_seconds': round(time.time() - start_time, 2)
        }

        self._print_step1_stats(merged_dataset)
        return merged_dataset

    def _print_step1_stats(self, dataset):
        """Print Step 1 statistics."""
        stats = self.stats['step1']
        print(f"\n{'='*50}")
        print("STEP 1 SUMMARY")
        print(f"{'='*50}")
        print(f"Original discharge notes: {stats['original_discharge_notes']:,}")
        print(f"F-code diagnoses: {stats['f_code_diagnoses']:,}")
        print(f"Final merged records: {stats['final_merged_records']:,}")
        print(f"Unique patients: {stats['unique_patients']:,}")
        print(f"Unique F-codes: {stats['unique_f_codes']}")
        print(f"Filter: {stats['filter_criteria']}")
        print(f"Processing time: {stats['processing_time_seconds']:.1f}s")

    def step2_remove_conflicts(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """
        Step 2: Remove temporal conflicts from dataset.

        Conflicts:
        - F32 (single episode depression) + F33 (recurrent depression)
        - F30 (single manic episode) + F31 (bipolar disorder)
        """
        print("\n" + "=" * 70)
        print("STEP 2: REMOVE TEMPORAL CONFLICTS")
        print("=" * 70)

        start_time = time.time()
        conflicts_found = []

        print("\nChecking for temporal conflicts...")
        for idx, row in dataset.iterrows():
            f_codes = [code.strip() for code in str(row['f_codes_str']).split(',')]

            f32_codes = [c for c in f_codes if c.startswith('F32')]
            f33_codes = [c for c in f_codes if c.startswith('F33')]
            f30_codes = [c for c in f_codes if c.startswith('F30')]
            f31_codes = [c for c in f_codes if c.startswith('F31')]

            if f32_codes and f33_codes:
                conflicts_found.append({
                    'index': idx,
                    'hadm_id': row['hadm_id'],
                    'conflict_type': 'depression_temporal',
                    'detail': f"F32 + F33: {f32_codes} + {f33_codes}"
                })
            elif f30_codes and f31_codes:
                conflicts_found.append({
                    'index': idx,
                    'hadm_id': row['hadm_id'],
                    'conflict_type': 'bipolar_temporal',
                    'detail': f"F30 + F31: {f30_codes} + {f31_codes}"
                })

        print(f"Conflicts found: {len(conflicts_found)}")

        if conflicts_found:
            indices_to_remove = [c['index'] for c in conflicts_found]
            clean_dataset = dataset.drop(indices_to_remove).reset_index(drop=True)
            print(f"Removed {len(conflicts_found)} conflicting records")
        else:
            clean_dataset = dataset.copy()
            print("No conflicts found - dataset is clean")

        self.stats['step2'] = {
            'conflicts_found': len(conflicts_found),
            'original_size': len(dataset),
            'clean_size': len(clean_dataset),
            'retention_rate_percent': round(len(clean_dataset) / len(dataset) * 100, 2),
            'processing_time_seconds': round(time.time() - start_time, 2)
        }

        print(f"\n{'='*50}")
        print("STEP 2 SUMMARY")
        print(f"{'='*50}")
        print(f"Original size: {len(dataset):,}")
        print(f"Conflicts removed: {len(conflicts_found)}")
        print(f"Clean size: {len(clean_dataset):,}")
        print(f"Retention rate: {self.stats['step2']['retention_rate_percent']:.2f}%")

        return clean_dataset

    def step3_create_splits(
        self,
        dataset: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = RANDOM_SEED
    ) -> tuple:
        """
        Step 3: Create train/validation/test splits (70-15-15).
        """
        print("\n" + "=" * 70)
        print("STEP 3: CREATE TRAIN/VAL/TEST SPLITS (70-15-15)")
        print("=" * 70)

        start_time = time.time()

        # Check for rare categories
        category_counts = dataset['primary_category'].value_counts()
        rare_categories = category_counts[category_counts < 2].index.tolist()

        if rare_categories:
            print(f"\nFound {len(rare_categories)} rare categories (<2 samples)")
            print("Using random split instead of stratified")
            use_stratify = False
        else:
            print("\nUsing stratified split by primary F-code category")
            use_stratify = True

        # First split: train vs (val + test)
        test_val_ratio = val_ratio + test_ratio
        if use_stratify:
            train_df, temp_df = train_test_split(
                dataset,
                test_size=test_val_ratio,
                random_state=random_state,
                stratify=dataset['primary_category']
            )
        else:
            train_df, temp_df = train_test_split(
                dataset,
                test_size=test_val_ratio,
                random_state=random_state
            )

        # Second split: val vs test (50-50 of remaining)
        temp_category_counts = temp_df['primary_category'].value_counts()
        temp_rare = temp_category_counts[temp_category_counts < 2].index.tolist()

        if temp_rare or not use_stratify:
            val_df, test_df = train_test_split(
                temp_df,
                test_size=0.5,
                random_state=random_state
            )
        else:
            val_df, test_df = train_test_split(
                temp_df,
                test_size=0.5,
                random_state=random_state,
                stratify=temp_df['primary_category']
            )

        self.stats['step3'] = {
            'total_samples': len(dataset),
            'train_samples': len(train_df),
            'val_samples': len(val_df),
            'test_samples': len(test_df),
            'train_percentage': round(len(train_df) / len(dataset) * 100, 2),
            'val_percentage': round(len(val_df) / len(dataset) * 100, 2),
            'test_percentage': round(len(test_df) / len(dataset) * 100, 2),
            'stratified': use_stratify,
            'random_state': random_state,
            'processing_time_seconds': round(time.time() - start_time, 2)
        }

        print(f"\n{'='*50}")
        print("STEP 3 SUMMARY")
        print(f"{'='*50}")
        print(f"Total samples: {len(dataset):,}")
        print(f"Train: {len(train_df):,} ({self.stats['step3']['train_percentage']:.1f}%)")
        print(f"Val: {len(val_df):,} ({self.stats['step3']['val_percentage']:.1f}%)")
        print(f"Test: {len(test_df):,} ({self.stats['step3']['test_percentage']:.1f}%)")

        return train_df, val_df, test_df

    def step4_validate_data(self, train_df, val_df, test_df, output_dir: Path) -> bool:
        """
        Step 4: Validate data structure.

        Checks:
        - No null values in key columns
        - F-code format (F##.## with periods)
        - hadm_id uniqueness within each split
        - No overlap between train/val/test
        - F-code distribution consistency
        """
        print("\n" + "=" * 70)
        print("STEP 4: VALIDATE DATA STRUCTURE")
        print("=" * 70)

        validation_errors = []
        validation_warnings = []

        # Check 1: Null values in key columns
        print("\n[1/5] Checking for null values...")
        key_columns = ['hadm_id', 'subject_id', 'text', 'f_codes_str', 'f_code_count']
        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            for col in key_columns:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    validation_errors.append(f"{name}: {null_count} null values in '{col}'")
        print("      Done")

        # Check 2: F-code format (should have periods, not be empty)
        print("\n[2/5] Checking F-code format...")
        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            invalid_format = 0
            empty_codes = 0
            for codes_str in df['f_codes_str']:
                codes = [c.strip() for c in str(codes_str).split(',')]
                # Filter out empty strings
                codes = [c for c in codes if c]
                if not codes:
                    empty_codes += 1
                    continue
                for code in codes:
                    if len(code) > 3 and '.' not in code:
                        invalid_format += 1
                        break
            if invalid_format > 0:
                validation_errors.append(f"{name}: {invalid_format} records with F-codes missing periods")
            if empty_codes > 0:
                validation_errors.append(f"{name}: {empty_codes} records with empty F-codes")
        print("      Done")

        # Check 3: hadm_id uniqueness
        print("\n[3/5] Checking hadm_id uniqueness...")
        for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
            if df['hadm_id'].nunique() != len(df):
                duplicates = len(df) - df['hadm_id'].nunique()
                validation_errors.append(f"{name}: {duplicates} duplicate hadm_ids")
        print("      Done")

        # Check 4: No overlap between splits
        print("\n[4/5] Checking for overlap between splits...")
        train_ids = set(train_df['hadm_id'])
        val_ids = set(val_df['hadm_id'])
        test_ids = set(test_df['hadm_id'])

        train_val_overlap = train_ids & val_ids
        train_test_overlap = train_ids & test_ids
        val_test_overlap = val_ids & test_ids

        if train_val_overlap:
            validation_errors.append(f"Train-Val overlap: {len(train_val_overlap)} hadm_ids")
        if train_test_overlap:
            validation_errors.append(f"Train-Test overlap: {len(train_test_overlap)} hadm_ids")
        if val_test_overlap:
            validation_errors.append(f"Val-Test overlap: {len(val_test_overlap)} hadm_ids")
        print("      Done")

        # Check 5: F-code distribution consistency
        print("\n[5/5] Checking F-code distribution...")
        def get_top_categories(df, n=10):
            return set(df['primary_category'].value_counts().head(n).index)

        train_top = get_top_categories(train_df)
        val_top = get_top_categories(val_df)
        test_top = get_top_categories(test_df)

        if train_top != val_top or train_top != test_top:
            validation_warnings.append("Top 10 F-code categories differ between splits (may be expected)")
        print("      Done")

        # Report results
        print(f"\n{'='*50}")
        print("VALIDATION RESULTS")
        print(f"{'='*50}")

        if validation_errors:
            print(f"\nERRORS ({len(validation_errors)}):")
            for error in validation_errors:
                print(f"  - {error}")
        else:
            print("\nNo errors found")

        if validation_warnings:
            print(f"\nWARNINGS ({len(validation_warnings)}):")
            for warning in validation_warnings:
                print(f"  - {warning}")

        self.stats['step4'] = {
            'validation_passed': len(validation_errors) == 0,
            'errors': validation_errors,
            'warnings': validation_warnings
        }

        return len(validation_errors) == 0

    def save_datasets(self, train_df, val_df, test_df, output_dir: Path):
        """Save datasets and statistics."""
        print("\n" + "=" * 70)
        print("SAVING DATASETS")
        print("=" * 70)

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save CSVs
        train_df.to_csv(output_dir / 'mimic_iv_train_data.csv', index=False)
        val_df.to_csv(output_dir / 'mimic_iv_val_data.csv', index=False)
        test_df.to_csv(output_dir / 'mimic_iv_test_data.csv', index=False)

        print(f"\nSaved:")
        print(f"  - mimic_iv_train_data.csv: {len(train_df):,} samples")
        print(f"  - mimic_iv_val_data.csv: {len(val_df):,} samples")
        print(f"  - mimic_iv_test_data.csv: {len(test_df):,} samples")

        # Save statistics
        self.stats['metadata'] = {
            'created_at': datetime.now().isoformat(),
            'output_directory': str(output_dir),
            'cohort_filter': 'No primary_seq filter - all F-codes included'
        }

        with open(output_dir / 'mimic_iv_dataset_stats.json', 'w') as f:
            json.dump(self.stats, f, indent=2)
        print(f"  - mimic_iv_dataset_stats.json")


def main():
    parser = argparse.ArgumentParser(description='MIMIC-IV Dataset Preparation')
    parser.add_argument('--mimic-path', type=str,
                        default=MIMIC_RAW_DATA_DIR,
                        help=f'Path to MIMIC-IV data (default: {MIMIC_RAW_DATA_DIR})')
    parser.add_argument('--output-dir', type=str,
                        default=DATA_DIR,
                        help=f'Output directory for datasets (default: {DATA_DIR})')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("MIMIC-IV DATASET PREPARATION")
    print("No primary_seq filter - all F-codes included")
    print("=" * 70)

    creator = MIMICDatasetCreator(args.mimic_path)

    # Step 1: Create merged dataset
    merged_dataset = creator.step1_create_merged_dataset()

    # Step 2: Remove temporal conflicts
    clean_dataset = creator.step2_remove_conflicts(merged_dataset)

    # Step 3: Create 70-15-15 splits
    train_df, val_df, test_df = creator.step3_create_splits(clean_dataset)

    # Step 4: Validate data structure
    output_dir = Path(args.output_dir)
    validation_passed = creator.step4_validate_data(train_df, val_df, test_df, output_dir)

    if not validation_passed:
        print("\n" + "!" * 70)
        print("WARNING: Data validation failed! Check errors above.")
        print("!" * 70)

    # Save datasets
    creator.save_datasets(train_df, val_df, test_df, output_dir)

    print("\n" + "=" * 70)
    print("DATASET PREPARATION COMPLETE")
    print("=" * 70)
    print(f"Total samples: {len(clean_dataset):,}")
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
    print(f"Validation: {'PASSED' if validation_passed else 'FAILED'}")
    print("=" * 70 + "\n")

    return 0 if validation_passed else 1


if __name__ == "__main__":
    exit(main())
