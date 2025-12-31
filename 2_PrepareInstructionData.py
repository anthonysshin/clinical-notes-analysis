#!/usr/bin/env python3
"""
2_PrepareInstructionData.py - Convert CSV to Instruction Format with Chunking

This script converts the MIMIC-IV CSV datasets to instruction format (JSON)
for LLM fine-tuning using CHUNKING approach.

Chunking Strategy:
- Split documents into overlapping chunks (2800 tokens, 200 overlap)
- Each chunk becomes a separate training sample
- All chunks from the same document share the same F-code labels
- Preserves information that would be lost with truncation

Token Allocation per Chunk (max_seq_length = 4096):
  System Prompt (CoT):    ~550 tokens
  User Template:          ~30 tokens
  Chunk Content:          2800 tokens
  Output (JSON array):    100 tokens
  Buffer:                 ~616 tokens
  TOTAL:                  4096 tokens

Output Files:
    - train_chunked.json (training samples from chunked documents)
    - val_chunked.json (validation samples from chunked documents)
    - test_chunked.json (test samples from chunked documents)
    - dataset_info.json

Usage:
    python 2_PrepareInstructionData.py --data-dir ./data
"""

import pandas as pd
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
from transformers import AutoTokenizer

# Import centralized config for reproducibility
from config import (
    RANDOM_SEED, set_all_seeds, CHUNK_SIZE, CHUNK_OVERLAP, BASE_MODEL_NAME
)

# Set random seeds for reproducibility
set_all_seeds(RANDOM_SEED)


def format_f_code(code: str) -> str:
    """Format F-code with period at 3rd position if not present."""
    code = str(code).strip()
    if len(code) > 3 and '.' not in code:
        return f"{code[:3]}.{code[3:]}"
    return code


def chunk_text(
    text: str,
    tokenizer,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Tuple[str, int, int]]:
    """
    Split text into overlapping chunks.

    Args:
        text: Full clinical note text
        tokenizer: HuggingFace tokenizer for token counting
        chunk_size: Maximum tokens per chunk
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of tuples: (chunk_text, start_token, end_token)
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(tokens)

    # If text fits in one chunk, return as-is
    if total_tokens <= chunk_size:
        return [(text, 0, total_tokens)]

    chunks = []
    start = 0

    while start < total_tokens:
        end = min(start + chunk_size, total_tokens)
        chunk_tokens = tokens[start:end]
        decoded_chunk = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append((decoded_chunk, start, end))

        if end >= total_tokens:
            break

        start = end - chunk_overlap

    return chunks


def csv_to_instruction_chunked(
    row: pd.Series,
    tokenizer,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP
) -> List[Dict]:
    """
    Convert a CSV row to instruction format with chunking.

    Each chunk becomes a separate training sample with the same labels.

    Returns:
        List of instruction dictionaries (one per chunk)
    """
    # Parse and format F-codes
    f_codes = [format_f_code(c.strip()) for c in str(row['f_codes_str']).split(',')]

    # Output as simple JSON array (no explanations)
    f_codes_json = json.dumps(f_codes)

    # Get chunks
    chunks = chunk_text(row['text'], tokenizer, chunk_size, chunk_overlap)

    # Create a sample for each chunk
    samples = []
    for chunk_idx, (chunk_text_content, start_token, end_token) in enumerate(chunks):
        samples.append({
            'instruction': "Analyze the clinical note and extract psychiatric F-codes (F00-F99).",
            'input': chunk_text_content,
            'output': f_codes_json,  # Same labels for all chunks
            'hadm_id': int(row['hadm_id']) if pd.notna(row['hadm_id']) else None,
            'subject_id': int(row['subject_id']) if pd.notna(row['subject_id']) else None,
            'chunk_info': {
                'chunk_idx': chunk_idx,
                'total_chunks': len(chunks),
                'start_token': start_token,
                'end_token': end_token
            }
        })

    return samples


def validate_instruction_data(data: List[Dict], name: str) -> List[str]:
    """
    Validate instruction data structure.

    Returns list of errors (empty if valid).
    """
    errors = []

    required_fields = ['instruction', 'input', 'output', 'hadm_id', 'subject_id']

    for i, item in enumerate(data):
        for field in required_fields:
            if field not in item:
                errors.append(f"{name}[{i}]: Missing field '{field}'")
            elif item[field] is None and field in ['input', 'output']:
                errors.append(f"{name}[{i}]: Null value in '{field}'")

        # Check output has valid F-codes (now in JSON array format)
        if 'output' in item and item['output']:
            try:
                codes = json.loads(item['output'])
                # Verify it's a list, not a string or other type
                if not isinstance(codes, list):
                    errors.append(f"{name}[{i}]: Output is not a JSON array: {type(codes).__name__}")
                    continue
                for code in codes:
                    if not isinstance(code, str) or not code.startswith('F'):
                        errors.append(f"{name}[{i}]: Invalid F-code format: '{code}'")
            except json.JSONDecodeError:
                errors.append(f"{name}[{i}]: Invalid JSON in output")

    return errors


def main():
    parser = argparse.ArgumentParser(description='Prepare Chunked Instruction Data')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory containing CSV files')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE,
                        help=f'Chunk size in tokens (default: {CHUNK_SIZE})')
    parser.add_argument('--chunk-overlap', type=int, default=CHUNK_OVERLAP,
                        help=f'Overlap between chunks (default: {CHUNK_OVERLAP})')
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    print("\n" + "=" * 70)
    print("PREPARE INSTRUCTION DATA WITH CHUNKING")
    print("=" * 70)
    print(f"Strategy: Split documents into overlapping chunks")
    print(f"Chunk size: {args.chunk_size} tokens")
    print(f"Chunk overlap: {args.chunk_overlap} tokens")
    print(f"Output format: JSON array (F-codes only, no explanations)")
    print("=" * 70)

    # Load tokenizer for token counting (use same tokenizer as training model)
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
    print(f"      Tokenizer loaded ({BASE_MODEL_NAME})")

    # Load CSV files
    print("\n[2/6] Loading CSV files...")
    train_csv = pd.read_csv(data_dir / 'mimic_iv_train_data.csv')
    val_csv = pd.read_csv(data_dir / 'mimic_iv_val_data.csv')
    test_csv = pd.read_csv(data_dir / 'mimic_iv_test_data.csv')

    print(f"      Train CSV: {len(train_csv):,} documents")
    print(f"      Val CSV: {len(val_csv):,} documents")
    print(f"      Test CSV: {len(test_csv):,} documents")

    # Convert to instruction format with chunking
    print("\n[3/6] Converting to instruction format with chunking...")

    # Process training data
    print("      Processing training data...")
    train_data = []
    train_doc_count = 0
    for idx, (_, row) in enumerate(train_csv.iterrows()):
        if (idx + 1) % 5000 == 0:
            print(f"        Processed {idx + 1}/{len(train_csv)} documents...")
        samples = csv_to_instruction_chunked(row, tokenizer, args.chunk_size, args.chunk_overlap)
        train_data.extend(samples)
        train_doc_count += 1

    # Process validation data
    print("      Processing validation data...")
    val_data = []
    val_doc_count = 0
    for _, row in val_csv.iterrows():
        samples = csv_to_instruction_chunked(row, tokenizer, args.chunk_size, args.chunk_overlap)
        val_data.extend(samples)
        val_doc_count += 1

    # Process test data
    print("      Processing test data...")
    test_data = []
    test_doc_count = 0
    for _, row in test_csv.iterrows():
        samples = csv_to_instruction_chunked(row, tokenizer, args.chunk_size, args.chunk_overlap)
        test_data.extend(samples)
        test_doc_count += 1

    print(f"\n      Train: {train_doc_count:,} documents -> {len(train_data):,} chunks")
    print(f"      Val: {val_doc_count:,} documents -> {len(val_data):,} chunks")
    print(f"      Test: {test_doc_count:,} documents -> {len(test_data):,} chunks")

    # Calculate chunking statistics
    def calc_stats(data: List[Dict], doc_count: int) -> Dict:
        total_chunks = len(data)
        multi_chunk_docs = sum(1 for d in data if d['chunk_info']['total_chunks'] > 1 and d['chunk_info']['chunk_idx'] == 0)
        single_chunk_docs = sum(1 for d in data if d['chunk_info']['total_chunks'] == 1)
        chunks_per_doc = total_chunks / doc_count if doc_count > 0 else 0

        # Chunk distribution
        chunk_counts = {}
        for d in data:
            if d['chunk_info']['chunk_idx'] == 0:
                num_chunks = d['chunk_info']['total_chunks']
                chunk_counts[num_chunks] = chunk_counts.get(num_chunks, 0) + 1

        return {
            'documents': doc_count,
            'total_chunks': total_chunks,
            'chunks_per_doc': chunks_per_doc,
            'single_chunk_docs': single_chunk_docs,
            'single_chunk_pct': single_chunk_docs / doc_count * 100 if doc_count > 0 else 0,
            'multi_chunk_docs': multi_chunk_docs,
            'multi_chunk_pct': multi_chunk_docs / doc_count * 100 if doc_count > 0 else 0,
            'chunk_distribution': chunk_counts
        }

    train_stats = calc_stats(train_data, train_doc_count)
    val_stats = calc_stats(val_data, val_doc_count)
    test_stats = calc_stats(test_data, test_doc_count)

    print("\n      Chunking Statistics:")
    print(f"      Train: {train_stats['chunks_per_doc']:.2f} chunks/doc, "
          f"{train_stats['single_chunk_pct']:.1f}% single-chunk, "
          f"{train_stats['multi_chunk_pct']:.1f}% multi-chunk")

    # Validate data structure
    print("\n[4/6] Validating data structure...")
    all_errors = []
    all_errors.extend(validate_instruction_data(train_data, 'train'))
    all_errors.extend(validate_instruction_data(val_data, 'val'))
    all_errors.extend(validate_instruction_data(test_data, 'test'))

    if all_errors:
        print(f"      ERRORS ({len(all_errors)}):")
        for error in all_errors[:10]:
            print(f"        - {error}")
    else:
        print("      All validations passed")

    # Save JSON files (without chunk_info in saved data to reduce file size)
    print("\n[5/6] Saving JSON files...")

    def clean_for_save(data: List[Dict]) -> List[Dict]:
        """Remove chunk_info from saved data to reduce file size."""
        return [{k: v for k, v in d.items() if k != 'chunk_info'} for d in data]

    with open(data_dir / 'train_chunked.json', 'w') as f:
        json.dump(clean_for_save(train_data), f, indent=2)
    print(f"      train_chunked.json: {len(train_data):,} samples")

    with open(data_dir / 'val_chunked.json', 'w') as f:
        json.dump(clean_for_save(val_data), f, indent=2)
    print(f"      val_chunked.json: {len(val_data):,} samples")

    with open(data_dir / 'test_chunked.json', 'w') as f:
        json.dump(clean_for_save(test_data), f, indent=2)
    print(f"      test_chunked.json: {len(test_data):,} samples")

    # Save dataset info
    print("\n[6/6] Saving dataset info...")
    dataset_info = {
        'created_at': datetime.now().isoformat(),
        'pipeline_version': 'CNA_Chunking_1222_2025',
        'approach': 'Chunking (overlapping chunks with union aggregation)',
        'description': f'Documents split into {args.chunk_size} token chunks with {args.chunk_overlap} overlap',
        'source_files': {
            'train_csv': str(data_dir / 'mimic_iv_train_data.csv'),
            'val_csv': str(data_dir / 'mimic_iv_val_data.csv'),
            'test_csv': str(data_dir / 'mimic_iv_test_data.csv')
        },
        'chunking_config': {
            'chunk_size': args.chunk_size,
            'chunk_overlap': args.chunk_overlap,
            'aggregation_strategy': 'union',
            'output_format': 'JSON array (F-codes only)'
        },
        'counts': {
            'train_documents': train_doc_count,
            'train_chunks': len(train_data),
            'val_documents': val_doc_count,
            'val_chunks': len(val_data),
            'test_documents': test_doc_count,
            'test_chunks': len(test_data),
            'total_documents': train_doc_count + val_doc_count + test_doc_count,
            'total_chunks': len(train_data) + len(val_data) + len(test_data)
        },
        'chunking_statistics': {
            'train': train_stats,
            'val': val_stats,
            'test': test_stats
        },
        'validation': {
            'passed': len(all_errors) == 0,
            'error_count': len(all_errors),
            'errors': all_errors[:20] if all_errors else []
        }
    }

    with open(data_dir / 'dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    print(f"      dataset_info.json")

    # Summary
    print("\n" + "=" * 70)
    print("CHUNKED INSTRUCTION DATA PREPARATION COMPLETE")
    print("=" * 70)
    print(f"Approach: {args.chunk_size} token chunks with {args.chunk_overlap} overlap")
    print(f"Output format: JSON array (F-codes only)")
    print(f"\nDocument counts:")
    print(f"  Train: {train_doc_count:,} documents -> {len(train_data):,} chunks")
    print(f"  Val: {val_doc_count:,} documents -> {len(val_data):,} chunks")
    print(f"  Test: {test_doc_count:,} documents -> {len(test_data):,} chunks")
    print(f"\nChunking Statistics (Train):")
    print(f"  Chunks per document: {train_stats['chunks_per_doc']:.2f}")
    print(f"  Single-chunk documents: {train_stats['single_chunk_pct']:.1f}%")
    print(f"  Multi-chunk documents: {train_stats['multi_chunk_pct']:.1f}%")
    print(f"\nValidation: {'PASSED' if len(all_errors) == 0 else 'FAILED'}")
    print("=" * 70 + "\n")

    return 0 if len(all_errors) == 0 else 1


if __name__ == "__main__":
    exit(main())
