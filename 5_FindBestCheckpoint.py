#!/usr/bin/env python3
"""
5_FindBestCheckpoint.py - Find Best Checkpoint for Chunking Approach

This script serves TWO purposes:
1. When run as main: Evaluates all checkpoints on validation set and saves best to best_checkpoint.json
2. When imported: Provides get_best_checkpoint_path() function for evaluation scripts

Text Handling: Chunking (2800 tokens per chunk, 200 token overlap) with Union aggregation

Usage as script:
    python 5_FindBestCheckpoint.py

Usage as module:
    from 5_FindBestCheckpoint import get_best_checkpoint_path
    checkpoint_path = get_best_checkpoint_path()
"""

import os
os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"

import json
import re
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# Import centralized config
from config import (
    RANDOM_SEED, set_all_seeds,
    MAX_SEQ_LENGTH, CHUNK_SIZE, CHUNK_OVERLAP, MAX_NEW_TOKENS,
    LOAD_IN_4BIT, TEMPERATURE,
    VAL_DATA_PATH, CHECKPOINT_DIR, FINAL_MODEL_DIR,
    DEFAULT_MAX_SAMPLES, COLORS, apply_figure_style,
    OUTPUT_DIR_5_CHECKPOINT, MIN_CHECKPOINT_STEP
)

# Set seeds for reproducibility
set_all_seeds(RANDOM_SEED)

# Apply figure style
apply_figure_style()

# =============================================================================
# CONFIGURATION SECTION (for importing)
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.resolve()

# Best checkpoint config file (used by evaluation scripts)
BEST_CHECKPOINT_FILE = PROJECT_ROOT / "best_checkpoint.json"

# Default paths
DEFAULT_CHECKPOINT_PATH = PROJECT_ROOT / FINAL_MODEL_DIR
DEFAULT_CHECKPOINT_DIR = PROJECT_ROOT / CHECKPOINT_DIR
DEFAULT_VAL_DATA_PATH = PROJECT_ROOT / VAL_DATA_PATH


def get_best_checkpoint_path(require_exists: bool = True) -> str:
    """
    Get the path to the best checkpoint.

    Priority:
    1. From best_checkpoint.json (if exists and valid)
    2. From environment variable CHECKPOINT_PATH (if set)
    3. Default to ./models/final_model

    Args:
        require_exists: If True, raise FileNotFoundError if no valid checkpoint found.
                       If False, return default path even if it doesn't exist.

    Returns:
        Absolute path to the checkpoint directory

    Raises:
        FileNotFoundError: If require_exists=True and no valid checkpoint is found.
    """
    # Try loading from config file first
    if BEST_CHECKPOINT_FILE.exists():
        try:
            with open(BEST_CHECKPOINT_FILE, 'r') as f:
                config = json.load(f)
            checkpoint_path = config.get('checkpoint')
            if checkpoint_path and os.path.exists(checkpoint_path):
                print(f"[checkpoint] Using best checkpoint: {Path(checkpoint_path).name} (Micro F1: {config.get('micro_f1', 'N/A'):.4f})")
                return checkpoint_path
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {BEST_CHECKPOINT_FILE}: {e}")

    # Try environment variable
    env_checkpoint = os.environ.get('CHECKPOINT_PATH')
    if env_checkpoint and os.path.exists(env_checkpoint):
        print(f"[checkpoint] Using checkpoint from environment: {env_checkpoint}")
        return env_checkpoint

    # Fall back to default
    default_path = str(DEFAULT_CHECKPOINT_PATH)
    if os.path.exists(default_path):
        print(f"[checkpoint] Using default checkpoint: {default_path}")
        return default_path

    # No valid checkpoint found
    if require_exists:
        raise FileNotFoundError(
            f"No valid checkpoint found.\n"
            f"Checked:\n"
            f"  1. {BEST_CHECKPOINT_FILE} (not found or invalid)\n"
            f"  2. Environment variable CHECKPOINT_PATH (not set)\n"
            f"  3. Default path: {default_path} (not found)\n\n"
            f"Please run 4_FineTuning.py first to create a model checkpoint,\n"
            f"then run 5_FindBestCheckpoint.py to select the best checkpoint."
        )

    print(f"Warning: No checkpoint found. Will try: {default_path}")
    return default_path


def get_checkpoint_info() -> Dict[str, Any]:
    """Get detailed information about the best checkpoint."""
    info = {
        'checkpoint_path': None,
        'checkpoint_name': None,
        'micro_f1': None,
        'source': 'default',
        'approach': 'chunking'
    }

    if BEST_CHECKPOINT_FILE.exists():
        try:
            with open(BEST_CHECKPOINT_FILE, 'r') as f:
                config = json.load(f)
            checkpoint_path = config.get('checkpoint')
            if checkpoint_path and os.path.exists(checkpoint_path):
                info['checkpoint_path'] = checkpoint_path
                info['checkpoint_name'] = config.get('checkpoint_name', Path(checkpoint_path).name)
                info['micro_f1'] = config.get('micro_f1')
                info['source'] = 'config_file'
                return info
        except (json.JSONDecodeError, IOError):
            pass

    env_checkpoint = os.environ.get('CHECKPOINT_PATH')
    if env_checkpoint and os.path.exists(env_checkpoint):
        info['checkpoint_path'] = env_checkpoint
        info['checkpoint_name'] = Path(env_checkpoint).name
        info['source'] = 'environment'
        return info

    info['checkpoint_path'] = str(DEFAULT_CHECKPOINT_PATH)
    info['checkpoint_name'] = 'final_model'
    return info


# =============================================================================
# CHECKPOINT EVALUATION SECTION
# =============================================================================

def format_f_code(code: str) -> str:
    """Format F-code with period at 3rd position if not present."""
    code = code.strip()
    if len(code) > 3 and '.' not in code:
        return f"{code[:3]}.{code[3:]}"
    return code


# Psychiatric F-code system prompt (CoT - must match training prompt)
SYSTEM_PROMPT = '''Analyze the clinical text step by step to extract psychiatric F-codes (F00-F99).

STEP-BY-STEP PROCESS:

Step 1: IDENTIFY psychiatric keywords and conditions
- Scan for mental health diagnoses, symptoms, medications
- Look for: depression, anxiety, substance use, cognitive disorders, psychosis
- Note psychiatric medications and their implications

Step 2: MAP identified conditions to appropriate F-codes
- Depression -> F32.9 (single episode) or F33.9 (recurrent)
- Anxiety -> F41.9 (general anxiety) or F41.0 (panic disorder)
- Delirium -> F05
- Dementia -> F03.90
- Bipolar -> F31.9
- PTSD -> F43.10
- Schizophrenia -> F20.9
- ADHD -> F90.9

SUBSTANCE USE DISORDERS (F10-F19):
- F10.xx = Alcohol, F11.xx = Opioids, F12.xx = Cannabis, F14.xx = Cocaine, F17.xx = Nicotine
- Severity: .10 = Abuse, .20 = Dependence, .90 = Unspecified
- Course: .x0 = Active, .x1 = In remission (ONLY if explicitly stated: "sober", "quit", "former")
- Examples: F10.20 (alcohol dependence), F17.210 (nicotine dependence, cigarettes)

Step 3: SELECT most confident codes (maximum 5)
- Prioritize explicitly mentioned diagnoses
- Include codes implied by medications (e.g., antidepressants -> depression)

Step 4: OUTPUT as JSON array only
["F32.9", "F17.210", "F41.9"]'''


class ChunkingCheckpointEvaluator:
    """Evaluate checkpoints using chunking approach with union aggregation."""

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.tokenizer = None

    def get_checkpoints(self) -> List[str]:
        """Get list of checkpoint directories sorted by step number."""
        checkpoint_dir = Path(self.config['checkpoint_dir'])
        checkpoints = []

        min_step = self.config.get('min_checkpoint_step', 0)

        if not checkpoint_dir.exists():
            print(f"Warning: Checkpoint directory not found: {checkpoint_dir}")
            return []

        for item in checkpoint_dir.iterdir():
            if item.is_dir() and item.name.startswith('checkpoint-'):
                try:
                    step = int(item.name.split('-')[1])
                    if step >= min_step:
                        checkpoints.append((step, str(item)))
                except ValueError:
                    continue

        checkpoints.sort(key=lambda x: x[0])
        return [cp[1] for cp in checkpoints]

    def load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        from unsloth import FastLanguageModel

        print(f"\n  Loading checkpoint: {checkpoint_path}")

        if self.model is not None:
            del self.model
            del self.tokenizer
            torch.cuda.empty_cache()

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=self.config['max_seq_length'],
            dtype=None,
            load_in_4bit=True,
            full_finetuning=False,
            device_map={"": 0},
            attn_implementation="eager",
        )

        FastLanguageModel.for_inference(self.model)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chunk_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into overlapping chunks.

        Returns:
            List of (chunk_text, start_token, end_token) tuples
        """
        chunk_size = self.config.get('chunk_size', CHUNK_SIZE)
        chunk_overlap = self.config.get('chunk_overlap', CHUNK_OVERLAP)

        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        total_tokens = len(tokens)

        # If text fits in one chunk, return as-is
        if total_tokens <= chunk_size:
            return [(text, 0, total_tokens)]

        chunks = []
        start = 0

        while start < total_tokens:
            end = min(start + chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            decoded_chunk = self.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append((decoded_chunk, start, end))

            if end >= total_tokens:
                break
            start = end - chunk_overlap

        return chunks

    def parse_response(self, response_text: str) -> List[str]:
        """Parse model response to extract F-codes."""
        try:
            cleaned = re.sub(r'```json\s*', '', response_text)
            cleaned = re.sub(r'```\s*', '', cleaned)
            cleaned = cleaned.strip()

            if cleaned.startswith('['):
                end_idx = cleaned.rfind(']') + 1
                if end_idx > 0:
                    cleaned = cleaned[:end_idx]
                data = json.loads(cleaned)
                if isinstance(data, list):
                    codes = []
                    for item in data:
                        if isinstance(item, dict) and 'code' in item:
                            codes.append(item['code'])
                        elif isinstance(item, str):
                            codes.append(item)
                    return [c for c in codes if c.startswith('F')]
        except json.JSONDecodeError:
            pass

        f_codes = re.findall(r'F\d{2}(?:\.\d{1,3})?', response_text)
        return list(dict.fromkeys(f_codes))[:5]

    def generate_chunk_prediction(self, chunk_text: str) -> List[str]:
        """Generate prediction for a single chunk."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Discharge Summary:\n{chunk_text}"}
        ]

        inputs = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get('max_new_tokens', MAX_NEW_TOKENS),
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        torch.cuda.empty_cache()

        input_length = inputs['input_ids'].shape[1]
        response = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
        return self.parse_response(response)

    def generate_prediction(self, text: str) -> Tuple[List[str], Dict]:
        """Generate prediction using chunking with union aggregation."""
        try:
            original_tokens = self.count_tokens(text)
            chunks = self.chunk_text(text)
            num_chunks = len(chunks)

            # UNION strategy: combine predictions from all chunks
            all_predicted_codes = set()

            for chunk_text_content, start, end in chunks:
                chunk_codes = self.generate_chunk_prediction(chunk_text_content)
                all_predicted_codes.update(chunk_codes)

            stats = {
                "original_tokens": original_tokens,
                "num_chunks": num_chunks,
                "chunk_size": self.config.get('chunk_size', CHUNK_SIZE),
                "chunk_overlap": self.config.get('chunk_overlap', CHUNK_OVERLAP),
            }

            return list(all_predicted_codes), stats

        except Exception as e:
            print(f"    Error: {e}")
            return [], {}

    def evaluate_checkpoint(self, checkpoint_path: str, val_df: pd.DataFrame) -> Dict:
        """Evaluate a single checkpoint on validation data using chunking."""
        self.load_model(checkpoint_path)

        total_tp = total_fp = total_fn = 0
        total_chunks = 0
        multi_chunk_count = 0

        for idx, (_, row) in enumerate(val_df.iterrows()):
            if (idx + 1) % 20 == 0:
                print(f"    Progress: {idx + 1}/{len(val_df)}")

            actual_codes = set([format_f_code(c) for c in str(row['f_codes_str']).split(',') if c.strip().startswith('F')])
            predicted_codes, stats = self.generate_prediction(row['text'])
            predicted_codes = set(predicted_codes)

            # Track chunking stats
            num_chunks = stats.get('num_chunks', 1)
            total_chunks += num_chunks
            if num_chunks > 1:
                multi_chunk_count += 1

            tp = len(actual_codes & predicted_codes)
            fp = len(predicted_codes - actual_codes)
            fn = len(actual_codes - predicted_codes)

            total_tp += tp
            total_fp += fp
            total_fn += fn

        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                    if (micro_precision + micro_recall) > 0 else 0.0)

        return {
            'checkpoint': checkpoint_path,
            'checkpoint_name': Path(checkpoint_path).name,
            'num_samples': len(val_df),
            'micro_precision': micro_precision,
            'micro_recall': micro_recall,
            'micro_f1': micro_f1,
            'total_tp': total_tp,
            'total_fp': total_fp,
            'total_fn': total_fn,
            'chunking_stats': {
                'total_chunks': total_chunks,
                'avg_chunks_per_sample': total_chunks / len(val_df) if len(val_df) > 0 else 0,
                'multi_chunk_count': multi_chunk_count,
                'multi_chunk_rate': multi_chunk_count / len(val_df) if len(val_df) > 0 else 0,
            },
            'timestamp': datetime.now().isoformat()
        }

    def find_best_checkpoint(self) -> Dict:
        """Evaluate all checkpoints and find the best one."""
        print("=" * 70)
        print("FINDING BEST CHECKPOINT (CHUNKING APPROACH)")
        print("=" * 70)

        # Load validation data
        print(f"\nLoading validation data: {self.config['val_data_path']}")
        val_df = pd.read_csv(self.config['val_data_path'])
        val_df = val_df[val_df['f_codes_str'].notna()]

        # Sample validation data
        val_samples = self.config['val_samples']
        if val_samples and len(val_df) > val_samples:
            val_df = val_df.sample(n=val_samples, random_state=self.config['random_seed'])
        print(f"Using {len(val_df)} validation samples")

        # Get checkpoints
        checkpoints = self.get_checkpoints()

        # If no checkpoints found, use final_model
        if not checkpoints:
            final_model = self.config.get('final_model_path', str(DEFAULT_CHECKPOINT_PATH))
            if os.path.exists(final_model):
                print(f"\nNo checkpoints found. Using final model: {final_model}")
                checkpoints = [final_model]
            else:
                print("ERROR: No checkpoints or final model found!")
                return {}

        print(f"\nFound {len(checkpoints)} checkpoint(s):")
        for cp in checkpoints:
            print(f"  - {Path(cp).name}")

        # Evaluate each checkpoint
        results = []
        for i, checkpoint in enumerate(checkpoints):
            print(f"\n[{i + 1}/{len(checkpoints)}] Evaluating {Path(checkpoint).name}")
            result = self.evaluate_checkpoint(checkpoint, val_df)
            results.append(result)
            print(f"    Micro F1: {result['micro_f1']:.4f}")

        # Find best checkpoint
        best_result = max(results, key=lambda x: x['micro_f1'])

        print("\n" + "=" * 70)
        print("CHECKPOINT COMPARISON RESULTS (CHUNKING)")
        print("=" * 70)
        print(f"\n{'Checkpoint':<20} {'Micro P':<10} {'Micro R':<10} {'Micro F1':<10}")
        print("-" * 50)
        for result in results:
            marker = " *" if result['checkpoint'] == best_result['checkpoint'] else ""
            print(f"{result['checkpoint_name']:<20} "
                  f"{result['micro_precision']:.4f}    "
                  f"{result['micro_recall']:.4f}    "
                  f"{result['micro_f1']:.4f}{marker}")

        print(f"\nBest Checkpoint: {best_result['checkpoint_name']}")
        print(f"  Micro F1: {best_result['micro_f1']:.4f}")
        print(f"  Micro Precision: {best_result['micro_precision']:.4f}")
        print(f"  Micro Recall: {best_result['micro_recall']:.4f}")

        return {
            'best_checkpoint': best_result,
            'all_results': results,
            'config': {
                'val_samples': len(val_df),
                'random_seed': self.config['random_seed'],
                'approach': 'chunking',
                'chunk_size': self.config.get('chunk_size', CHUNK_SIZE),
                'chunk_overlap': self.config.get('chunk_overlap', CHUNK_OVERLAP)
            }
        }


def visualize_checkpoint_results(results: Dict, output_dir: Path, total_steps: int = 12500):
    """Generate visualization of checkpoint evaluation results."""
    all_results = results['all_results']
    val_samples = results['config']['val_samples']

    if len(all_results) < 2:
        print("Skipping visualization: need at least 2 checkpoints")
        return

    # Extract data
    checkpoints = []
    micro_f1_scores = []

    for r in all_results:
        checkpoint_name = r['checkpoint_name']
        if 'checkpoint-' in checkpoint_name:
            step = int(checkpoint_name.split('-')[1])
        else:
            step = total_steps
        checkpoints.append(step)
        micro_f1_scores.append(r['micro_f1'])

    best_idx = np.argmax(micro_f1_scores)
    best_step = checkpoints[best_idx]
    best_f1 = micro_f1_scores[best_idx]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(checkpoints, micro_f1_scores, 'b-o', linewidth=2, markersize=8, label='Micro F1')
    ax.scatter([best_step], [best_f1], color='red', s=200, zorder=5,
                edgecolors='darkred', linewidths=2, label=f'Best (Step {best_step})')
    ax.axhline(y=best_f1, color='red', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Micro F1 Score', fontsize=12)
    ax.set_title(f'Checkpoint Selection (Chunking Approach)\n(Validation Set, n={val_samples})', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right')

    plt.tight_layout()

    # Save figures
    png_path = output_dir / 'checkpoint_selection.png'
    plt.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\nFigure saved: {png_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Find best checkpoint for chunking approach')
    parser.add_argument('--checkpoint-dir', type=str, default=str(DEFAULT_CHECKPOINT_DIR),
                        help='Directory containing checkpoints')
    parser.add_argument('--val-data', type=str, default=str(DEFAULT_VAL_DATA_PATH),
                        help='Path to validation data CSV')
    parser.add_argument('--output-dir', type=str, default=OUTPUT_DIR_5_CHECKPOINT,
                        help='Output directory for results')
    parser.add_argument('--val-samples', type=int, default=200,
                        help='Number of validation samples (default: 200)')
    parser.add_argument('--max-seq-length', type=int, default=MAX_SEQ_LENGTH,
                        help='Maximum sequence length')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE,
                        help='Chunk size in tokens (default: 2800)')
    parser.add_argument('--chunk-overlap', type=int, default=CHUNK_OVERLAP,
                        help='Chunk overlap in tokens (default: 200)')
    parser.add_argument('--random-seed', type=int, default=RANDOM_SEED,
                        help='Random seed')
    parser.add_argument('--min-checkpoint-step', type=int, default=MIN_CHECKPOINT_STEP,
                        help=f'Minimum checkpoint step to evaluate (default: {MIN_CHECKPOINT_STEP})')

    args = parser.parse_args()

    config = {
        'checkpoint_dir': args.checkpoint_dir,
        'val_data_path': args.val_data,
        'output_dir': args.output_dir,
        'val_samples': args.val_samples,
        'max_seq_length': args.max_seq_length,
        'chunk_size': args.chunk_size,
        'chunk_overlap': args.chunk_overlap,
        'max_new_tokens': MAX_NEW_TOKENS,
        'random_seed': args.random_seed,
        'min_checkpoint_step': args.min_checkpoint_step,
        'final_model_path': str(DEFAULT_CHECKPOINT_PATH),
    }

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find best checkpoint
    evaluator = ChunkingCheckpointEvaluator(config)
    results = evaluator.find_best_checkpoint()

    if not results:
        print("ERROR: No results generated")
        return 1

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_path = output_dir / f"checkpoint_comparison_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_path}")

    # Save comparison CSV
    comparison_df = pd.DataFrame(results['all_results'])
    csv_path = output_dir / f"checkpoint_comparison_{timestamp}.csv"
    comparison_df.to_csv(csv_path, index=False)
    print(f"Comparison CSV saved to: {csv_path}")

    # Save best checkpoint config to project root
    best = results['best_checkpoint']
    best_config = {
        'checkpoint': best['checkpoint'],
        'checkpoint_name': best['checkpoint_name'],
        'micro_f1': best['micro_f1'],
        'micro_precision': best['micro_precision'],
        'micro_recall': best['micro_recall'],
        'num_samples': best['num_samples'],
        'approach': 'chunking',
        'chunk_size': config['chunk_size'],
        'chunk_overlap': config['chunk_overlap'],
        'timestamp': best['timestamp']
    }

    with open(BEST_CHECKPOINT_FILE, 'w') as f:
        json.dump(best_config, f, indent=2)

    print(f"\n{'='*70}")
    print(f"BEST CHECKPOINT SAVED TO: {BEST_CHECKPOINT_FILE}")
    print(f"{'='*70}")
    print(f"  Checkpoint: {best['checkpoint_name']}")
    print(f"  Path: {best['checkpoint']}")
    print(f"  Micro F1: {best['micro_f1']:.4f}")
    print(f"\nEvaluation scripts will now use this checkpoint.")

    # Generate visualization
    visualize_checkpoint_results(results, output_dir)

    return 0


if __name__ == "__main__":
    exit(main())
