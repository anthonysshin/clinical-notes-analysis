#!/usr/bin/env python3
"""
Evaluation Strategy 6b: FEW-SHOT EXEMPLAR with CHUNKING
Purpose: Test in-context learning with concrete examples
Text Handling: Chunking (2800 tokens per chunk, 200 token overlap) with Union aggregation
Preprocessing: Chunking on-the-fly
CoT Reasoning: NO
F-code Mappings: NO (only implicit in examples)

Tests whether few-shot examples improve performance over zero-shot.

Checkpoint Selection:
    Automatically uses best checkpoint from best_checkpoint.json (created by 5_FindBestCheckpoint.py).
    Override with environment variable: CHECKPOINT_PATH=/path/to/checkpoint
"""

import os
os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"

# Import centralized config for reproducibility and settings
from config import (
    RANDOM_SEED, set_all_seeds, TEST_DATA_PATH,
    MAX_SEQ_LENGTH, CHUNK_SIZE, CHUNK_OVERLAP,
    MAX_NEW_TOKENS, LOAD_IN_4BIT, TEMPERATURE, DEFAULT_MAX_SAMPLES,
    EVAL_OUTPUT_DIR_6B
)

# Set random seeds for reproducibility
set_all_seeds(RANDOM_SEED)

import sys
import json
import pandas as pd
import torch
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Import checkpoint selection helper
sys.path.insert(0, str(Path(__file__).parent))
from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location("checkpoint_finder", Path(__file__).parent / "5_FindBestCheckpoint.py")
_checkpoint_module = module_from_spec(_spec)
_spec.loader.exec_module(_checkpoint_module)
get_best_checkpoint_path = _checkpoint_module.get_best_checkpoint_path

from unsloth import FastLanguageModel


def format_f_code(code: str) -> str:
    """Format F-code with period at 3rd position if not present."""
    code = code.strip()
    if len(code) > 3 and '.' not in code:
        return f"{code[:3]}.{code[3:]}"
    return code


# Automatically get best checkpoint (from best_checkpoint.json or default)
# CHUNKING Configuration
# Token Allocation (max_seq_length = 4096):
#   System Prompt (few-shot):  ~350 tokens
#   User Template:             ~30 tokens
#   Chunk:                     2800 tokens
#   Output (JSON array):       100 tokens
#   Buffer:                    ~816 tokens
#   TOTAL:                     4096 tokens
CONFIG = {
    'checkpoint_path': get_best_checkpoint_path(),
    'max_seq_length': MAX_SEQ_LENGTH,
    'chunk_size': CHUNK_SIZE,
    'chunk_overlap': CHUNK_OVERLAP,
    'max_new_tokens': MAX_NEW_TOKENS,
    'max_samples': DEFAULT_MAX_SAMPLES,
    'test_data_path': TEST_DATA_PATH,
    'output_dir': EVAL_OUTPUT_DIR_6B,
    'load_in_4bit': LOAD_IN_4BIT,
    'temperature': TEMPERATURE,
    'random_seed': RANDOM_SEED
}

SYSTEM_PROMPT = '''Extract psychiatric F-codes (F00-F99) from clinical text.

### Examples:

Example 1:
Input: Patient reports persistent depressive symptoms with sleep disturbances and low energy.
Output: ["F32.9"]

Example 2:
Input: History of alcohol use disorder, currently in remission. Patient has anxiety symptoms.
Output: ["F10.21", "F41.9"]

Example 3:
Input: Patient with bipolar disorder on lithium. Also reports tobacco use, one pack per day.
Output: ["F31.9", "F17.210"]

Example 4:
Input: Admitted with delirium secondary to UTI. History of major depressive disorder.
Output: ["F05", "F32.9"]

### Instructions:
- Extract ONLY psychiatric F-codes (F00-F99)
- Maximum 5 most confident codes

Format response as JSON array:
["F32.9", "F17.210"]'''


class FewShotExemplarEvaluator:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.tokenizer = None

    def load_model(self):
        checkpoint_path = self.config['checkpoint_path']
        print(f"\n{'='*80}")
        print(f"Strategy 6b: FEW-SHOT EXEMPLAR with CHUNKING")
        print(f"Text Handling: Chunking ({self.config['chunk_size']} tokens, {self.config['chunk_overlap']} overlap) + Union")
        print(f"Preprocessing: Chunking on-the-fly | CoT: NO | Mappings: NO (implicit in examples)")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*80}\n", flush=True)

        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=self.config['max_seq_length'],
            dtype=None,
            load_in_4bit=self.config['load_in_4bit'],
            full_finetuning=False,
            device_map={"": 0},
            attn_implementation="eager",
        )

        FastLanguageModel.for_inference(self.model)
        print("Model loaded successfully!\n", flush=True)

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=False))

    def chunk_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Split text into overlapping chunks.

        Returns:
            List of (chunk_text, start_token, end_token) tuples
        """
        chunk_size = self.config['chunk_size']
        chunk_overlap = self.config['chunk_overlap']

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
        """Parse F-codes from model response (expects simple JSON array)."""
        try:
            cleaned = re.sub(r'```json\s*', '', response_text)
            cleaned = re.sub(r'```\s*', '', cleaned)

            left = cleaned.find('[')
            right = cleaned.rfind(']')

            if left == -1 or right == -1:
                # Fallback: extract F-codes using regex
                return list(set(re.findall(r'F\d{2}(?:\.\d+)?', response_text)))

            json_part = cleaned[left:right+1]

            try:
                parsed = json.loads(json_part)
                # Handle simple JSON array: ["F32.9", "F17.210"]
                if isinstance(parsed, list):
                    codes = []
                    for item in parsed:
                        if isinstance(item, str) and item.startswith('F'):
                            codes.append(item)
                        elif isinstance(item, dict) and 'code' in item:
                            # Backward compatibility with old format
                            codes.append(item['code'])
                    return codes
                return []
            except json.JSONDecodeError:
                return list(set(re.findall(r'F\d{2}(?:\.\d+)?', response_text)))
        except Exception:
            return []

    def generate_chunk_prediction(self, chunk_text: str) -> List[str]:
        """Generate prediction for a single chunk."""
        user_prompt = f"Discharge Summary:\n{chunk_text}"

        conversation = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        inputs = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config['max_new_tokens'],
                temperature=self.config['temperature'],
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )

        torch.cuda.empty_cache()

        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        assistant_marker = "<|start|>assistant"
        if assistant_marker in full_response:
            generated = full_response.split(assistant_marker)[-1].strip()
        else:
            input_length = inputs['input_ids'].shape[1]
            generated = self.tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

        return self.parse_response(generated)

    def generate_prediction(self, clinical_text: str) -> Dict:
        """Generate prediction using chunking with union aggregation."""
        try:
            original_tokens = self.count_tokens(clinical_text)
            chunks = self.chunk_text(clinical_text)
            num_chunks = len(chunks)

            # UNION strategy: combine predictions from all chunks
            all_predicted_codes = set()
            chunk_predictions = []

            for chunk_text_content, start, end in chunks:
                chunk_codes = self.generate_chunk_prediction(chunk_text_content)
                all_predicted_codes.update(chunk_codes)
                chunk_predictions.append({
                    "start_token": start,
                    "end_token": end,
                    "codes": chunk_codes
                })

            return {
                "predicted_codes": list(all_predicted_codes),
                "chunk_predictions": chunk_predictions,
                "success": True,
                "extraction_stats": {
                    "text_handling": "chunking_union",
                    "original_tokens": original_tokens,
                    "num_chunks": num_chunks,
                    "chunk_size": self.config['chunk_size'],
                    "chunk_overlap": self.config['chunk_overlap'],
                }
            }
        except Exception as e:
            print(f"Error: {e}", flush=True)
            return {
                "predicted_codes": [],
                "chunk_predictions": [],
                "success": False,
                "extraction_stats": {}
            }

    def load_test_data(self) -> pd.DataFrame:
        print(f"Loading test data: {self.config['test_data_path']}", flush=True)
        df = pd.read_csv(self.config['test_data_path'])
        df = df[df['f_codes_str'].notna()]
        df = df[df['f_code_count'] >= 1]

        max_samples = self.config['max_samples']
        if max_samples and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=self.config['random_seed'])

        print(f"Loaded {len(df)} test samples\n", flush=True)
        return df

    def parse_actual_codes(self, codes_str: str) -> List[str]:
        if pd.isna(codes_str) or codes_str == '':
            return []
        codes = [format_f_code(code) for code in str(codes_str).split(',')]
        return [code for code in codes if code.startswith('F')]

    def calculate_metrics(self, actual: List[str], predicted: List[str]) -> Dict[str, float]:
        actual_set = set(actual)
        predicted_set = set(predicted)

        if len(actual_set) == 0 and len(predicted_set) == 0:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "tp": 0, "fp": 0, "fn": 0}

        tp = len(actual_set.intersection(predicted_set))
        fp = len(predicted_set - actual_set)
        fn = len(actual_set - predicted_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}

    def analyze_per_code_performance(self, results: List[Dict]) -> Dict:
        code_stats = {}

        for result in results:
            actual = set(result['actual_codes'])
            predicted = set(result['predicted_codes'])

            all_codes = actual | predicted
            for code in all_codes:
                if code not in code_stats:
                    code_stats[code] = {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0}

                if code in actual:
                    code_stats[code]['count'] += 1
                    if code in predicted:
                        code_stats[code]['tp'] += 1
                    else:
                        code_stats[code]['fn'] += 1
                else:
                    code_stats[code]['fp'] += 1

        per_code_metrics = {}
        for code, stats in code_stats.items():
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

            per_code_metrics[code] = {
                'occurrences': stats['count'],
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }

        return per_code_metrics

    def evaluate(self) -> Tuple[List[Dict], Dict[str, float]]:
        print(f"\n{'='*80}")
        print(f"Evaluating: Strategy 6b - FEW-SHOT EXEMPLAR with CHUNKING")
        print(f"Text Handling: Chunking ({self.config['chunk_size']} tokens, {self.config['chunk_overlap']} overlap) + Union")
        print(f"Samples: {self.config['max_samples']}")
        print(f"{'='*80}\n", flush=True)

        test_df = self.load_test_data()
        results = []
        total_tp = total_fp = total_fn = 0
        successful = 0

        # Chunking statistics
        total_chunks = 0
        multi_chunk_count = 0

        start_time = datetime.now()

        for idx, (_, row) in enumerate(test_df.iterrows()):
            if (idx + 1) % 50 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = (idx + 1) / elapsed
                remaining = (len(test_df) - (idx + 1)) / rate if rate > 0 else 0
                print(f"Progress: {idx+1}/{len(test_df)} ({(idx+1)/len(test_df)*100:.1f}%) - "
                      f"ETA: {remaining/60:.1f}min", flush=True)

            actual_codes = self.parse_actual_codes(row['f_codes_str'])
            prediction = self.generate_prediction(row['text'])

            if prediction["success"]:
                successful += 1
                stats = prediction.get("extraction_stats", {})
                num_chunks = stats.get("num_chunks", 1)
                total_chunks += num_chunks
                if num_chunks > 1:
                    multi_chunk_count += 1

            metrics = self.calculate_metrics(actual_codes, prediction["predicted_codes"])

            total_tp += metrics["tp"]
            total_fp += metrics["fp"]
            total_fn += metrics["fn"]

            result = {
                "sample_id": int(row['hadm_id']),
                "subject_id": int(row['subject_id']),
                "actual_codes": actual_codes,
                "predicted_codes": prediction["predicted_codes"],
                "success": prediction["success"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "tp": metrics["tp"],
                "fp": metrics["fp"],
                "fn": metrics["fn"],
                "perfect_match": 1 if metrics["f1"] == 1.0 else 0,
                "extraction_stats": prediction.get("extraction_stats", {})
            }
            results.append(result)

        # Calculate overall metrics
        micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

        # Calculate total evaluation time
        total_evaluation_time = (datetime.now() - start_time).total_seconds()

        # Calculate per-code metrics first (needed for true Macro F1)
        per_code_metrics = self.analyze_per_code_performance(results)

        # Macro F1: Average F1 per-code (label-averaged), matching PLM-ICD standard
        code_f1_scores = [m['f1'] for m in per_code_metrics.values()]
        code_precision_scores = [m['precision'] for m in per_code_metrics.values()]
        code_recall_scores = [m['recall'] for m in per_code_metrics.values()]

        macro_precision = sum(code_precision_scores) / len(code_precision_scores) if code_precision_scores else 0.0
        macro_recall = sum(code_recall_scores) / len(code_recall_scores) if code_recall_scores else 0.0
        macro_f1 = sum(code_f1_scores) / len(code_f1_scores) if code_f1_scores else 0.0

        # Sample-averaged F1 for reference
        sample_avg_f1 = sum(r["f1"] for r in results) / len(results) if results else 0.0

        avg_chunks = total_chunks / len(results) if results else 0.0

        overall_metrics = {
            "checkpoint": self.config['checkpoint_path'],
            "total_samples": len(results),
            "successful_predictions": successful,
            "success_rate": successful / len(results) if results else 0.0,
            "micro_precision": micro_precision,
            "micro_recall": micro_recall,
            "micro_f1": micro_f1,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "sample_avg_f1": sample_avg_f1,
            "num_unique_codes": len(per_code_metrics),
            "perfect_matches": sum(r["perfect_match"] for r in results),
            "perfect_match_rate": sum(r["perfect_match"] for r in results) / len(results) if results else 0.0,
            "chunking_stats": {
                "chunk_size": self.config['chunk_size'],
                "chunk_overlap": self.config['chunk_overlap'],
                "total_chunks": total_chunks,
                "avg_chunks_per_sample": avg_chunks,
                "multi_chunk_samples": multi_chunk_count,
                "multi_chunk_rate": multi_chunk_count / len(results) if results else 0.0,
            },
            "evaluation_time_seconds": total_evaluation_time,
            "per_code_metrics": per_code_metrics
        }

        print(f"\n{'='*80}")
        print(f"Strategy 6b: FEW-SHOT EXEMPLAR with CHUNKING - RESULTS")
        print(f"{'='*80}")
        print(f"Micro F1: {overall_metrics['micro_f1']:.4f}")
        print(f"Macro F1 (per-code avg): {overall_metrics['macro_f1']:.4f}")
        print(f"Sample-avg F1: {overall_metrics['sample_avg_f1']:.4f}")
        print(f"Micro Precision: {overall_metrics['micro_precision']:.4f}")
        print(f"Micro Recall: {overall_metrics['micro_recall']:.4f}")
        print(f"Unique codes: {overall_metrics['num_unique_codes']}")
        print(f"Perfect matches: {overall_metrics['perfect_matches']}/{len(results)} ({overall_metrics['perfect_match_rate']:.1%})")
        print(f"\nChunking Stats:")
        print(f"  Total chunks: {total_chunks}")
        print(f"  Avg chunks/sample: {avg_chunks:.2f}")
        print(f"  Multi-chunk samples: {multi_chunk_count}/{len(results)} ({multi_chunk_count/len(results)*100:.1f}%)")
        print(f"\nEvaluation Time: {total_evaluation_time/60:.1f} minutes ({total_evaluation_time:.0f} seconds)")
        print(f"{'='*80}\n", flush=True)

        return results, overall_metrics

    def save_results(self, results: List[Dict], metrics: Dict[str, float]):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')

        evaluation_report = {
            "evaluation_info": {
                "timestamp": datetime.now().isoformat(),
                "strategy": "6b_FewShotExemplar",
                "strategy_description": "Few-shot learning with chunking (union aggregation)",
                "text_handling": "chunking_union",
                "chunk_size": self.config['chunk_size'],
                "chunk_overlap": self.config['chunk_overlap'],
                "preprocessing": False,
                "cot_reasoning": False,
                "fcode_mappings": False,
                "checkpoint_path": self.config['checkpoint_path'],
                "max_new_tokens": self.config['max_new_tokens'],
                "max_samples": self.config['max_samples'],
            },
            "configuration": self.config,
            "performance_metrics": metrics,
            "sample_results": results
        }

        json_path = self.output_dir / f"evaluation_results_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_report, f, indent=2, ensure_ascii=False)
        print(f"Results saved: {json_path}", flush=True)

        # Save predictions CSV
        csv_data = []
        for result in results:
            csv_data.append({
                "sample_id": result["sample_id"],
                "subject_id": result["subject_id"],
                "actual_f_codes": "; ".join(result["actual_codes"]),
                "predicted_f_codes": "; ".join(result["predicted_codes"]),
                "num_chunks": result.get("extraction_stats", {}).get("num_chunks", 1),
                "precision": result["precision"],
                "recall": result["recall"],
                "f1_score": result["f1"],
                "perfect_match": result["perfect_match"],
                "tp": result["tp"],
                "fp": result["fp"],
                "fn": result["fn"],
            })

        csv_path = self.output_dir / f"predictions_{timestamp}.csv"
        pd.DataFrame(csv_data).to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Predictions saved: {csv_path}", flush=True)

        # Save per-code performance
        if 'per_code_metrics' in metrics:
            per_code_data = []
            for code, code_metrics in sorted(metrics['per_code_metrics'].items(),
                                            key=lambda x: x[1]['occurrences'],
                                            reverse=True):
                per_code_data.append({
                    "f_code": code,
                    "occurrences": code_metrics['occurrences'],
                    "f1_score": round(code_metrics['f1'], 4),
                    "precision": round(code_metrics['precision'], 4),
                    "recall": round(code_metrics['recall'], 4),
                    "tp": code_metrics['tp'],
                    "fp": code_metrics['fp'],
                    "fn": code_metrics['fn']
                })

            per_code_path = self.output_dir / f"per_code_performance_{timestamp}.csv"
            pd.DataFrame(per_code_data).to_csv(per_code_path, index=False, encoding='utf-8')
            print(f"Per-code performance saved: {per_code_path}", flush=True)


def main():
    print("\n" + "=" * 80)
    print("EVALUATION STRATEGY 6b: FEW-SHOT EXEMPLAR with CHUNKING")
    print(f"Text Handling: Chunking ({CHUNK_SIZE} tokens, {CHUNK_OVERLAP} overlap) + Union aggregation")
    print("Preprocessing: Chunking on-the-fly | CoT: NO | Mappings: NO")
    print("=" * 80)

    evaluator = FewShotExemplarEvaluator(CONFIG)

    try:
        evaluator.load_model()
        results, metrics = evaluator.evaluate()
        evaluator.save_results(results, metrics)

        print("\n" + "=" * 80)
        print("FINAL RESULTS - Strategy 6b: FEW-SHOT EXEMPLAR with CHUNKING")
        print("=" * 80)
        print(f"Micro F1:              {metrics['micro_f1']:.4f}")
        print(f"Macro F1 (per-code):   {metrics['macro_f1']:.4f}")
        print(f"Sample-avg F1:         {metrics['sample_avg_f1']:.4f}")
        print(f"Perfect Match:         {metrics['perfect_match_rate']:.1%}")
        print(f"Unique codes:          {metrics['num_unique_codes']}")
        print(f"\nChunking Statistics:")
        print(f"  Avg chunks/sample:   {metrics['chunking_stats']['avg_chunks_per_sample']:.2f}")
        print(f"  Multi-chunk rate:    {metrics['chunking_stats']['multi_chunk_rate']:.1%}")
        print("=" * 80 + "\n")

        return 0
    except Exception as e:
        print(f"\nError: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
