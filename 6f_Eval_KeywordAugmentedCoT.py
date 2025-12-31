#!/usr/bin/env python3
"""
Evaluation Strategy 6f: KEYWORD-AUGMENTED CHAIN-OF-THOUGHT
Purpose: Best performing strategy with keyword preprocessing and CoT reasoning
Text Handling: Keyword extraction only (no chunking) - extracts psychiatric-relevant sentences
Preprocessing: PSYCH_KEYWORDS extraction
CoT Reasoning: YES
F-code Mappings: YES (with key distinctions + ambiguous code handling)

Features:
- Keyword-based psychiatric content extraction using PSYCH_KEYWORDS
- Single inference on extracted content (no chunking needed)
- Step-by-step Chain-of-Thought reasoning
- Guidance for ambiguous/NOS codes (F29, F39, F09, F99)
- Guidance for unspecified substance use codes (.90)
- Key distinctions for severity levels

Checkpoint Selection:
    Automatically uses best checkpoint from best_checkpoint.json (created by 5_FindBestCheckpoint.py).
    Override with environment variable: CHECKPOINT_PATH=/path/to/checkpoint
"""

import os
os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"

# Import centralized config for reproducibility and settings
from config import (
    RANDOM_SEED, set_all_seeds, TEST_DATA_PATH,
    MAX_SEQ_LENGTH, MAX_NEW_TOKENS, LOAD_IN_4BIT, TEMPERATURE, DEFAULT_MAX_SAMPLES,
    EVAL_OUTPUT_DIR_6F
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


# Automatically get best checkpoint
# KEYWORD EXTRACTION Configuration
# Token Allocation (max_seq_length = 4096):
#   System Prompt (CoT):    ~550 tokens
#   User Template:          ~30 tokens
#   Extracted Content:      ~3300 tokens (keyword extraction reduces text significantly)
#   Output (JSON array):    100 tokens
#   Buffer:                 ~116 tokens
#   TOTAL:                  4096 tokens
CONFIG = {
    'checkpoint_path': get_best_checkpoint_path(),
    'max_seq_length': MAX_SEQ_LENGTH,
    'max_new_tokens': MAX_NEW_TOKENS,
    'max_samples': DEFAULT_MAX_SAMPLES,
    'test_data_path': TEST_DATA_PATH,
    'output_dir': EVAL_OUTPUT_DIR_6F,
    'load_in_4bit': LOAD_IN_4BIT,
    'temperature': TEMPERATURE,
    'random_seed': RANDOM_SEED
}

# PSYCH_KEYWORDS for psychiatric content extraction
PSYCH_KEYWORDS = [
    # F-codes (37 codes covering ~80% of occurrences)
    'F01.50', 'F02.80', 'F02.81', 'F03.90', 'F03.91', 'F05', 'F10.10', 'F10.11',
    'F10.20', 'F10.21', 'F11.10', 'F11.20', 'F12.10', 'F14.10', 'F17.200', 'F17.210',
    'F20.0', 'F20.9', 'F25.9', 'F31.81', 'F31.9', 'F32.3', 'F32.9', 'F33.2', 'F39',
    'F40.240', 'F41.0', 'F41.1', 'F41.8', 'F41.9', 'F42.9', 'F43.10', 'F43.20',
    'F43.23', 'F60.3', 'F79', 'F90.9',
    # Abbreviations (5)
    'ADHD', 'GAD', 'MDD', 'OCD', 'PTSD',
    # Conditions (47)
    'abuse', 'alcohol', 'alcohol abuse', 'alcohol use disorder', 'alcoholic cirrhosis',
    'alzheimer', 'anorexia', 'anxiety', 'anxious', 'asperger', 'attention deficit',
    'autism', 'bipolar', 'borderline', 'bulimia', 'cognitive', 'cognitive impairment',
    'confused', 'confusion', 'delirium', 'dementia', 'depressed', 'depression',
    'depressive', 'disorder', 'eating disorder', 'generalized anxiety',
    'hepatic encephalopathy', 'insomnia', 'major', 'major depressive', 'mania',
    'manic', 'mental', 'nicotine', 'obsessive', 'panic', 'polysubstance abuse',
    'post-traumatic', 'psychiatric', 'psychosis', 'psychotic', 'schizophrenia',
    'seizure', 'seizures', 'sleep disorder', 'sleep disturbance', 'smoker', 'smoking',
    'stress disorder', 'substance', 'substance abuse', 'tobacco', 'tobacco abuse',
    'tobacco use', 'trauma',
    # Medications (31)
    'alprazolam', 'amitriptyline', 'aripiprazole', 'benzodiazepine', 'bupropion',
    'buspirone', 'carbamazepine', 'citalopram', 'clonazepam', 'dextroamphetamine',
    'diazepam', 'duloxetine', 'escitalopram', 'fluoxetine', 'lamotrigine', 'lithium',
    'lorazepam', 'memantine', 'methylphenidate', 'midazolam', 'mirtazapine',
    'nortriptyline', 'olanzapine', 'paroxetine', 'quetiapine', 'risperidone',
    'sertraline', 'trazodone', 'valproate', 'venlafaxine', 'zolpidem',
    # Section headers (12)
    'diagnoses', 'diagnoses:', 'diagnosis', 'diagnosis:', 'discharge medications:',
    'home medications:', 'medication', 'medication:', 'medications', 'medications:',
    'mental status:', 'psychiatric diagnoses:', 'psychiatric history:',
    'social history', 'social history:', 'substance use:'
]

SYSTEM_PROMPT = '''Analyze the clinical text step by step to extract psychiatric F-codes (F00-F99).

STEP-BY-STEP PROCESS:

Step 1: IDENTIFY psychiatric conditions
- Look for: depression, anxiety, substance use, cognitive disorders, psychosis
- Note psychiatric medications and their implications

Step 2: MAP to F-codes with KEY DISTINCTIONS:

Depression:
- F32.9 (single episode) vs F33.x (recurrent - if history of prior episodes)

Anxiety:
- F41.9 (unspecified), F41.1 (GAD), F41.0 (panic), F43.10 (PTSD)

Substance Use (distinguish severity):
- Abuse (.10): harmful use pattern
- Dependence (.20): tolerance/withdrawal present
- Unspecified (.90): use mentioned but severity unclear
- Examples: F10.10 (alcohol abuse), F10.20 (alcohol dependence), F12.90 (cannabis use unspecified)

Cognitive:
- F05 (delirium)
- F03.90 (dementia without behavioral disturbance)
- F03.91 (dementia WITH behavioral disturbance)
- F02.80/F02.81 (dementia in other diseases, without/with behavioral)

NOS/Unspecified codes (use when specific type unclear):
- F29: Psychosis NOS (psychotic symptoms without clear schizophrenia/bipolar)
- F39: Mood disorder NOS (mood symptoms without clear depression/bipolar)
- F09: Mental disorder due to physiological condition, unspecified
- Substance .90 codes: F11.90 (opioid), F12.90 (cannabis), F14.90 (cocaine), F15.90 (stimulant)

Other: F17.210 (tobacco), F31.9 (bipolar), F20.9 (schizophrenia), F90.9 (ADHD)

Step 3: SELECT up to 5 most confident codes

Step 4: OUTPUT as JSON array only
["F32.9", "F17.210", "F41.9"]'''


class KeywordAugmentedCoTEvaluator:
    def __init__(self, config: dict):
        self.config = config
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = None
        self.tokenizer = None

    def load_model(self):
        checkpoint_path = self.config['checkpoint_path']
        print(f"\n{'='*80}")
        print(f"Strategy 6f: KEYWORD-AUGMENTED CHAIN-OF-THOUGHT")
        print(f"Text Handling: Keyword extraction only (no chunking)")
        print(f"Preprocessing: PSYCH_KEYWORDS extraction | CoT: YES | Key Distinctions: YES")
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

    def extract_psychiatric_content(self, text: str) -> Tuple[str, Dict]:
        """Extract psychiatric-relevant content using PSYCH_KEYWORDS.

        Returns:
            Tuple of (extracted_text, stats_dict)
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        relevant = []

        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(kw.lower() in sentence_lower for kw in PSYCH_KEYWORDS):
                relevant.append(sentence)

        extracted = ' '.join(relevant)
        extraction_method = "psychiatric_extraction"

        # If no psychiatric content found, fallback to original text
        if not extracted or len(extracted.strip()) < 100:
            extracted = text
            extraction_method = "full_text_fallback"

        stats = {
            "extraction_method": extraction_method,
            "original_tokens": self.count_tokens(text),
            "extracted_tokens": self.count_tokens(extracted),
        }

        return extracted, stats

    def parse_response(self, response_text: str) -> List[str]:
        """Parse F-codes from model response (expects simple JSON array: ["F32.9", "F17.210"])."""
        try:
            cleaned = re.sub(r'```json\s*', '', response_text)
            cleaned = re.sub(r'```\s*', '', cleaned)

            left = cleaned.find('[')
            right = cleaned.rfind(']')

            if left == -1 or right == -1:
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

    def generate_single_prediction(self, extracted_text: str) -> List[str]:
        """Generate prediction for extracted psychiatric content."""
        user_prompt = f"Discharge Summary:\n{extracted_text}"

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
        """Generate prediction using keyword extraction only (no chunking)."""
        try:
            # Extract psychiatric content using keywords
            extracted_text, extraction_stats = self.extract_psychiatric_content(clinical_text)

            # Generate prediction on extracted content (single inference)
            predicted_codes = self.generate_single_prediction(extracted_text)

            # Update stats
            extraction_stats.update({
                "text_handling": "keyword_extraction_only",
            })

            return {
                "predicted_codes": predicted_codes,
                "success": True,
                "extraction_stats": extraction_stats
            }
        except Exception as e:
            print(f"Error: {e}", flush=True)
            return {
                "predicted_codes": [],
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
        print(f"Evaluating: Strategy 6f - KEYWORD-AUGMENTED CHAIN-OF-THOUGHT")
        print(f"Text Handling: Keyword extraction only (no chunking)")
        print(f"Samples: {self.config['max_samples']}")
        print(f"{'='*80}\n", flush=True)

        test_df = self.load_test_data()
        results = []
        total_tp = total_fp = total_fn = 0
        successful = 0

        # Statistics for keyword extraction
        extraction_methods = {}

        start_time = datetime.now()

        for idx, (_, row) in enumerate(test_df.iterrows()):
            if (idx + 1) % 50 == 0:
                elapsed = (datetime.now() - start_time).total_seconds()
                rate = (idx + 1) / elapsed
                remaining = (len(test_df) - (idx + 1)) / rate if rate > 0 else 0
                print(f"Progress: {idx+1}/{len(test_df)} ({(idx+1)/len(test_df)*100:.1f}%) - ETA: {remaining/60:.1f}min", flush=True)

            actual_codes = self.parse_actual_codes(row['f_codes_str'])
            prediction = self.generate_prediction(row['text'])

            if prediction["success"]:
                successful += 1
                stats = prediction.get("extraction_stats", {})
                # Track extraction methods
                method = stats.get("extraction_method", "unknown")
                extraction_methods[method] = extraction_methods.get(method, 0) + 1

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
        # This iterates over unique codes and averages their F1 scores
        code_f1_scores = [m['f1'] for m in per_code_metrics.values()]
        code_precision_scores = [m['precision'] for m in per_code_metrics.values()]
        code_recall_scores = [m['recall'] for m in per_code_metrics.values()]

        macro_precision = sum(code_precision_scores) / len(code_precision_scores) if code_precision_scores else 0.0
        macro_recall = sum(code_recall_scores) / len(code_recall_scores) if code_recall_scores else 0.0
        macro_f1 = sum(code_f1_scores) / len(code_f1_scores) if code_f1_scores else 0.0

        # Also keep sample-averaged F1 for reference (different from standard Macro F1)
        sample_avg_f1 = sum(r["f1"] for r in results) / len(results) if results else 0.0

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
            "extraction_methods": extraction_methods,
            "evaluation_time_seconds": total_evaluation_time,
            "per_code_metrics": per_code_metrics
        }

        print(f"\n{'='*80}")
        print(f"Strategy 6f: KEYWORD-AUGMENTED CHAIN-OF-THOUGHT - RESULTS")
        print(f"{'='*80}")
        print(f"Micro F1: {overall_metrics['micro_f1']:.4f}")
        print(f"Macro F1 (per-code): {overall_metrics['macro_f1']:.4f} (over {len(per_code_metrics)} unique codes)")
        print(f"Sample-Avg F1: {overall_metrics['sample_avg_f1']:.4f}")
        print(f"Precision: {overall_metrics['micro_precision']:.4f}")
        print(f"Recall: {overall_metrics['micro_recall']:.4f}")
        print(f"Perfect matches: {overall_metrics['perfect_matches']}/{len(results)} ({overall_metrics['perfect_match_rate']:.1%})")
        print(f"\nExtraction Methods:")
        for method, count in extraction_methods.items():
            print(f"  {method}: {count}/{len(results)} ({count/len(results)*100:.1f}%)")
        print(f"\nEvaluation Time: {total_evaluation_time/60:.1f} minutes ({total_evaluation_time:.0f} seconds)")
        print(f"{'='*80}\n", flush=True)

        return results, overall_metrics

    def save_results(self, results: List[Dict], metrics: Dict[str, float]):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M')

        evaluation_report = {
            "evaluation_info": {
                "timestamp": datetime.now().isoformat(),
                "strategy": "6f_KeywordAugmentedCoT",
                "strategy_description": "Keyword-augmented CoT with F-code mappings",
                "text_handling": "keyword_extraction_only",
                "preprocessing": True,
                "cot_reasoning": True,
                "fcode_mappings": True,
                "key_distinctions": True,
                "ambiguous_codes": True,
                "psych_keywords_count": len(PSYCH_KEYWORDS),
                "checkpoint": self.config['checkpoint_path'],
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
                "num_chunks": 1,  # Keyword extraction always uses single inference
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
    print("EVALUATION STRATEGY 6f: KEYWORD-AUGMENTED CHAIN-OF-THOUGHT")
    print("Text Handling: Keyword extraction only (no chunking)")
    print("Preprocessing: PSYCH_KEYWORDS extraction | CoT: YES | Key Distinctions: YES")
    print("=" * 80)

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

    evaluator = KeywordAugmentedCoTEvaluator(CONFIG)

    try:
        evaluator.load_model()
        results, metrics = evaluator.evaluate()
        evaluator.save_results(results, metrics)

        print("\n" + "=" * 80)
        print("FINAL RESULTS - Strategy 6f: KEYWORD-AUGMENTED CHAIN-OF-THOUGHT")
        print("=" * 80)
        print(f"Micro F1:        {metrics['micro_f1']:.4f}")
        print(f"Macro F1:        {metrics['macro_f1']:.4f}")
        print(f"Perfect Match:   {metrics['perfect_match_rate']:.1%}")
        print(f"\nExtraction Methods:")
        for method, count in metrics.get('extraction_methods', {}).items():
            pct = count / metrics['total_samples'] * 100 if metrics['total_samples'] > 0 else 0
            print(f"  {method}: {pct:.1f}%")
        print("=" * 80 + "\n")

        return 0
    except Exception as e:
        print(f"\nError: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
