# Psychiatric F-Code Classification from Clinical Notes

Fine-tuned GPT-OSS 20B for multi-label psychiatric diagnosis code (ICD-10 F00-F99) extraction from MIMIC-IV clinical discharge summaries.

## Overview

This project implements a chunking-based approach to handle long clinical documents that exceed typical LLM context windows. Documents are split into overlapping chunks, each chunk is processed independently, and predictions are aggregated via set union.

### Key Features

- **Chunking Strategy**: 2800-token chunks with 200-token overlap (handles 66.8% of documents that exceed single-chunk capacity)
- **Union Aggregation**: Combines F-code predictions from all chunks
- **LoRA Fine-tuning**: Efficient 4-bit quantized training with 0.04% trainable parameters
- **8 Evaluation Strategies**: Systematic comparison of prompting approaches

## Requirements

### Hardware
- GPU: NVIDIA H100/A100 (80GB+ VRAM recommended) or RTX 4090 (24GB)
- RAM: 96GB system memory
- Storage: ~100GB for model checkpoints and data

### Software
- Python 3.10+
- PyTorch 2.0+ with CUDA
- Unsloth (efficient fine-tuning)
- Transformers 4.40+

### Data
- MIMIC-IV v3.1 (PhysioNet credentialed access required)
- MIMIC-IV-Note v2.2 (discharge summaries)

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install torch transformers datasets pandas numpy scikit-learn
pip install unsloth
pip install matplotlib seaborn scipy
```

## Project Structure

```
CNA_Chunking_1222_2025/
├── config.py                       # Centralized configuration
├── 1_DataPrep.py                   # MIMIC-IV data preparation
├── 2_PrepareInstructionData.py     # Instruction format + chunking
├── 3_EDA.py                        # Exploratory data analysis
├── 4_FineTuning.py                 # LoRA fine-tuning
├── 5_FindBestCheckpoint.py         # Checkpoint selection
├── 6a_Eval_ZeroShotBaseline.py     # Zero-shot evaluation
├── 6b_Eval_FewShotExemplar.py      # Few-shot evaluation
├── 6c_Eval_RuleConstrained.py      # Rule-constrained evaluation
├── 6d_Eval_ChainOfThought.py       # Chain-of-thought evaluation
├── 6e_Eval_KeywordAugmented.py     # Keyword preprocessing + zero-shot
├── 6f_Eval_KeywordAugmentedCoT.py  # Keyword preprocessing + CoT
├── 6g1_Eval_BaseModel_ZeroShot.py  # Base model zero-shot
├── 6g2_Eval_BaseModel_CoT.py       # Base model CoT
├── 8_MultiEvalAnalysis.py          # Comparative analysis
├── 9_ErrorAnalysis.py              # Error pattern analysis
├── 10_AdditionalFigures.py         # Additional analysis figures
├── data/                           # Processed datasets
├── models/                         # Fine-tuned checkpoints
├── outputs/                        # Evaluation results
└── slurm_scripts/                  # HPC job scripts
```

## Pipeline

### Step 1: Data Preparation
```bash
python 1_DataPrep.py
```
- Loads MIMIC-IV discharge notes and ICD diagnoses
- Filters to psychiatric F-codes (F00-F99)
- Creates stratified train/validation/test splits (70/15/15)

### Step 2: Prepare Instruction Data
```bash
python 2_PrepareInstructionData.py
```
- Chunks clinical notes (2800 tokens, 200 overlap)
- Creates instruction-tuning format with CoT system prompt
- Each chunk inherits parent document's F-code labels

### Step 3: Exploratory Data Analysis
```bash
python 3_EDA.py
```
- Generates 11 analysis figures
- F-code distribution, co-occurrence patterns
- Text length and chunking statistics

### Step 4: Fine-Tuning
```bash
python 4_FineTuning.py
# Or via SLURM:
sbatch slurm_scripts/4_finetune.slurm
```
- Fine-tunes GPT-OSS 20B with LoRA adapters
- 4-bit quantization, 5 epochs, 20K training samples
- Saves checkpoints every 500 steps

### Step 5: Checkpoint Selection
```bash
python 5_FindBestCheckpoint.py
```
- Evaluates all checkpoints on 200 validation samples
- Selects best by Micro F1 score
- Saves to `best_checkpoint.json`

### Step 6: Evaluation (8 Strategies)
```bash
# Run all evaluations
sbatch slurm_scripts/run_evals_only.slurm

# Or individually:
python 6a_Eval_ZeroShotBaseline.py
python 6b_Eval_FewShotExemplar.py
python 6c_Eval_RuleConstrained.py
python 6d_Eval_ChainOfThought.py
python 6e_Eval_KeywordAugmented.py
python 6f_Eval_KeywordAugmentedCoT.py
python 6g1_Eval_BaseModel_ZeroShot.py
python 6g2_Eval_BaseModel_CoT.py
```

### Step 7: Analysis
```bash
python 8_MultiEvalAnalysis.py
python 9_ErrorAnalysis.py
python 10_PublicationFigures.py
```

## Evaluation Strategies

| Strategy | Preprocessing | Prompt Style | Description |
|----------|---------------|--------------|-------------|
| **6a** Zero-Shot | None | Zero-shot | Baseline fine-tuned model |
| **6b** Few-Shot | None | Few-shot | Includes example predictions |
| **6c** Rule-Constrained | None | With mappings | F-code mapping guidance |
| **6d** Chain-of-Thought | None | CoT | Step-by-step reasoning (matches training) |
| **6e** Keyword-Augmented | PSYCH_KEYWORDS | Zero-shot | Keyword extraction preprocessing |
| **6f** Keyword + CoT | PSYCH_KEYWORDS | CoT | Combined approach |
| **6g1** Base Zero-Shot | None | Zero-shot | Unfine-tuned baseline |
| **6g2** Base CoT | None | CoT | Unfine-tuned + CoT |

### Prompt Styles

**Zero-Shot** (6a, 6e, 6g1):
```
Extract all psychiatric F-codes (F00-F99) from the clinical text.

Respond with only the F-codes as a JSON array.

Format response as:
["F32.9", "F17.210"]
```

**Chain-of-Thought** (6d, 6f, 6g2, Training):
```
Analyze the clinical text step by step to extract psychiatric F-codes (F00-F99).

STEP-BY-STEP PROCESS:
Step 1: IDENTIFY psychiatric keywords and conditions
Step 2: MAP identified conditions to appropriate F-codes
Step 3: SELECT most confident codes (maximum 5)
Step 4: OUTPUT as JSON array only
["F32.9", "F17.210", "F41.9"]
```

## Configuration

Key parameters in `config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `RANDOM_SEED` | 42 | Reproducibility seed |
| `BASE_MODEL_NAME` | unsloth/gpt-oss-20b-unsloth-bnb-4bit | Base model |
| `MAX_SEQ_LENGTH` | 4096 | Model context window |
| `CHUNK_SIZE` | 2800 | Input chunk size (tokens) |
| `CHUNK_OVERLAP` | 200 | Overlap between chunks |
| `MAX_NEW_TOKENS` | 100 | Max generation length |
| `LORA_R` | 16 | LoRA rank |
| `LORA_ALPHA` | 32 | LoRA scaling factor |
| `NUM_EPOCHS` | 5 | Training epochs |
| `MAX_TRAIN_SAMPLES` | 20000 | Training sample limit |

## Dataset Statistics

| Split | Documents | Chunked Samples |
|-------|-----------|-----------------|
| Train | 39,848 (70%) | 70,100 |
| Validation | 8,539 (15%) | - |
| Test | 8,539 (15%) | - |

- **Unique F-codes**: 466
- **Avg F-codes per sample**: 1.76
- **Avg chunks per document**: 1.76
- **Multi-chunk documents**: 66.8%

## Output Format

Predictions are simple JSON arrays:
```json
["F32.9", "F17.210", "F41.9"]
```

## Metrics

- **Micro F1**: Overall performance (primary metric)
- **Macro F1**: Average per-code performance
- **Sample-Avg F1**: Average per-sample performance
- **Perfect Match Rate**: Exact match accuracy

## Known Limitations

### Label Imbalance: F17.200 vs F17.210

The model exhibits zero performance (Precision=0, Recall=0, F1=0) for **F17.200** (Nicotine dependence, unspecified) due to severe class imbalance with the semantically similar **F17.210** (Nicotine dependence, cigarettes).

**Data Distribution:**

| Code | Definition | Training | Test | Ratio |
|------|------------|----------|------|-------|
| F17.210 | Nicotine dependence, cigarettes, uncomplicated | 13,848 | 1,721 | 10x |
| F17.200 | Nicotine dependence, unspecified, uncomplicated | 1,382 | 172 | 1x |

**Model Behavior (17 test cases with F17.200 ground truth):**
- 12 cases (71%): Model predicted F17.210 instead
- 5 cases (29%): Model predicted other codes only
- 0 cases (0%): Model correctly predicted F17.200

**Root Cause:**
1. F17.210 is 10x more frequent than F17.200 in both training and test data
2. Both codes are semantically nearly identical (cigarette vs. unspecified nicotine dependence)
3. The model learned to always predict the dominant class

This is expected behavior for discriminative models on imbalanced multi-label data and represents a known limitation of the approach.

## Citation

```bibtex
@article{psychiatric_fcode_2025,
  title={Psychiatric F-Code Classification from Clinical Notes using Fine-tuned LLMs with Chunking},
  author={[Authors]},
  year={2025}
}
```

## License

Research use only. MIMIC-IV data requires PhysioNet credentialed access and appropriate data use agreements.

## Acknowledgments

- MIMIC-IV dataset (PhysioNet)
- Unsloth for efficient fine-tuning
- GPT-OSS 20B base model
