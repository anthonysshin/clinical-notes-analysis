# Psychiatric F-Code Classification from Clinical Notes

Fine-tuned GPT-OSS 20B for multi-label psychiatric diagnosis code (ICD-10 F00-F99) extraction from MIMIC-IV clinical discharge summaries.

## Overview

This project implements a chunking-based approach to handle long clinical documents that exceed typical LLM context windows. Documents are split into overlapping chunks, each chunk is processed independently, and predictions are aggregated via set union.

### Key Features

- **Chunking Strategy**: 2800-token chunks with 200-token overlap (handles 66.8% of documents requiring multiple chunks)
- **LoRA Fine-tuning**: Efficient 4-bit quantized training with 0.04% trainable parameters
- **6 Evaluation Strategies**: Systematic comparison of prompting approaches (plus 2 base model baselines)
- **Personal Workstation Support**: Optimized for consumer GPU (tested on RTX 5090)

## Requirements

### Hardware

**Tested Configuration:**
- GPU: NVIDIA RTX 5090 (32GB VRAM)
- System RAM: 32GB
- Storage: 100GB

> **Note:** This project was developed and tested on RTX 5090. Compatibility with other GPUs (e.g., RTX 4090 24GB) has not been verified.

### Software

- Python 3.10+
- PyTorch 2.0+ with CUDA 12.1+
- Unsloth (efficient fine-tuning)
- Transformers 4.35+

### Data

- MIMIC-IV v3.1 ([PhysioNet credentialed access required](https://physionet.org/content/mimiciv/))
- MIMIC-IV-Note v2.2 (discharge summaries)

## Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/anthonysshin/clinical-notes-analysis.git
cd clinical-notes-analysis
```

### Step 2: WSL2 Configuration (Windows Users)

> **Critical for preventing out-of-memory errors during model loading.**

Create or edit `C:\Users\<username>\.wslconfig`:

```ini
[wsl2]
memory=28GB
swap=8GB
```

Restart WSL2:

```powershell
wsl --shutdown
```

### Step 3: Create Virtual Environment

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS/WSL

# Install PyTorch with CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt

# Install Unsloth for efficient fine-tuning
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### Step 4: Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "from unsloth import FastLanguageModel; print('Unsloth: OK')"
```

## Project Structure

```
clinical-notes-analysis/
├── config.py                       # Centralized configuration
├── requirements.txt                # Python dependencies
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
├── 7_SingleEvalAnalysis.py         # Single strategy analysis
├── 8_MultiEvalAnalysis.py          # Comparative analysis
├── 9_ErrorAnalysis.py              # Error pattern analysis
├── 10_AdditionalFigures.py         # Additional analysis figures
├── data/                           # Processed datasets (created by scripts)
├── models/                         # Fine-tuned checkpoints (created by scripts)
└── outputs/                        # Evaluation results (created by scripts)
```

## Pipeline

### Directory Setup

Create required directories before running:

```bash
mkdir -p data models/checkpoints outputs logs raw_data
```

Place MIMIC-IV data files in `raw_data/` directory.

### Phase 1: Data Preparation

```bash
# Step 1: Prepare MIMIC-IV data (filter F-codes, create splits)
python 1_DataPrep.py

# Step 2: Create chunked instruction data for training
python 2_PrepareInstructionData.py

# Step 3: Exploratory data analysis (generates 11 figures)
python 3_EDA.py
```

**Outputs:**
- `data/mimic_iv_{train,val,test}_data.csv`
- `data/{train,val,test}_chunked.json`
- `outputs/3_EDA/*.png`

### Phase 2: Fine-Tuning

```bash
# Step 4: Fine-tune GPT-OSS 20B with LoRA adapters
python 4_FineTuning.py

# Step 5: Find best checkpoint using validation set
python 5_FindBestCheckpoint.py
```

**Outputs:**
- `models/checkpoints/checkpoint-*/`
- `best_checkpoint.json`

> **Note:** Training time varies significantly based on hardware configuration.

### Phase 3: Evaluation

Run all evaluation strategies (6 fine-tuned + 2 base model baselines):

```bash
python 6a_Eval_ZeroShotBaseline.py
python 6b_Eval_FewShotExemplar.py
python 6c_Eval_RuleConstrained.py
python 6d_Eval_ChainOfThought.py
python 6e_Eval_KeywordAugmented.py
python 6f_Eval_KeywordAugmentedCoT.py
python 6g1_Eval_BaseModel_ZeroShot.py
python 6g2_Eval_BaseModel_CoT.py
```

> **Note:** Evaluation time varies based on hardware and the number of test samples (default: 1000).

### Phase 4: Analysis

```bash
python 7_SingleEvalAnalysis.py
python 8_MultiEvalAnalysis.py
python 9_ErrorAnalysis.py
python 10_AdditionalFigures.py
```

## Evaluation Strategies

| Strategy | Text Handling | Prompt Style | Description |
|----------|---------------|--------------|-------------|
| **6a** Zero-Shot | Chunking | Zero-shot | Baseline fine-tuned model |
| **6b** Few-Shot | Chunking | Few-shot | Includes example predictions |
| **6c** Rule-Constrained | Chunking | With mappings | F-code mapping guidance |
| **6d** Chain-of-Thought | Chunking | CoT | Step-by-step reasoning (matches training) |
| **6e** Keyword-Augmented | Keywords | Zero-shot | Keyword extraction preprocessing |
| **6f** Keyword + CoT | Keywords | CoT | Combined approach (best performance) |
| **6g1** Base Zero-Shot | Chunking | Zero-shot | Unfine-tuned baseline |
| **6g2** Base CoT | Chunking | CoT | Unfine-tuned + CoT |

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
| `TRAIN_BATCH_SIZE` | 1 | Per-device batch size (32GB VRAM) |
| `GRADIENT_ACCUMULATION_STEPS` | 4 | Effective batch size = 4 |

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

## Troubleshooting

### Out-of-Memory During Model Loading (Exit Code 137)

This occurs when system RAM is exhausted during model loading.

**Solution (WSL2 users):**

1. Create/edit `C:\Users\<username>\.wslconfig`:
   ```ini
   [wsl2]
   memory=28GB
   swap=8GB
   ```

2. Restart WSL2: `wsl --shutdown`

3. Verify: `free -h` should show ~28GB available

### CUDA Out-of-Memory During Training

**Solution:** Verify batch size configuration in `config.py`:

```python
TRAIN_BATCH_SIZE = 1  # Per-device batch size
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 4
```

### Module Not Found Errors

**Solution:** Ensure virtual environment is activated:

```bash
source venv/bin/activate
which python  # Should show: .../venv/bin/python
```

### Check GPU Status

```bash
nvidia-smi  # View GPU memory usage
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Citation

```bibtex
@article{mental_health_coding_2025,
  title={Automated ICD-10 Mental Health Diagnosis Coding from Clinical Notes using Domain-Adapted Large Language Models},
  author={[Authors]},
  year={2025}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.

**Note:** MIMIC-IV data requires PhysioNet credentialed access and appropriate data use agreements.

## Acknowledgments

- [MIMIC-IV dataset](https://physionet.org/content/mimiciv/) (PhysioNet)
- [Unsloth](https://github.com/unslothai/unsloth) for efficient fine-tuning
- GPT-OSS 20B base model
