# Psychiatric F-Code Classification from Clinical Notes

Code for fine-tuning a large language model (GPT-OSS 20B, LoRA) to assign psychiatric ICD-10 F-codes (F00–F99) to MIMIC-IV discharge summaries, with chunking for documents longer than the context window, and for the evaluation and analysis scripts that accompany the study.

## Requirements

- NVIDIA GPU with at least 32 GB VRAM (developed on an RTX 5090; other GPUs are untested)
- 32 GB system RAM
- Python 3.10 or later
- MIMIC-IV v3.1 and MIMIC-IV-Note v2.2, obtained through [PhysioNet credentialed access](https://physionet.org/content/mimiciv/)

## Installation

```bash
git clone https://github.com/anthonysshin/clinical-notes-analysis.git
cd clinical-notes-analysis

conda create -n unsloth-blackwell python=3.12 -y
conda activate unsloth-blackwell

# RTX 50-series (Blackwell) GPUs require the CUDA 12.8 nightly build of PyTorch
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
pip install -r requirements.txt
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

Verify the installation:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "from unsloth import FastLanguageModel; print('Unsloth OK')"
```

The PyTorch version should include the `+cu128` suffix.

## Data setup

Place the MIMIC-IV files in `raw_data/`, then create the working directories:

```bash
mkdir -p data models/checkpoints outputs logs raw_data
```

All paths and hyperparameters are set in `config.py`.

## Usage

Run the scripts in numeric order. Each script reads the outputs of the previous steps.

| Step | Script | Output |
|------|--------|--------|
| 1 | `1_DataPrep.py` | Train, validation, and test splits in `data/` |
| 2 | `2_PrepareInstructionData.py` | Chunked instruction data in `data/` |
| 3 | `3_EDA.py` | Figures in `outputs/3_EDA/` |
| 4 | `4_FineTuning.py` | LoRA checkpoints in `models/checkpoints/` |
| 5 | `5_FindBestCheckpoint.py` | `best_checkpoint.json` |
| 6a–6g2 | Evaluation scripts (see below) | `outputs/<script>/` |
| 7 | `7_MultiEvalAnalysis.py` | Cross-strategy comparison and statistical tests |
| 8 | `8_BestEvalAnalysis.py` | Detailed analysis of the best strategy |
| 9 | `9_ErrorAnalysis.py` | Error patterns and per-code performance |
| 10 | `10_XRLAT_Encoder_Baseline.py` | Encoder baseline results |
| 11 | `11_PatientOverlapAnalysis.py` | Patient-level overlap and patient-disjoint re-scoring |

Training takes many hours on a single GPU. Each fine-tuned evaluation strategy takes roughly 15–45 minutes, and each base-model strategy takes about 2.5 hours.

### Evaluation strategies

| Script | Text handling | Prompt |
|--------|---------------|--------|
| `6a_Eval_ZeroShotBaseline.py` | Chunking | Zero-shot |
| `6b_Eval_FewShotExemplar.py` | Chunking | Few-shot |
| `6c_Eval_RuleConstrained.py` | Chunking | Rule-constrained |
| `6d_Eval_ChainOfThought.py` | Chunking | Chain-of-thought |
| `6e_Eval_KeywordAugmented.py` | Keyword extraction | Zero-shot |
| `6f_Eval_KeywordAugmentedCoT.py` | Keyword extraction | Chain-of-thought |
| `6g1_Eval_BaseModel_ZeroShot.py` | Chunking | Zero-shot, base model (no fine-tuning) |
| `6g2_Eval_BaseModel_CoT.py` | Chunking | Chain-of-thought, base model (no fine-tuning) |

### Keyword list construction

The keyword list used by strategies 6e and 6f is built by three scripts:

| Script | Purpose |
|--------|---------|
| `12_PsychKeywordsReproducibility.py` | F-code frequency ranking and split-stability check |
| `13_PsychKeywordsConstruction.py` | Builds the keyword list from the training set |
| `14_PsychKeywordsTestDerived.py` | Builds the same list from the test set, as a leakage check |

Running `13_PsychKeywordsConstruction.py` writes `outputs/13_PsychKeywordsConstruction/psych_keywords_final.py`. The `--reproduce-from-scratch` option rebuilds the list from the candidate files in `reference_data/`, which is useful for checking the procedure on a new machine.

The candidate medication and condition lists in `reference_data/` were compiled with an AI assistant. The exact prompts are given in the Supplementary Information.

## Output format

Each model response is a JSON array of F-codes, for example `["F32.9", "F17.210", "F41.9"]`.

## Troubleshooting

**Out-of-memory during model loading (exit code 137).** On WSL2, set the memory limit in `C:\Users\<username>\.wslconfig`:

```ini
[wsl2]
memory=28GB
swap=8GB
```

Then run `wsl --shutdown` and restart WSL2.

**CUDA "no kernel image is available" or `sm_120` errors.** The stable PyTorch release does not support RTX 50-series GPUs. Install the cu128 nightly build as shown under Installation. Do not use cu121 or cu124 builds for these GPUs.

**Out-of-memory during training.** Lower `TRAIN_BATCH_SIZE` in `config.py` and raise `GRADIENT_ACCUMULATION_STEPS` to keep the effective batch size the same.

**Check GPU status:**

```bash
nvidia-smi
```

## External validation

External validation used proprietary data from the University of Illinois Chicago and cannot be shared. Validation scripts are available on reasonable request to the corresponding author.

## License

Apache License 2.0. See [LICENSE](LICENSE). MIMIC-IV data are governed by the PhysioNet data use agreement and are not included in this repository.

## Acknowledgments

- [MIMIC-IV](https://physionet.org/content/mimiciv/) (PhysioNet)
- [Unsloth](https://github.com/unslothai/unsloth)
- GPT-OSS 20B base model
