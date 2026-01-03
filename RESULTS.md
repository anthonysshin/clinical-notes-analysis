# Experimental Results

Summary of evaluation results from the psychiatric F-code classification study.

## Model Configuration

- **Base Model**: GPT-OSS 20B (4-bit quantized, ~12GB VRAM)
- **Fine-tuning**: LoRA (Rank=16, Alpha=32, 0.04% trainable parameters)
- **Training**: 5 epochs, 25,000 steps, ~37 hours on RTX 5090
- **Best Checkpoint**: checkpoint-21500 (Validation Micro F1: 0.6700)

## Evaluation Results

Evaluated on 1,000 test samples from MIMIC-IV discharge summaries.

### Strategy Comparison

| Strategy | Fine-tuned | Micro P | Micro R | Micro F1 | Macro F1 | Perfect Match | Time |
|----------|------------|---------|---------|----------|----------|---------------|------|
| Zero-Shot (Base) | No | 0.139 | 0.290 | 0.188 | 0.011 | 0.0% | 143 min |
| CoT (Base) | No | 0.266 | 0.192 | 0.223 | 0.044 | 5.7% | 147 min |
| Zero-Shot | Yes | 0.542 | 0.674 | 0.601 | 0.201 | 33.3% | 40 min |
| Few-Shot | Yes | 0.565 | 0.686 | 0.620 | 0.225 | 35.1% | 42 min |
| Rule-Constrained | Yes | 0.549 | 0.677 | 0.606 | 0.199 | 34.8% | 42 min |
| Chain-of-Thought | Yes | 0.594 | 0.702 | 0.644 | 0.226 | 38.1% | 44 min |
| Keyword-Augmented | Yes | 0.628 | 0.621 | 0.625 | 0.167 | 43.8% | 15 min |
| **Keyword + CoT** | **Yes** | **0.694** | **0.657** | **0.675** | **0.211** | **46.9%** | **17 min** |

### Key Findings

1. **Best Strategy**: Keyword + CoT achieves the highest Micro F1 (0.675) and Perfect Match rate (46.9%)

2. **Fine-tuning Impact**:
   - Improvement over base model: +0.487 Micro F1 (+260%)
   - Base model: 0.188 vs Fine-tuned best: 0.675

3. **Statistical Significance**:
   - Keyword + CoT vs Chain-of-Thought: t = 2.505, **p = 0.012** (statistically significant at p < 0.05)
   - Effect size (Cohen's d): 0.079 (negligible practical difference)
   - Note: Large sample size (n=1000) enables detection of small differences

4. **Efficiency**:
   - Keyword-based strategies are 2-3x faster than chunking-based strategies
   - Best strategy (Keyword + CoT): 17 min vs 44 min (Chain-of-Thought with chunking)

### Per-Code Performance

**Top Performing F-Codes** (by F1, minimum 5 occurrences):
| F-Code | Description | F1 | Support |
|--------|-------------|-----|---------|
| F20.0 | Paranoid schizophrenia | 0.933 | 7 |
| F32.9 | Major depressive disorder, unspecified | 0.872 | 437 |
| F01.50 | Vascular dementia | 0.870 | 13 |
| F31.9 | Bipolar disorder, unspecified | 0.847 | 40 |
| F20.9 | Schizophrenia, unspecified | 0.818 | 11 |

**Challenging F-Codes** (lowest F1, minimum 5 occurrences):
| F-Code | Description | F1 | Support |
|--------|-------------|-----|---------|
| F17.200 | Nicotine dependence, unspecified | 0.000 | 17 |
| F17.290 | Nicotine dependence, other tobacco product | 0.000 | 9 |
| F10.11 | Alcohol abuse, in remission | 0.000 | 8 |
| F11.90 | Opioid use, unspecified | 0.000 | 5 |
| F19.11 | Other psychoactive substance abuse | 0.000 | 5 |

### Error Analysis

**Error Distribution**:
- Perfect Match: 469 samples (46.9%)
- Partial Match: 351 samples (35.1%)
- Complete Miss: 180 samples (18.0%)

**Common Confusion Patterns**:
1. F17.210 (Nicotine dependence, cigarettes) confused with F41.9 (Anxiety disorder)
2. F17.200 confused with F17.210 (nicotine dependence subtypes)
3. F32.9 (Depression) confused with F41.9 (Anxiety) - comorbidity overlap

## Reproducibility

All results can be reproduced by running the evaluation scripts with the same random seed (42) and checkpoint (checkpoint-21500).

```bash
# Run all evaluations
python 6a_Eval_ZeroShotBaseline.py
python 6b_Eval_FewShotExemplar.py
python 6c_Eval_RuleConstrained.py
python 6d_Eval_ChainOfThought.py
python 6e_Eval_KeywordAugmented.py
python 6f_Eval_KeywordAugmentedCoT.py
python 6g1_Eval_BaseModel_ZeroShot.py
python 6g2_Eval_BaseModel_CoT.py

# Run analysis
python 7_MultiEvalAnalysis.py
python 8_BestEvalAnalysis.py --results-dir outputs/6f_KeywordAugmentedCoT
python 9_ErrorAnalysis.py
```

## Hardware

- **GPU**: NVIDIA RTX 5090 (32GB VRAM)
- **System RAM**: 32GB
- **Platform**: WSL2 on Windows 11
- **CUDA**: 12.8, PyTorch 2.9.1

## Citation

See README.md for citation information.
