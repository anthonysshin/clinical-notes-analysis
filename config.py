#!/usr/bin/env python3
"""
config.py - Centralized Configuration for Clinical Note Analysis Project

This module provides centralized configuration settings for reproducibility
and consistency across all scripts in the project.

================================================================================
VIRTUAL ENVIRONMENT SETUP (For New Users)
================================================================================

1. Create a new virtual environment:
    python -m venv venv

2. Activate the virtual environment:
    # Linux/macOS:
    source venv/bin/activate

    # Windows:
    venv\\Scripts\\activate

3. Install PyTorch with CUDA support (adjust for your CUDA version):
    # For CUDA 12.1:
    pip install torch --index-url https://download.pytorch.org/whl/cu121

    # For CUDA 11.8:
    pip install torch --index-url https://download.pytorch.org/whl/cu118

4. Install project dependencies:
    pip install -r requirements.txt

5. Install Unsloth for efficient fine-tuning:
    pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

   Alternative: Install from PyPI (may be older version):
    pip install unsloth

6. Verify installation:
    python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
    python -c "from unsloth import FastLanguageModel; print('Unsloth OK')"

================================================================================

Usage:
    from config import RANDOM_SEED, set_all_seeds, PROJECT_CONFIG

    # Set all random seeds at the start of any script
    set_all_seeds()

    # Or access individual settings
    seed = RANDOM_SEED

Author: Clinical Note Analysis Study
"""

import os
import random
from typing import Optional

import numpy as np

# Try to import torch (may not be available in all environments)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# REPRODUCIBILITY SETTINGS
# =============================================================================

# Master random seed for all operations
# This seed is used across: data splitting, sampling, model training, evaluation
RANDOM_SEED = 42


def set_all_seeds(seed: Optional[int] = None):
    """
    Set random seeds for all libraries to ensure reproducibility.

    Args:
        seed: Random seed value. If None, uses RANDOM_SEED from config.

    Usage:
        from config import set_all_seeds
        set_all_seeds()  # Call at the start of each script
    """
    if seed is None:
        seed = RANDOM_SEED

    # Python's built-in random
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # Environment variable for hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)

    # PyTorch (if available)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU

        # For deterministic behavior (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    print(f"[config] Random seeds set to {seed} for reproducibility")


# =============================================================================
# PROJECT PATHS
# =============================================================================

# Base project directory
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data directories
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
OUTPUT_DIR = os.path.join(PROJECT_DIR, 'outputs')
LOGS_DIR = os.path.join(PROJECT_DIR, 'logs')

# Raw MIMIC-IV data path (within project directory)
# This can be overridden via command-line argument in 1_DataPrep.py
MIMIC_RAW_DATA_DIR = os.path.join(PROJECT_DIR, 'raw_data')

# Data files
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'mimic_iv_train_data.csv')
VAL_DATA_PATH = os.path.join(DATA_DIR, 'mimic_iv_val_data.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'mimic_iv_test_data.csv')

# Chunked instruction data files (for chunking approach)
TRAIN_CHUNKED_PATH = os.path.join(DATA_DIR, 'train_chunked.json')
VAL_CHUNKED_PATH = os.path.join(DATA_DIR, 'val_chunked.json')
TEST_CHUNKED_PATH = os.path.join(DATA_DIR, 'test_chunked.json')

# Legacy truncated paths (kept for compatibility)
TRAIN_TRUNCATED_PATH = os.path.join(DATA_DIR, 'train_truncated.json')
VAL_TRUNCATED_PATH = os.path.join(DATA_DIR, 'val_truncated.json')
TEST_TRUNCATED_PATH = os.path.join(DATA_DIR, 'test_truncated.json')

# Model directories
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
CHECKPOINT_DIR = os.path.join(MODELS_DIR, 'checkpoints')
FINAL_MODEL_DIR = os.path.join(MODELS_DIR, 'final_model')

# =============================================================================
# OUTPUT DIRECTORIES (Numbered to match script names)
# =============================================================================
# All outputs go under outputs/ with numbered prefixes for easy identification
# Format: outputs/{script_number}_{description}/

# Phase 1: Data Preparation & EDA
OUTPUT_DIR_3_EDA = os.path.join(OUTPUT_DIR, '3_EDA')

# Phase 3: Checkpoint Selection
OUTPUT_DIR_5_CHECKPOINT = os.path.join(OUTPUT_DIR, '5_CheckpointSelection')

# Phase 4: Evaluation (6a-6g2)
OUTPUT_DIR_6A = os.path.join(OUTPUT_DIR, '6a_ZeroShotBaseline')
OUTPUT_DIR_6B = os.path.join(OUTPUT_DIR, '6b_FewShotExemplar')
OUTPUT_DIR_6C = os.path.join(OUTPUT_DIR, '6c_RuleConstrained')
OUTPUT_DIR_6D = os.path.join(OUTPUT_DIR, '6d_ChainOfThought')
OUTPUT_DIR_6E = os.path.join(OUTPUT_DIR, '6e_KeywordAugmented')
OUTPUT_DIR_6F = os.path.join(OUTPUT_DIR, '6f_KeywordAugmentedCoT')
OUTPUT_DIR_6G1 = os.path.join(OUTPUT_DIR, '6g1_BaseModel_ZeroShot')
OUTPUT_DIR_6G2 = os.path.join(OUTPUT_DIR, '6g2_BaseModel_CoT')

# Phase 5: Analysis
OUTPUT_DIR_7_MULTI = os.path.join(OUTPUT_DIR, '7_MultiEvalAnalysis')
OUTPUT_DIR_8_BEST = os.path.join(OUTPUT_DIR, '8_BestEvalAnalysis')
OUTPUT_DIR_9_ERROR = os.path.join(OUTPUT_DIR, '9_ErrorAnalysis')

# Legacy aliases for backward compatibility (will be removed in future)
EVAL_OUTPUT_DIR_6A = OUTPUT_DIR_6A
EVAL_OUTPUT_DIR_6B = OUTPUT_DIR_6B
EVAL_OUTPUT_DIR_6C = OUTPUT_DIR_6C
EVAL_OUTPUT_DIR_6D = OUTPUT_DIR_6D
EVAL_OUTPUT_DIR_6E = OUTPUT_DIR_6E
EVAL_OUTPUT_DIR_6F = OUTPUT_DIR_6F
EVAL_OUTPUT_DIR_6G1 = OUTPUT_DIR_6G1
EVAL_OUTPUT_DIR_6G2 = OUTPUT_DIR_6G2


# =============================================================================
# MODEL SETTINGS
# =============================================================================

# Base model (before fine-tuning)
# This is the 4-bit quantized version used for fine-tuning
# Must match the model used in 4_FineTuning.py for fair comparison
BASE_MODEL_NAME = 'unsloth/gpt-oss-20b-unsloth-bnb-4bit'

# Model context window
# GPT-OSS 20B supports up to 4096 tokens per sequence
MAX_SEQ_LENGTH = 4096

# 4-bit quantization reduces memory from ~40GB to ~12GB VRAM
# Enables running 20B parameter model on a single 32GB GPU workstation
LOAD_IN_4BIT = True


# =============================================================================
# LORA FINE-TUNING HYPERPARAMETERS
# =============================================================================

# LoRA rank: Controls adapter capacity (higher = more parameters, more expressive)
# 16 is standard for instruction tuning; balances capacity vs. efficiency
LORA_R = 16

# LoRA alpha: Scaling factor for LoRA weights
# alpha/r ratio of 2 (32/16) is standard; higher ratio = stronger adaptation
LORA_ALPHA = 32

# LoRA dropout: Regularization during training (0 = no dropout)
LORA_DROPOUT = 0


# =============================================================================
# TEXT CHUNKING PARAMETERS
# =============================================================================
# Chunking Strategy: Split long documents into overlapping chunks
# Each chunk is processed independently, predictions combined using union strategy
#
# Token Allocation per chunk (max_seq_length = 4096):
#   System Prompt (CoT):    ~550 tokens
#   User Template:          ~30 tokens
#   Clinical Note Chunk:    2800 tokens
#   Output (JSON array):    100 tokens
#   Buffer:                 ~616 tokens
#   TOTAL:                  4096 tokens
#
# Chunking provides higher recall by processing entire document
# Union strategy combines all predictions from all chunks

# Chunk size in tokens
CHUNK_SIZE = 2800

# Overlap between chunks to avoid missing information at boundaries
CHUNK_OVERLAP = 200

# Text handling approach: "chunking" or "truncation"
TEXT_HANDLING = "chunking"

# Maximum tokens for single chunk (used when document fits in one chunk)
MAX_INPUT_TOKENS = 2800


# =============================================================================
# GENERATION PARAMETERS
# =============================================================================

# Maximum tokens to generate for F-code predictions
# 100 tokens is sufficient for JSON array output: ["F32.9", "F17.210", ...]
MAX_NEW_TOKENS = 100

# Temperature for text generation
# 0.0 = deterministic (greedy decoding) for reproducible predictions
TEMPERATURE = 0.0


# =============================================================================
# TRAINING SETTINGS
# =============================================================================

# Number of training epochs
NUM_EPOCHS = 5

# Maximum training samples for fine-tuning
MAX_TRAIN_SAMPLES = 20000

# Training batch configuration (optimized for 32GB VRAM workstation)
TRAIN_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size = 1 * 4 = 4

# Learning rate
LEARNING_RATE = 2e-4

# Warmup steps
WARMUP_STEPS = 100

# Checkpoint saving
SAVE_STEPS = 500
SAVE_TOTAL_LIMIT = None  # Keep all checkpoints for selection
MIN_CHECKPOINT_STEP = 15000  # Skip early checkpoints during selection (evaluate from step 15000+)


# =============================================================================
# EVALUATION SETTINGS
# =============================================================================

# Default number of test samples for evaluation
# VALIDATION: Set to 100 for quick test, change to 1000 for full run
DEFAULT_MAX_SAMPLES = 1000  # Full evaluation sample size

# Validation samples for checkpoint selection
VAL_SAMPLES = 200


# =============================================================================
# VISUALIZATION COLOR PALETTE
# =============================================================================
# Professional pastel color palette for journal publication
# - Colorblind-friendly (distinguishable in deuteranopia/protanopia)
# - Prints well in grayscale (varying luminance)
# - Consistent across all figures

# Strategy colors: Ordered by performance (base models in grey, fine-tuned in pastels)
STRATEGY_COLORS = {
    # Base models (no fine-tuning) - Grey tones
    '6g1_BaseModel_ZeroShot': '#B0B0B0',  # Light grey
    '6g2_BaseModel_CoT': '#808080',        # Medium grey

    # Fine-tuned models - Pastel spectrum (ordered by typical performance)
    '6a_ZeroShotBaseline': '#F4A5A5',      # Pastel red/coral
    '6b_FewShotExemplar': '#F4C79A',       # Pastel orange/peach
    '6c_RuleConstrained': '#F4E59A',       # Pastel yellow
    '6d_ChainOfThought': '#A5C8E4',        # Pastel blue
    '6e_KeywordAugmented': '#A5D6A5',      # Pastel green
    '6f_KeywordAugmentedCoT': '#8BB8E8',   # Pastel sky blue (best)
}

# General-purpose colors for other visualizations
COLORS = {
    # Primary palette (pastel)
    'primary': '#8BB8E8',      # Pastel blue
    'secondary': '#A5D6A5',    # Pastel green
    'tertiary': '#F4C79A',     # Pastel orange
    'quaternary': '#C9A5D6',   # Pastel purple

    # Semantic colors
    'positive': '#A5D6A5',     # Pastel green - for improvements/gains
    'negative': '#F4A5A5',     # Pastel red - for decreases/errors
    'neutral': '#B0B0B0',      # Grey - for baseline/neutral
    'highlight': '#F4E59A',    # Pastel yellow - for emphasis

    # Metrics colors (for multi-metric comparisons)
    'precision': '#8BB8E8',    # Pastel blue
    'recall': '#A5D6A5',       # Pastel green
    'f1_score': '#C9A5D6',     # Pastel purple
    'accuracy': '#F4C79A',     # Pastel orange

    # Error analysis colors
    'true_positive': '#A5D6A5',   # Pastel green
    'false_positive': '#F4A5A5',  # Pastel red
    'false_negative': '#F4C79A',  # Pastel orange
    'true_negative': '#B0B0B0',   # Grey

    # Category colors for F-code groups
    'F00-F09': '#8BB8E8',      # Organic disorders - blue
    'F10-F19': '#F4A5A5',      # Substance use - red
    'F20-F29': '#C9A5D6',      # Schizophrenia - purple
    'F30-F39': '#A5D6A5',      # Mood disorders - green
    'F40-F48': '#F4C79A',      # Anxiety/stress - orange
    'F50-F59': '#F4E59A',      # Behavioral syndromes - yellow
    'F60-F69': '#E8A5C9',      # Personality disorders - pink
    'F70-F79': '#A5E8E4',      # Intellectual disabilities - teal
    'F80-F89': '#D6C9A5',      # Developmental disorders - tan
    'F90-F99': '#B8B8E8',      # Childhood disorders - lavender
}

# Figure style settings for publication
FIGURE_STYLE = {
    'figure.figsize': (10, 6),
    'figure.dpi': 300,
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'axes.linewidth': 1.0,
    'axes.edgecolor': '#333333',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.framealpha': 0.9,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
}


def get_strategy_color(strategy_id: str) -> str:
    """Get color for a specific strategy."""
    return STRATEGY_COLORS.get(strategy_id, COLORS['neutral'])


def get_category_color(f_code: str) -> str:
    """Get color for an F-code category based on code prefix."""
    if f_code.startswith('F0'):
        return COLORS['F00-F09']
    elif f_code.startswith('F1'):
        return COLORS['F10-F19']
    elif f_code.startswith('F2'):
        return COLORS['F20-F29']
    elif f_code.startswith('F3'):
        return COLORS['F30-F39']
    elif f_code.startswith('F4'):
        return COLORS['F40-F48']
    elif f_code.startswith('F5'):
        return COLORS['F50-F59']
    elif f_code.startswith('F6'):
        return COLORS['F60-F69']
    elif f_code.startswith('F7'):
        return COLORS['F70-F79']
    elif f_code.startswith('F8'):
        return COLORS['F80-F89']
    elif f_code.startswith('F9'):
        return COLORS['F90-F99']
    return COLORS['neutral']


def apply_figure_style():
    """Apply publication-ready figure style settings."""
    import matplotlib.pyplot as plt
    for key, value in FIGURE_STYLE.items():
        plt.rcParams[key] = value


# =============================================================================
# AGGREGATED CONFIG DICTIONARY
# =============================================================================

PROJECT_CONFIG = {
    # Reproducibility
    'random_seed': RANDOM_SEED,

    # Paths
    'project_dir': PROJECT_DIR,
    'data_dir': DATA_DIR,
    'output_dir': OUTPUT_DIR,
    'logs_dir': LOGS_DIR,
    'models_dir': MODELS_DIR,
    'checkpoint_dir': CHECKPOINT_DIR,
    'final_model_dir': FINAL_MODEL_DIR,

    # Data paths
    'train_data_path': TRAIN_DATA_PATH,
    'val_data_path': VAL_DATA_PATH,
    'test_data_path': TEST_DATA_PATH,
    'train_truncated_path': TRAIN_TRUNCATED_PATH,
    'val_truncated_path': VAL_TRUNCATED_PATH,
    'test_truncated_path': TEST_TRUNCATED_PATH,

    # Model
    'base_model_name': BASE_MODEL_NAME,
    'max_seq_length': MAX_SEQ_LENGTH,
    'load_in_4bit': LOAD_IN_4BIT,

    # LoRA fine-tuning
    'lora_r': LORA_R,
    'lora_alpha': LORA_ALPHA,
    'lora_dropout': LORA_DROPOUT,

    # Chunking parameters
    'chunk_size': CHUNK_SIZE,
    'chunk_overlap': CHUNK_OVERLAP,
    'max_input_tokens': MAX_INPUT_TOKENS,

    # Generation
    'max_new_tokens': MAX_NEW_TOKENS,
    'temperature': TEMPERATURE,

    # Training
    'num_epochs': NUM_EPOCHS,
    'max_train_samples': MAX_TRAIN_SAMPLES,
    'train_batch_size': TRAIN_BATCH_SIZE,
    'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,
    'learning_rate': LEARNING_RATE,
    'warmup_steps': WARMUP_STEPS,
    'save_steps': SAVE_STEPS,

    # Evaluation
    'default_max_samples': DEFAULT_MAX_SAMPLES,
    'val_samples': VAL_SAMPLES,
}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_f_code(code: str) -> str:
    """
    Format F-code with period at 3rd position if not present.

    ICD-10 psychiatric F-codes have format F##.## where the period separates
    the category from the subcategory. This function ensures consistent formatting.

    Args:
        code: Raw F-code string (e.g., "F329" or "F32.9")

    Returns:
        Formatted F-code with period (e.g., "F32.9")

    Examples:
        >>> format_f_code("F329")
        'F32.9'
        >>> format_f_code("F32.9")
        'F32.9'
        >>> format_f_code("F17210")
        'F17.210'
    """
    code = str(code).strip()
    if len(code) > 3 and '.' not in code:
        return f"{code[:3]}.{code[3:]}"
    return code


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_config():
    """Return a copy of the project configuration dictionary."""
    return PROJECT_CONFIG.copy()


def print_config():
    """Print current configuration settings."""
    print("\n" + "=" * 60)
    print("PROJECT CONFIGURATION")
    print("=" * 60)
    for key, value in PROJECT_CONFIG.items():
        print(f"  {key}: {value}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    # When run directly, print configuration
    print_config()
    set_all_seeds()
