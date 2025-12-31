#!/usr/bin/env python3
"""
4_FineTuning.py - Fine-tuning GPT-OSS 20B with Chunking Approach

This script implements the chunking approach for fine-tuning:
- Documents split into overlapping chunks (2800 tokens, 200 overlap)
- Each chunk is a separate training example with the same labels
- Output: Simple JSON array of F-codes ["F32.9", "F17.210"]

CONFIGURATION:
- 20,000 training samples (sampled from chunked samples)
- 5 epochs
- Chunking approach: overlapping chunks preserve all information
- Checkpoints: Every 500 steps

Token Allocation per Chunk (max_seq_length = 4096):
  System Prompt (CoT):    ~550 tokens
  User Template:          ~30 tokens
  Chunk Content:          2800 tokens
  Output (JSON array):    100 tokens
  Buffer:                 ~616 tokens
  TOTAL:                  4096 tokens

Usage:
    python 4_FineTuning.py
"""

import os
os.environ["UNSLOTH_STABLE_DOWNLOADS"] = "1"

import sys
import random
import torch
import json
from pathlib import Path
from datetime import datetime

from unsloth import FastLanguageModel
from transformers import TextStreamer
from unsloth.chat_templates import train_on_responses_only
from trl import SFTConfig, SFTTrainer
from datasets import Dataset

# Import centralized config for reproducibility
from config import (
    RANDOM_SEED, set_all_seeds, BASE_MODEL_NAME,
    MAX_SEQ_LENGTH, LOAD_IN_4BIT, CHUNK_SIZE, CHUNK_OVERLAP, MAX_NEW_TOKENS,
    LORA_R, LORA_ALPHA, LORA_DROPOUT,
    NUM_EPOCHS, MAX_TRAIN_SAMPLES, TRAIN_BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE, WARMUP_STEPS, SAVE_STEPS, SAVE_TOTAL_LIMIT,
    TRAIN_CHUNKED_PATH, CHECKPOINT_DIR, FINAL_MODEL_DIR
)

# Set random seeds for reproducibility
set_all_seeds(RANDOM_SEED)

print("=" * 80, flush=True)
print("GPT-OSS 20B Fine-tuning with CHUNKING Approach", flush=True)
print("Overlapping chunks (2800 tokens, 200 overlap) | Output: JSON array only", flush=True)
print("=" * 80, flush=True)

# TRAINING CONFIGURATION
CONFIG = {
    # Model settings from centralized config
    'model_name': BASE_MODEL_NAME,
    'max_seq_length': MAX_SEQ_LENGTH,
    'load_in_4bit': LOAD_IN_4BIT,

    # Data paths - USE CHUNKED JSON DATA
    'train_data_path': TRAIN_CHUNKED_PATH,
    'output_dir': FINAL_MODEL_DIR,
    'checkpoint_dir': CHECKPOINT_DIR,

    # Chunking settings (from config)
    'chunk_size': CHUNK_SIZE,              # 2800 tokens
    'chunk_overlap': CHUNK_OVERLAP,        # 200 tokens overlap
    'max_new_tokens': MAX_NEW_TOKENS,      # 100 tokens for JSON array

    # Training parameters
    'max_samples': MAX_TRAIN_SAMPLES,      # 20,000
    'num_train_epochs': NUM_EPOCHS,        # 5 epochs
    'max_steps': -1,
    'per_device_train_batch_size': TRAIN_BATCH_SIZE,  # 2
    'gradient_accumulation_steps': GRADIENT_ACCUMULATION_STEPS,  # 4
    'learning_rate': LEARNING_RATE,        # 2e-4
    'warmup_steps': WARMUP_STEPS,          # 100
    'logging_steps': 100,
    'save_steps': SAVE_STEPS,              # 500
    'save_total_limit': SAVE_TOTAL_LIMIT,  # None (keep all)

    # LoRA parameters
    'lora_r': LORA_R,                      # 16
    'lora_alpha': LORA_ALPHA,              # 32
    'lora_dropout': LORA_DROPOUT,          # 0

    'random_seed': RANDOM_SEED,            # 42
}

# Psychiatric F-code system prompt (CoT - must match 6d evaluation)
# Token count: ~550 tokens
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


def load_chunked_data(config):
    """Load pre-chunked training data from JSON."""
    print("\n" + "=" * 80, flush=True)
    print("Step 1: Loading Chunked Training Data", flush=True)
    print("=" * 80, flush=True)

    data_path = Path(config['train_data_path'])
    print(f"Reading: {data_path}", flush=True)

    if not data_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {data_path}\n"
            "Please run 2_PrepareInstructionData.py first."
        )

    with open(data_path, 'r') as f:
        train_data = json.load(f)

    print(f"Loaded {len(train_data)} chunked training examples", flush=True)

    # Sample if needed (random already seeded via set_all_seeds at module level)
    if config['max_samples'] and len(train_data) > config['max_samples']:
        train_data = random.sample(train_data, config['max_samples'])
        print(f"Sampled {len(train_data)} examples for training", flush=True)

    return train_data


def format_training_example(item):
    """
    Convert chunked item to GPT-OSS conversation format.

    Each chunk is a complete training example.
    Output: Simple JSON array of F-codes (no explanations).
    """
    # The output from 2_PrepareInstructionData.py is already a JSON array string
    # e.g., '["F32.9", "F17.210"]'
    f_code_output = item['output']

    # Create conversation
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": f"Discharge Summary:\n{item['input']}"
        },
        {
            "role": "assistant",
            "content": f_code_output
        }
    ]

    return {"messages": messages}


def create_dataset(train_data, tokenizer, config):
    """
    Create HuggingFace dataset from chunked data.
    """
    print("\n" + "=" * 80, flush=True)
    print("Step 2: Converting to Training Format (Chunked Data)", flush=True)
    print("=" * 80, flush=True)

    print("Converting to GPT-OSS conversation format...", flush=True)
    formatted_data = []

    for idx, item in enumerate(train_data):
        if idx % 2000 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(train_data)} samples...", flush=True)
        try:
            formatted_example = format_training_example(item)
            formatted_data.append(formatted_example)
        except Exception as e:
            print(f"Warning: Skipped item {idx}: {e}", flush=True)
            continue

    print(f"Created {len(formatted_data)} training examples", flush=True)

    # Create HuggingFace dataset
    dataset = Dataset.from_list(formatted_data)

    # Apply chat template
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False
        ) for convo in convos]
        return {"text": texts}

    print("Applying chat template...", flush=True)
    dataset = dataset.map(formatting_prompts_func, batched=True)

    print(f"Dataset ready: {len(dataset)} training examples", flush=True)

    # Show sample
    print("\n" + "-" * 80, flush=True)
    print("Sample training example (first 500 chars):", flush=True)
    print("-" * 80, flush=True)
    print(dataset[0]['text'][:500])
    print("...", flush=True)
    print("-" * 80, flush=True)

    return dataset


def main():
    print("\nCHUNKING Training Configuration:", flush=True)
    print(f"  Approach: Overlapping chunks ({CONFIG['chunk_size']} tokens, {CONFIG['chunk_overlap']} overlap)", flush=True)
    print(f"  Output format: JSON array (F-codes only)", flush=True)
    for key, value in CONFIG.items():
        if key not in ['train_data_path', 'model_name']:
            print(f"  {key}: {value}", flush=True)
    print(flush=True)

    # Estimate training
    estimated_examples = CONFIG['max_samples']
    steps_per_epoch = estimated_examples / (CONFIG['per_device_train_batch_size'] * CONFIG['gradient_accumulation_steps'])
    total_steps = steps_per_epoch * CONFIG['num_train_epochs']
    estimated_hours = (total_steps * 8) / 3600  # ~8 sec/step estimate
    num_checkpoints = int(total_steps / CONFIG['save_steps'])

    print(f"\nTraining Estimates (CHUNKING):", flush=True)
    print(f"  Training examples: {estimated_examples:.0f} (sampled from chunks)", flush=True)
    print(f"  Steps per epoch: ~{steps_per_epoch:.0f}", flush=True)
    print(f"  Total steps ({CONFIG['num_train_epochs']} epochs): ~{total_steps:.0f}", flush=True)
    print(f"  Number of checkpoints: ~{num_checkpoints}", flush=True)
    print(f"  Estimated time: ~{estimated_hours:.1f} hours", flush=True)
    print(flush=True)

    # Load model
    print("=" * 80, flush=True)
    print("Step 0: Loading Base Model", flush=True)
    print("=" * 80, flush=True)
    print(f"Model: {CONFIG['model_name']}", flush=True)
    print(f"Max sequence length: {CONFIG['max_seq_length']}", flush=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CONFIG['model_name'],
        dtype=None,
        max_seq_length=CONFIG['max_seq_length'],
        load_in_4bit=CONFIG['load_in_4bit'],
        full_finetuning=False,
    )

    print("Model loaded!", flush=True)

    # Add LoRA adapters
    print("\nAdding LoRA adapters...", flush=True)
    model = FastLanguageModel.get_peft_model(
        model,
        r=CONFIG['lora_r'],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=CONFIG['lora_alpha'],
        lora_dropout=CONFIG['lora_dropout'],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=CONFIG['random_seed'],
        use_rslora=False,
        loftq_config=None,
    )

    print("LoRA adapters added!", flush=True)

    # Load chunked data
    train_data = load_chunked_data(CONFIG)
    dataset = create_dataset(train_data, tokenizer, CONFIG)

    # Setup trainer
    print("\n" + "=" * 80, flush=True)
    print("Step 3: Configuring Trainer", flush=True)
    print("=" * 80, flush=True)

    # Create output directories
    Path(CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)
    Path(CONFIG['checkpoint_dir']).mkdir(parents=True, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            per_device_train_batch_size=CONFIG['per_device_train_batch_size'],
            gradient_accumulation_steps=CONFIG['gradient_accumulation_steps'],
            warmup_steps=CONFIG['warmup_steps'],
            num_train_epochs=CONFIG['num_train_epochs'],
            max_steps=CONFIG['max_steps'],
            learning_rate=CONFIG['learning_rate'],
            logging_steps=CONFIG['logging_steps'],
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=CONFIG['random_seed'],
            output_dir=CONFIG['checkpoint_dir'],
            save_steps=CONFIG['save_steps'],
            save_total_limit=CONFIG['save_total_limit'],
            report_to="none",
        ),
    )

    print("Trainer configured!", flush=True)

    # Apply train_on_responses_only
    print("\nApplying train_on_responses_only masking...", flush=True)
    gpt_oss_kwargs = dict(
        instruction_part="<|start|>user<|message|>",
        response_part="<|start|>assistant<|message|>"
    )

    trainer = train_on_responses_only(trainer, **gpt_oss_kwargs)
    print("Response-only training configured!", flush=True)

    # Check GPU memory
    print("\n" + "=" * 80, flush=True)
    print("GPU Memory Status", flush=True)
    print("=" * 80, flush=True)
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU: {gpu_stats.name}", flush=True)
    print(f"Max memory: {max_memory} GB", flush=True)
    print(f"Reserved memory: {start_gpu_memory} GB", flush=True)

    # Train
    print("\n" + "=" * 80, flush=True)
    print(f"Step 4: Starting CHUNKING Training ({CONFIG['num_train_epochs']} epochs)", flush=True)
    print("=" * 80, flush=True)
    print(f"Training examples: {len(dataset)} (from chunked documents)", flush=True)
    print(f"Epochs: {CONFIG['num_train_epochs']}", flush=True)
    print(f"Effective batch size: {CONFIG['per_device_train_batch_size'] * CONFIG['gradient_accumulation_steps']}", flush=True)
    print("=" * 80, flush=True)
    print(flush=True)

    start_time = datetime.now()

    trainer_stats = trainer.train()

    end_time = datetime.now()
    training_time = (end_time - start_time).total_seconds()

    # Training complete
    print("\n" + "=" * 80, flush=True)
    print("CHUNKING Training Complete!", flush=True)
    print("=" * 80, flush=True)

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)

    print(f"Training time: {training_time:.2f} seconds ({training_time/60:.2f} minutes, {training_time/3600:.2f} hours)", flush=True)
    print(f"Peak reserved memory: {used_memory} GB ({used_percentage}%)", flush=True)
    print(f"Final train loss: {trainer_stats.metrics.get('train_loss', 'N/A')}", flush=True)

    # Test fine-tuned model
    print("\n" + "=" * 80, flush=True)
    print("Step 5: Testing Fine-tuned Model", flush=True)
    print("=" * 80, flush=True)

    test_note = train_data[0]['input'][:2000]
    actual_codes = train_data[0]['output']

    print(f"Test note (first 300 chars): {test_note[:300]}...", flush=True)
    print(f"\nActual F-codes: {actual_codes}", flush=True)
    print("\nModel prediction:", flush=True)
    print("-" * 80, flush=True)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Discharge Summary:\n{test_note}"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to("cuda")

    FastLanguageModel.for_inference(model)
    _ = model.generate(**inputs, max_new_tokens=CONFIG['max_new_tokens'], streamer=TextStreamer(tokenizer))

    # Save final model
    print("\n" + "=" * 80, flush=True)
    print("Step 6: Saving Final Fine-tuned Model", flush=True)
    print("=" * 80, flush=True)

    save_path = CONFIG['output_dir']
    print(f"Saving to: {save_path}", flush=True)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # Save training config
    config_path = os.path.join(save_path, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'approach': 'Chunking',
            'description': f'Overlapping chunks ({CONFIG["chunk_size"]} tokens, {CONFIG["chunk_overlap"]} overlap)',
            'output_format': 'JSON array (F-codes only)',
            'chunking': True,
            'chunk_size': CONFIG['chunk_size'],
            'chunk_overlap': CONFIG['chunk_overlap'],
            'aggregation_strategy': 'union',
            'config': CONFIG,
            'training_stats': {
                'train_loss': trainer_stats.metrics.get('train_loss', None),
                'train_runtime': trainer_stats.metrics.get('train_runtime', None),
                'training_examples': len(dataset),
                'actual_steps': trainer_stats.metrics.get('train_steps', None),
            },
            'timestamp': datetime.now().isoformat(),
        }, f, indent=2)

    print(f"Model saved to: {save_path}", flush=True)
    print(f"Config saved to: {config_path}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("Fine-tuning Complete!", flush=True)
    print("=" * 80, flush=True)
    print(f"\nApproach: Chunking ({CONFIG['chunk_size']} tokens, {CONFIG['chunk_overlap']} overlap)", flush=True)
    print(f"Output format: JSON array (F-codes only)", flush=True)
    print(f"Training examples: {len(dataset)} (from chunked documents)", flush=True)
    print(f"Training time: {training_time/60:.1f} minutes ({training_time/3600:.2f} hours)", flush=True)
    print(f"Final loss: {trainer_stats.metrics.get('train_loss', 'N/A')}", flush=True)
    print(f"\nCheckpoints saved to: {CONFIG['checkpoint_dir']}/", flush=True)
    print(f"Final model saved to: {CONFIG['output_dir']}/", flush=True)
    print(f"\nNext step: Run 5_FindBestCheckpoint.py to select best checkpoint", flush=True)
    print("=" * 80, flush=True)

    return 0


if __name__ == "__main__":
    try:
        exit(main())
    except Exception as e:
        print(f"\nError: {e}", flush=True)
        import traceback
        traceback.print_exc()
        exit(1)
