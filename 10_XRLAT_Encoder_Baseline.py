#!/usr/bin/env python3
"""
10_XRLAT_Encoder_Baseline.py
XR-LAT encoder baseline with Clinical-Longformer backbone.

Architecture:
- Clinical-Longformer encoder (yikuan8/Clinical-Longformer, 4096-token window)
- Label-wise attention mechanism (Mullenbach et al., 2018)
- Multi-label classification over 466 ICD-10 F-codes

Training: up to 20 epochs, best checkpoint by validation micro-F1.
Evaluation: same 1000 test samples (random_state=42) used by all LLM strategies.

Outputs to: outputs/10_XRLAT_Encoder/
"""

import os
import json
import time
import datetime
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, precision_score, recall_score

from config import RANDOM_SEED, set_all_seeds, TRAIN_DATA_PATH, VAL_DATA_PATH, TEST_DATA_PATH, OUTPUT_DIR

set_all_seeds(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    "encoder_name":    "yikuan8/Clinical-Longformer",
    "max_length":      4096,
    "hidden_size":     768,       # Clinical-Longformer hidden dim
    "batch_size":      4,         # per-GPU (34GB VRAM, Longformer sparse attention)
    "grad_accum":      2,         # effective batch = 8
    "lr":              2e-5,
    "num_epochs":      20,
    "warmup_ratio":    0.1,
    "threshold":       0.5,       # sigmoid threshold for multi-label
    "max_samples_eval": 1000,
    "random_seed":     RANDOM_SEED,
    "output_dir":      os.path.join(OUTPUT_DIR, "10_XRLAT_Encoder"),
    "model_dir":       os.path.join(OUTPUT_DIR, "10_XRLAT_Encoder", "model"),
    "patience":        5,         # early stopping: stop if no val-F1 improvement for N epochs
}

os.makedirs(CONFIG["output_dir"], exist_ok=True)
os.makedirs(CONFIG["model_dir"], exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")


# ---------------------------------------------------------------------------
# Label utilities
# ---------------------------------------------------------------------------
def parse_codes(codes_str: str):
    """Parse comma- or space-separated F-codes, stripping punctuation."""
    return [c.strip().rstrip(",") for c in str(codes_str).replace(",", " ").split()
            if c.strip().rstrip(",").startswith("F")]


def build_label_vocab(train_df: pd.DataFrame):
    """Collect all unique F-codes from training set and sort them."""
    codes = set()
    for codes_str in train_df["f_codes_str"].dropna():
        codes.update(parse_codes(codes_str))
    label_list = sorted(codes)
    label2id = {c: i for i, c in enumerate(label_list)}
    return label_list, label2id


def codes_str_to_vector(codes_str: str, label2id: dict, num_labels: int) -> np.ndarray:
    vec = np.zeros(num_labels, dtype=np.float32)
    for c in parse_codes(codes_str):
        if c in label2id:
            vec[label2id[c]] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ClinicalNoteDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, label2id: dict,
                 num_labels: int, max_length: int):
        self.texts = df["text"].tolist()
        self.labels = [
            codes_str_to_vector(str(row["f_codes_str"]), label2id, num_labels)
            for _, row in df.iterrows()
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels":         torch.tensor(self.labels[idx], dtype=torch.float),
        }


# ---------------------------------------------------------------------------
# XR-LAT Model
# ---------------------------------------------------------------------------
class XRLATModel(nn.Module):
    """
    Label-wise attention over Clinical-Longformer encoder outputs.
    Each label has a learned query vector that attends over all token positions.
    """
    def __init__(self, encoder_name: str, num_labels: int, hidden_size: int = 768):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            encoder_name,
            output_hidden_states=False,
            add_pooling_layer=False,  # we do our own pooling
        )
        # Label-wise attention queries: shape (num_labels, hidden_size)
        self.label_queries = nn.Parameter(
            torch.randn(num_labels, hidden_size) * 0.02
        )
        # Final classification weight: (num_labels, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_labels, bias=True)

    def forward(self, input_ids, attention_mask):
        # H: (batch, seq_len, hidden_size)
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        H = outputs.last_hidden_state  # (B, L, D)

        # Label-wise attention
        # scores: (B, num_labels, L)  via einsum
        scores = torch.einsum("bld,nd->bln", H, self.label_queries)  # (B, L, N) -> transpose -> (B, N, L)
        scores = scores.permute(0, 2, 1)  # (B, N, L)

        # Mask padding tokens
        mask = attention_mask.unsqueeze(1).float()  # (B, 1, L)
        scores = scores + (1.0 - mask) * -1e9
        attn = torch.softmax(scores, dim=-1)  # (B, N, L)

        # Context vectors per label: (B, N, D)
        context = torch.einsum("bnl,bld->bnd", attn, H)

        # Per-label prediction: diagonal of classifier applied to each label's context
        # classifier.weight: (N, D), bias: (N,)
        # logits_n = context_n · w_n + b_n
        logits = (context * self.classifier.weight.unsqueeze(0)).sum(-1) + self.classifier.bias
        # logits: (B, N)
        return logits


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(all_preds: np.ndarray, all_labels: np.ndarray,
                    threshold: float = 0.5):
    """
    all_preds:  (N, num_labels) float logits or probabilities
    all_labels: (N, num_labels) binary ground-truth
    Returns dict with micro_f1, micro_precision, micro_recall, exact_match.
    """
    binary_preds = (all_preds >= threshold).astype(int)

    micro_f1  = f1_score(all_labels, binary_preds, average="micro", zero_division=0)
    micro_p   = precision_score(all_labels, binary_preds, average="micro", zero_division=0)
    micro_r   = recall_score(all_labels, binary_preds, average="micro", zero_division=0)

    exact_match = np.mean(
        np.all(binary_preds == all_labels, axis=1)
    )

    return {
        "micro_f1":        float(micro_f1),
        "micro_precision": float(micro_p),
        "micro_recall":    float(micro_r),
        "exact_match":     float(exact_match),
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_epoch(model, loader, optimizer, scheduler, scaler, grad_accum):
    model.train()
    total_loss = 0.0
    criterion = nn.BCEWithLogitsLoss()
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].to(DEVICE)

        with torch.amp.autocast("cuda"):
            logits = model(input_ids, attention_mask)
            loss   = criterion(logits, labels) / grad_accum

        scaler.scale(loss).backward()
        total_loss += loss.item() * grad_accum

        if (step + 1) % grad_accum == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        if (step + 1) % 100 == 0:
            print(f"  step {step+1}/{len(loader)}, loss={total_loss/(step+1):.4f}", flush=True)

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, threshold=0.5):
    model.eval()
    all_probs  = []
    all_labels = []

    for batch in loader:
        input_ids      = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels         = batch["labels"].numpy()

        with torch.amp.autocast("cuda"):
            logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().float().numpy()

        all_probs.append(probs)
        all_labels.append(labels)

    all_probs  = np.vstack(all_probs)
    all_labels = np.vstack(all_labels)
    return compute_metrics(all_probs, all_labels, threshold)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("XR-LAT Encoder Baseline Training & Evaluation")
    print("=" * 60)
    t0 = time.time()

    # ---- Load data ----
    print("\nLoading data...", flush=True)
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    val_df   = pd.read_csv(VAL_DATA_PATH)
    test_df  = pd.read_csv(TEST_DATA_PATH)

    # Sample 1000 test cases with same seed used by all LLM scripts
    test_df  = test_df.sample(n=CONFIG["max_samples_eval"],
                              random_state=CONFIG["random_seed"])

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}", flush=True)

    # ---- Label vocabulary ----
    label_list, label2id = build_label_vocab(train_df)
    num_labels = len(label_list)
    print(f"Unique F-codes (labels): {num_labels}", flush=True)

    # ---- Tokenizer ----
    print(f"\nLoading tokenizer: {CONFIG['encoder_name']}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["encoder_name"])

    # ---- Datasets & loaders ----
    train_ds = ClinicalNoteDataset(train_df, tokenizer, label2id,
                                   num_labels, CONFIG["max_length"])
    val_ds   = ClinicalNoteDataset(val_df,   tokenizer, label2id,
                                   num_labels, CONFIG["max_length"])
    test_ds  = ClinicalNoteDataset(test_df,  tokenizer, label2id,
                                   num_labels, CONFIG["max_length"])

    g = torch.Generator(); g.manual_seed(RANDOM_SEED)
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"],
                              shuffle=True, num_workers=4, generator=g,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=CONFIG["batch_size"] * 2,
                              shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=CONFIG["batch_size"] * 2,
                              shuffle=False, num_workers=4, pin_memory=True)

    # ---- Model ----
    best_ckpt_path = os.path.join(CONFIG["model_dir"], "best_model.pt")

    if os.path.exists(best_ckpt_path):
        print(f"\nFound existing best model: {best_ckpt_path}", flush=True)
        print("Skipping training — loading saved weights for evaluation.", flush=True)
        model = XRLATModel(CONFIG["encoder_name"], num_labels, CONFIG["hidden_size"])
        model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
        model.to(DEVICE)
    else:
        print(f"\nBuilding model: {CONFIG['encoder_name']}", flush=True)
        model = XRLATModel(CONFIG["encoder_name"], num_labels, CONFIG["hidden_size"])
        model.to(DEVICE)

        total_params     = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,} | Trainable: {trainable_params:,}", flush=True)

        # ---- Optimizer & scheduler ----
        num_update_steps = (len(train_loader) // CONFIG["grad_accum"]) * CONFIG["num_epochs"]
        warmup_steps     = int(num_update_steps * CONFIG["warmup_ratio"])

        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"],
                                      weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=num_update_steps
        )
        scaler = torch.amp.GradScaler("cuda")

        # ---- Training ----
        best_val_f1 = 0.0
        patience_counter = 0
        history = []

        for epoch in range(1, CONFIG["num_epochs"] + 1):
            ep_t0 = time.time()
            print(f"\n--- Epoch {epoch}/{CONFIG['num_epochs']} ---", flush=True)
            train_loss = train_epoch(model, train_loader, optimizer, scheduler,
                                     scaler, CONFIG["grad_accum"])

            val_metrics = evaluate(model, val_loader, CONFIG["threshold"])
            elapsed = (time.time() - ep_t0) / 60.0

            print(f"  train_loss={train_loss:.4f} | val_micro_F1={val_metrics['micro_f1']:.4f} "
                  f"| val_P={val_metrics['micro_precision']:.3f} "
                  f"| val_R={val_metrics['micro_recall']:.3f} "
                  f"| {elapsed:.1f} min", flush=True)

            history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"val_{k}": v for k, v in val_metrics.items()},
            })

            if val_metrics["micro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["micro_f1"]
                torch.save(model.state_dict(), best_ckpt_path)
                print(f"  >> New best val micro-F1: {best_val_f1:.4f} — model saved", flush=True)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"  No improvement ({patience_counter}/{CONFIG['patience']})", flush=True)
                if patience_counter >= CONFIG["patience"]:
                    print("  Early stopping triggered.", flush=True)
                    break

        # Save training history
        history_path = os.path.join(CONFIG["output_dir"], "training_history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"\nTraining history saved: {history_path}", flush=True)

        # Reload best weights for evaluation
        model.load_state_dict(torch.load(best_ckpt_path, map_location=DEVICE))
        print(f"Best model reloaded (val micro-F1={best_val_f1:.4f})", flush=True)

    # ---- Test evaluation ----
    print("\nEvaluating on test set (n=1000)...", flush=True)
    test_metrics = evaluate(model, test_loader, CONFIG["threshold"])

    print("\n" + "=" * 40)
    print("TEST RESULTS")
    print("=" * 40)
    print(f"Micro-F1:    {test_metrics['micro_f1']:.4f}")
    print(f"Precision:   {test_metrics['micro_precision']:.4f}")
    print(f"Recall:      {test_metrics['micro_recall']:.4f}")
    print(f"Exact Match: {test_metrics['exact_match']:.4f} ({test_metrics['exact_match']*100:.1f}%)")
    print(f"Total time:  {(time.time()-t0)/60:.1f} min")
    print("=" * 40)

    # ---- Save results ----
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    results = {
        "model": "XR-LAT (Clinical-Longformer, 149M params)",
        "encoder": CONFIG["encoder_name"],
        "num_labels": num_labels,
        "test_samples": len(test_df),
        "random_seed": CONFIG["random_seed"],
        "timestamp": ts,
        "metrics": {
            "micro_f1":        test_metrics["micro_f1"],
            "micro_precision": test_metrics["micro_precision"],
            "micro_recall":    test_metrics["micro_recall"],
            "exact_match":     test_metrics["exact_match"],
            "exact_match_pct": round(test_metrics["exact_match"] * 100, 1),
        },
        "config": {k: v for k, v in CONFIG.items()
                   if k not in ("output_dir", "model_dir")},
    }

    results_path = os.path.join(CONFIG["output_dir"], f"xrlat_results_{ts}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}", flush=True)


if __name__ == "__main__":
    main()
