#!/usr/bin/env python3
"""
SIRET-BERT Training Script V3 - MPS Memory Optimized

Fine-tunes a sentence transformer model for French business name matching.
Uses MultipleNegativesRankingLoss (MNRL) for contrastive learning.

MPS Memory Optimizations:
- Gradient checkpointing enabled
- Aggressive cache clearing between steps
- PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to disable memory limits
- Optimized for MacBook M4 Pro with 24GB unified RAM
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# MPS Memory optimization - MUST be set before importing torch
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import torch
import yaml
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    losses,
)
from sentence_transformers.trainer import SentenceTransformerTrainer
from sentence_transformers.training_args import SentenceTransformerTrainingArguments
from transformers import TrainerCallback

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Memory Management Callback
# ============================================================================

class MPSMemoryCallback(TrainerCallback):
    """Callback to aggressively manage MPS memory during training."""
    
    def __init__(self, clear_every_n_steps: int = 10):
        self.clear_every_n_steps = clear_every_n_steps
        self.step_count = 0
        
    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        if self.step_count % self.clear_every_n_steps == 0:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
                gc.collect()

    def on_epoch_end(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
            gc.collect()
            logger.info("Cleared MPS memory cache at epoch end")


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def load_config(path: Path) -> Dict:
    """Load configuration from YAML file."""
    with open(path) as f:
        return yaml.safe_load(f)


# ============================================================================
# Dataset Loading
# ============================================================================

def load_semantic_dataset(data_path: Path, max_samples: Optional[int] = None) -> Dataset:
    """Load the JSONL dataset into a HuggingFace Dataset."""
    data = {"anchor": [], "positive": [], "negative": []}
    
    with open(data_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
                
            item = json.loads(line.strip())
            data["anchor"].append(item["anchor"])
            data["positive"].append(item["positive"])
            data["negative"].append(item.get("negative", ""))
    
    logger.info(f"Loaded {len(data['anchor'])} examples from {data_path}")
    return Dataset.from_dict(data)


# ============================================================================
# Evaluator  
# ============================================================================

class SimilarityEvaluator:
    """Simple evaluator that computes similarity between anchors and positives."""
    
    def __init__(self, data_path: Path, name: str = "eval", max_samples: int = 500):
        self.anchors = []
        self.positives = []
        self.name = name
        
        with open(data_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_samples:
                    break
                item = json.loads(line.strip())
                self.anchors.append(item["anchor"])
                self.positives.append(item["positive"])
        
        logger.info(f"Evaluator initialized with {len(self.anchors)} samples")
    
    def __call__(self, model: SentenceTransformer, output_path: str = None, epoch: int = -1, steps: int = -1) -> Dict[str, float]:
        """Compute mean cosine similarity between anchors and positives."""
        # Use smaller batch for evaluation to save memory
        anchor_embs = model.encode(
            self.anchors, 
            convert_to_tensor=True, 
            show_progress_bar=False,
            batch_size=8
        )
        positive_embs = model.encode(
            self.positives, 
            convert_to_tensor=True, 
            show_progress_bar=False,
            batch_size=8
        )
        
        similarities = torch.nn.functional.cosine_similarity(anchor_embs, positive_embs)
        mean_sim = similarities.mean().item()
        
        # Free memory
        del anchor_embs, positive_embs
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        
        logger.info(f"Epoch {epoch}, Steps {steps}: Mean similarity = {mean_sim:.4f}")
        return {"eval_similarity": mean_sim}


# ============================================================================
# Training
# ============================================================================

def enable_gradient_checkpointing(model: SentenceTransformer) -> bool:
    """Enable gradient checkpointing on the transformer model to save memory."""
    try:
        # Get the transformer model (first module)
        for module in model.modules():
            if hasattr(module, 'gradient_checkpointing_enable'):
                module.gradient_checkpointing_enable()
                logger.info("Gradient checkpointing enabled")
                return True
    except Exception as e:
        logger.warning(f"Could not enable gradient checkpointing: {e}")
    return False


def train(config: Dict, resume_from: Optional[str] = None):
    """Run the training loop with MPS memory optimizations."""
    
    model_cfg = config["model"]
    train_cfg = config["training"]
    data_cfg = config["dataset"]
    
    # Force MPS device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    
    # Clear memory before starting
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    
    # Output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(model_cfg["output_dir"]) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.safe_dump(config, f)
    
    logger.info(f"Output directory: {output_dir}")
    
    # Load base model
    logger.info(f"Loading base model: {model_cfg['base_model']}")
    model = SentenceTransformer(model_cfg["base_model"], device=device)
    max_seq_length = config.get("preprocessing", {}).get("max_length")
    if max_seq_length:
        model.max_seq_length = int(max_seq_length)
        logger.info(f"Max sequence length set to: {model.max_seq_length}")
    
    # Enable/disable gradient checkpointing for memory/speed tradeoff
    use_ckpt = bool(train_cfg.get("gradient_checkpointing", True))
    if use_ckpt:
        enable_gradient_checkpointing(model)
    else:
        logger.info("Gradient checkpointing disabled for speed")
    
    # Load datasets
    logger.info("Loading datasets...")
    train_data_path = Path(data_cfg["train_file"])
    eval_data_path = Path(data_cfg.get("eval_file", data_cfg["train_file"]))
    
    train_dataset = load_semantic_dataset(
        train_data_path,
        max_samples=data_cfg.get("max_samples")
    )
    
    # Create evaluator with smaller sample size
    evaluator = SimilarityEvaluator(eval_data_path, max_samples=500)
    
    # Loss function
    loss_cfg = config.get("loss", {})
    train_loss = losses.MultipleNegativesRankingLoss(
        model=model,
        scale=loss_cfg.get("scale", 20.0)
    )
    logger.info(f"Using MultipleNegativesRankingLoss (scale={loss_cfg.get('scale', 20.0)})")
    
    # Training arguments - VERY conservative for MPS
    eval_cfg = config.get("evaluation", {})
    batch_size = train_cfg.get("batch_size", 8)  # Very small batch
    grad_accum = train_cfg.get("gradient_accumulation_steps", 8)  # More accumulation
    
    metric_for_best = eval_cfg.get("metric_for_best_model", "eval_similarity")
    greater_is_better = metric_for_best in {"eval_similarity", "eval_mean_similarity"}

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg.get("epochs", 3),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=train_cfg.get("learning_rate", 2e-5),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.1),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        fp16=False,
        bf16=False,
        dataloader_num_workers=config.get("hardware", {}).get("dataloader_workers", 0),
        dataloader_pin_memory=config.get("hardware", {}).get("pin_memory", False),
        logging_steps=eval_cfg.get("logging_steps", 100),
        eval_strategy="steps",
        eval_steps=eval_cfg.get("eval_steps", 1000),  # Less frequent eval
        save_strategy="steps",
        save_steps=eval_cfg.get("save_steps", 1000),
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=greater_is_better,
        seed=config.get("seed", 42),
        report_to="none",
        # MPS optimizations
        optim="adamw_torch",  # Use standard Adam
        gradient_checkpointing=use_ckpt,
        max_grad_norm=1.0,  # Clip gradients
    )
    
    # Create trainer with memory callback
    # Note: evaluator must be a list for SentenceTransformerTrainer
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=train_loss,
        evaluator=[evaluator],  # Must be a list
        callbacks=[MPSMemoryCallback(clear_every_n_steps=train_cfg.get("clear_cache_every", 200))],
    )
    
    # Train
    logger.info("Starting training...")
    logger.info(f"  Epochs: {train_cfg.get('epochs', 3)}")
    logger.info(f"  Batch size: {batch_size} (effective: {batch_size * grad_accum})")
    logger.info(f"  Learning rate: {train_cfg.get('learning_rate', 2e-5)}")
    logger.info(f"  Training samples: {len(train_dataset)}")
    logger.info(f"  Gradient checkpointing: enabled")
    logger.info(f"  MPS high watermark: disabled")
    
    trainer.train()
    
    # Save final model
    final_path = output_dir / "final"
    model.save(str(final_path))
    logger.info(f"Model saved to: {final_path}")
    
    # Save training metadata
    metadata = {
        "completed": datetime.now().isoformat(),
        "base_model": model_cfg["base_model"],
        "epochs": train_cfg.get("epochs", 3),
        "batch_size": batch_size,
        "effective_batch_size": batch_size * grad_accum,
        "learning_rate": train_cfg.get("learning_rate", 2e-5),
        "train_samples": len(train_dataset),
        "device": device,
        "gradient_checkpointing": True,
    }
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Training complete!")
    
    return output_dir


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Fine-tune sentence transformer for SIRET matching")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Config YAML file")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    print("=" * 60)
    print("SIRET-BERT Fine-Tuning (MPS Memory Optimized)")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Base model: {config['model']['base_model']}")
    print(f"MPS High Watermark: DISABLED")
    print("=" * 60)
    
    # Set seed
    torch.manual_seed(config.get("seed", 42))
    
    # Run training
    output_dir = train(config, resume_from=args.resume)
    
    print("\n" + "=" * 60)
    print(f"✅ Training complete! Model saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
