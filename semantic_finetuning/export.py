#!/usr/bin/env python3
"""
SIRET-BERT Export Script

Exports the fine-tuned model for production use:
1. Copies model to standard location
2. Creates metadata file
3. Updates environment configuration

FIXED: Now properly handles output directory naming to avoid deleting parent.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def export_model(
    model_path: Path,
    output_dir: Path,
    version: str = "v1",
    metrics: dict = None,
):
    """Export model to production location."""
    
    # Use a specific subdirectory name to avoid deleting parent
    target_dir = output_dir / f"siret-bert-{version}"
    
    print(f"Exporting model from: {model_path}")
    print(f"Target directory: {target_dir}")
    
    # Safety check: don't delete if target contains other runs
    if target_dir.exists():
        # Only remove if it looks like a previous export (has sireto_metadata.json)
        if (target_dir / "sireto_metadata.json").exists():
            print(f"  Removing existing export: {target_dir}")
            shutil.rmtree(target_dir)
        else:
            raise ValueError(f"Target {target_dir} exists but doesn't look like a previous export. Aborting.")
    
    # Make sure parent exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy model files
    shutil.copytree(model_path, target_dir)
    print(f"  ✓ Copied model files")
    
    # Remove optimizer/training files to save space
    for pattern in ["optimizer.pt", "scheduler.pt", "training_args.bin", "rng_state.pth"]:
        for f in target_dir.glob(pattern):
            f.unlink()
            print(f"  ✓ Removed training artifact: {f.name}")
    
    # Create metadata
    metadata = {
        "name": f"siret-bert-{version}",
        "version": version,
        "exported": datetime.now().isoformat(),
        "source": str(model_path),
    }
    if metrics:
        metadata["metrics"] = metrics
    
    meta_path = target_dir / "sireto_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata: {meta_path}")
    
    # Print usage instructions
    print("\n" + "=" * 60)
    print("✅ Export complete!")
    print("=" * 60)
    print("\nTo use the fine-tuned model, set the environment variables:")
    print(f"\n  export XGB_SEMANTIC_MODEL={target_dir}")
    print(f"  export XGB_SEMANTIC_ENABLED=1")
    print("\nOr add to your script:")
    print(f"\n  import os")
    print(f"  os.environ['XGB_SEMANTIC_MODEL'] = '{target_dir}'")
    print(f"  os.environ['XGB_SEMANTIC_ENABLED'] = '1'")
    print()
    
    return target_dir


def main():
    parser = argparse.ArgumentParser(description="Export fine-tuned SIRET-BERT model")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output-dir", type=Path, default=Path("models/semantic"),
                        help="Output directory for exported model")
    parser.add_argument("--version", type=str, default="v1", help="Model version string")
    parser.add_argument("--metrics-file", type=Path, default=None, help="JSON file with evaluation metrics")
    args = parser.parse_args()
    
    if not args.model.exists():
        print(f"ERROR: Model path does not exist: {args.model}")
        sys.exit(1)
    
    metrics = None
    if args.metrics_file and args.metrics_file.exists():
        with open(args.metrics_file) as f:
            metrics = json.load(f)
    
    export_model(
        model_path=args.model,
        output_dir=args.output_dir,
        version=args.version,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
