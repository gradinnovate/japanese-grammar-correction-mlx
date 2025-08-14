#!/usr/bin/env python3
"""
Japanese Grammar Correction Training Script

This script handles the MLX LoRA fine-tuning process for Japanese GEC using the Qwen3-0.6B-4bit model.
It provides training progress monitoring, logging, and model output management.
"""

import argparse
import logging
import os
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.logging_utils import setup_logging


def load_config(config_path: str) -> Dict[str, Any]:
    """Load training configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate the training configuration."""
    required_keys = [
        'model', 'lora_rank', 'lora_alpha', 'learning_rate', 
        'batch_size', 'train', 'valid', 'adapter_path'
    ]
    
    for key in required_keys:
        if key not in config:
            logging.error(f"Missing required configuration key: {key}")
            return False
    
    # Validate data files exist
    for data_key in ['train', 'valid']:
        if not os.path.exists(config[data_key]):
            logging.error(f"Data file not found: {config[data_key]}")
            return False
    
    logging.info("Configuration validation passed")
    return True


def setup_output_directory(adapter_path: str) -> None:
    """Create output directory for model adapters."""
    os.makedirs(adapter_path, exist_ok=True)
    logging.info(f"Output directory prepared: {adapter_path}")


def build_mlx_command(config: Dict[str, Any]) -> list:
    """Build the MLX LoRA training command."""
    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--model", config['model'],
        "--train",
        "--data", "datasets",  # Directory containing train.jsonl, valid.jsonl
        "--adapter-path", config['adapter_path'],
        "--num-layers", str(config['lora_layers']),
        "--learning-rate", str(config['learning_rate']),
        "--batch-size", str(config['batch_size']),
        "--iters", str(config['iters']),
        "--val-batches", str(config['val_batches']),
        "--steps-per-report", str(config['steps_per_report']),
        "--steps-per-eval", str(config['steps_per_eval']),
        "--save-every", str(config['steps_per_save']),
        "--max-seq-length", str(config.get('max_seq_length', 512)),
        "--seed", str(config.get('seed', 42))
    ]
    
    # Add optional parameters
    if config.get('grad_checkpoint', False):
        cmd.append("--grad-checkpoint")
    
    return cmd


def monitor_training_progress(log_file: str) -> None:
    """Monitor training progress from log file."""
    if not os.path.exists(log_file):
        return
    
    try:
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                if "Iter" in last_line and "Loss" in last_line:
                    logging.info(f"Training progress: {last_line}")
    except Exception as e:
        logging.warning(f"Could not read training log: {e}")


def run_training(config: Dict[str, Any], dry_run: bool = False) -> bool:
    """Execute the MLX LoRA training process."""
    cmd = build_mlx_command(config)
    
    logging.info("Starting MLX LoRA training...")
    logging.info(f"Command: {' '.join(cmd)}")
    
    if dry_run:
        logging.info("Dry run mode - command would be executed but not actually run")
        return True
    
    try:
        import subprocess
        
        # Setup output directory
        setup_output_directory(config['adapter_path'])
        
        # Start training process
        start_time = time.time()
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Monitor training output
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                logging.info(f"MLX: {line}")
                
                # Log important training metrics
                if "Iter" in line and "Loss" in line:
                    logging.info(f"Training progress: {line}")
                elif "Validation" in line:
                    logging.info(f"Validation: {line}")
                elif "Saved" in line:
                    logging.info(f"Checkpoint: {line}")
        
        process.wait()
        end_time = time.time()
        
        if process.returncode == 0:
            duration = end_time - start_time
            logging.info(f"Training completed successfully in {duration:.2f} seconds")
            logging.info(f"Model adapters saved to: {config['adapter_path']}")
            return True
        else:
            logging.error(f"Training failed with return code: {process.returncode}")
            return False
            
    except Exception as e:
        logging.error(f"Training execution failed: {e}")
        return False


def main():
    """Main training script entry point."""
    parser = argparse.ArgumentParser(description="Train Japanese Grammar Correction model using MLX LoRA")
    parser.add_argument(
        "--config", 
        default="config/lora_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        default="logs/training.log",
        help="Path to log file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show command that would be executed without running it"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level, log_file=args.log_file)
    
    try:
        # Load and validate configuration
        config = load_config(args.config)
        if not validate_config(config):
            sys.exit(1)
        
        # Log training configuration
        logging.info("=== Japanese GEC Training Configuration ===")
        for key, value in config.items():
            logging.info(f"{key}: {value}")
        logging.info("=" * 45)
        
        # Run training
        success = run_training(config, dry_run=args.dry_run)
        
        if success:
            logging.info("Training completed successfully!")
            if not args.dry_run:
                logging.info(f"Fine-tuned adapters available at: {config['adapter_path']}")
        else:
            logging.error("Training failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Training script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()