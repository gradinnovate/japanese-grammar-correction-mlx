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
from typing import Dict, Any, Tuple

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
        'model', 'learning_rate', 'batch_size', 'adapter_path'
    ]
    
    for key in required_keys:
        if key not in config:
            logging.error(f"Missing required configuration key: {key}")
            return False
    
    # Validate LoRA parameters if present
    if 'lora_parameters' in config:
        lora_params = config['lora_parameters']
        required_lora_keys = ['rank', 'scale', 'dropout']
        for key in required_lora_keys:
            if key not in lora_params:
                logging.error(f"Missing required LoRA parameter: {key}")
                return False
    
    # Validate data directory exists and contains required files
    data_dir = config.get('data_dir', 'datasets/combined')
    if not os.path.exists(data_dir):
        logging.error(f"Data directory not found: {data_dir}")
        return False
    
    # Check for required files in data directory
    required_files = ['train.jsonl', 'valid.jsonl']
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            logging.error(f"Required data file not found: {filepath}")
            return False
    
    logging.info("Configuration validation passed")
    return True


def setup_output_directory(adapter_path: str) -> None:
    """Create output directory for model adapters."""
    os.makedirs(adapter_path, exist_ok=True)
    logging.info(f"Output directory prepared: {adapter_path}")


def create_mlx_config_file(config: Dict[str, Any], config_path: str) -> None:
    """Create MLX LoRA configuration file with proper LoRA parameters."""
    import yaml
    
    mlx_config = {
        "model": config['model'],
        "train": True,
        "data": config.get('data_dir', 'datasets/combined'),
        "adapter_path": config['adapter_path'],
        "num_layers": config.get('num_layers', 16),
        "learning_rate": config['learning_rate'],
        "batch_size": config['batch_size'],
        "iters": config['iters'],
        "val_batches": config['val_batches'],
        "steps_per_report": config['steps_per_report'],
        "steps_per_eval": config['steps_per_eval'],
        "save_every": config.get('steps_per_save', 100),
        "max_seq_length": config.get('max_seq_length', 512),
        "seed": config.get('seed', 42),
        "fine_tune_type": config.get('fine_tune_type', 'lora'),
        "lora_parameters": config.get('lora_parameters', {
            "rank": 16,
            "dropout": 0.1,
            "scale": 32
        })
    }
    
    # Add optional advanced parameters
    if config.get('grad_checkpoint', False):
        mlx_config["grad_checkpoint"] = True
    
    if config.get('optimizer'):
        mlx_config["optimizer"] = config['optimizer']
    
    if config.get('weight_decay'):
        mlx_config["optimizer_config"] = {
            config.get('optimizer', 'adam'): {
                "weight_decay": config['weight_decay']
            }
        }
    
    if config.get('lr_schedule'):
        mlx_config["lr_schedule"] = config['lr_schedule']
    

    
    # Write config file
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(mlx_config, f, default_flow_style=False)
    
    logging.info(f"Created MLX config file: {config_path}")


def build_mlx_command(config: Dict[str, Any]) -> Tuple[list, str]:
    """Build the MLX LoRA training command with config file."""
    # Create temporary config file
    config_path = f"{config['adapter_path']}_mlx_config.yaml"
    create_mlx_config_file(config, config_path)
    
    cmd = [
        "python", "-m", "mlx_lm", "lora",
        "--config", config_path
    ]
    
    return cmd, config_path


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
    cmd, config_path = build_mlx_command(config)
    
    logging.info("Starting MLX LoRA training...")
    logging.info(f"Command: {' '.join(cmd)}")
    logging.info(f"Using config file: {config_path}")
    
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