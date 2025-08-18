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

# MLflow imports
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    logging.warning("MLflow not available. Install with: pip install mlflow")


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
    """Validate the training configuration with detection-specific checks."""
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
    
    # Detection-specific validation warnings
    if 'gec_error_detection' in data_dir:
        logging.info("🎯 Detection training mode detected")
        
        # Check temperature (convert to float if needed)
        temp = config.get('temperature', 0.1)
        try:
            temp_val = float(temp)
            if temp_val > 0.1:
                logging.warning("⚠️  High temperature may affect detection precision")
        except (ValueError, TypeError):
            logging.warning("⚠️  Invalid temperature value")
        
        # Check mask_prompt
        if config.get('mask_prompt', True):
            logging.warning("⚠️  mask_prompt=True may hurt detection performance")
        
        # Check learning_rate (convert to float if needed)
        lr = config.get('learning_rate', 1e-5)
        try:
            lr_val = float(lr)
            if lr_val < 1e-5:
                logging.warning("⚠️  Very low learning rate may slow detection pattern learning")
        except (ValueError, TypeError):
            logging.warning("⚠️  Invalid learning rate value")
    
    logging.info("Configuration validation passed")
    return True


def setup_output_directory(adapter_path: str) -> None:
    """Create output directory for model adapters."""
    os.makedirs(adapter_path, exist_ok=True)
    logging.info(f"Output directory prepared: {adapter_path}")


def setup_mlflow(config: Dict[str, Any]) -> bool:
    """Setup MLflow tracking."""
    if not MLFLOW_AVAILABLE:
        return False
    
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://192.168.68.112:5001")
        
        # Set experiment name
        experiment_name = f"japanese-gec-{config.get('model', 'unknown').split('/')[-1]}"
        mlflow.set_experiment(experiment_name)
        
        # Start MLflow run
        mlflow.start_run()
        
        # Log configuration parameters
        mlflow.log_params({
            "model": config.get('model'),
            "data_dir": config.get('data_dir'),
            "num_epochs": config.get('num_epochs'),
            "batch_size": config.get('batch_size'),
            "learning_rate": config.get('learning_rate'),
            "lora_rank": config.get('lora_parameters', {}).get('rank'),
            "lora_scale": config.get('lora_parameters', {}).get('scale'),
            "lora_dropout": config.get('lora_parameters', {}).get('dropout'),
            "max_seq_length": config.get('max_seq_length'),
            "fine_tune_type": config.get('fine_tune_type'),
            "mask_prompt": config.get('mask_prompt'),
        })
        
        logging.info(f"MLflow tracking started: {mlflow.get_tracking_uri()}")
        logging.info(f"Experiment: {experiment_name}")
        logging.info(f"Run ID: {mlflow.active_run().info.run_id}")
        return True
        
    except Exception as e:
        logging.warning(f"Failed to setup MLflow: {e}")
        return False


def calculate_iters_from_epochs(config: Dict[str, Any]) -> int:
    """Calculate training iterations from num_epochs and training data size."""
    from pathlib import Path
    
    data_dir = config.get('data_dir', 'datasets/combined')
    train_file = Path(data_dir) / 'train.jsonl'
    
    if train_file.exists():
        with open(train_file, 'r', encoding='utf-8') as f:
            train_samples = sum(1 for _ in f)
        batch_size = config.get('batch_size', 4)
        num_epochs = config.get('num_epochs', 3)
        iters = int((num_epochs * train_samples + batch_size - 1) // batch_size)  # Convert to int
        logging.info(f"Calculated iters: {iters} (epochs: {num_epochs}, samples: {train_samples}, batch_size: {batch_size})")
        
        # Log to MLflow
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_metrics({
                "train_samples": train_samples,
                "calculated_iters": iters
            })
        
        return iters
    else:
        logging.warning(f"Train file not found: {train_file}, using default iters=1000")
        return 1000


def create_mlx_config_file(config: Dict[str, Any], config_path: str, resume_from: str = None) -> None:
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
        "iters": calculate_iters_from_epochs(config),
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
    
    # Add resume capability if specified
    if resume_from:
        mlx_config["resume_adapter_file"] = resume_from
        logging.info(f"🔄 Resuming training from: {resume_from}")
    
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


def build_mlx_command(config: Dict[str, Any], resume_from: str = None) -> Tuple[list, str]:
    """Build the MLX LoRA training command with config file."""
    # Create temporary config file
    config_path = f"{config['adapter_path']}_mlx_config.yaml"
    create_mlx_config_file(config, config_path, resume_from)
    
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


def run_training(config: Dict[str, Any], dry_run: bool = False, resume_from: str = None) -> bool:
    """Execute the MLX LoRA training process."""
    # Validate resume checkpoint if specified
    if resume_from:
        if not os.path.exists(resume_from):
            logging.error(f"Resume checkpoint not found: {resume_from}")
            return False
        logging.info(f"🔄 Will resume training from checkpoint: {resume_from}")
    
    cmd, config_path = build_mlx_command(config, resume_from)
    
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
        
        # Monitor training output with detection-specific focus
        best_val_loss = float('inf')
        detection_mode = 'gec_error_detection' in config.get('data_dir', '')
        
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if line:
                logging.info(f"MLX: {line}")
                
                # Enhanced logging for detection training
                if "Iter" in line and "Train loss" in line:
                    if detection_mode:
                        logging.info(f"🎯 Detection training progress: {line}")
                    else:
                        logging.info(f"Training progress: {line}")
                    
                    # Parse and log training loss to MLflow
                    try:
                        # Parse line like "Iter 100: Train loss 2.456, Learning Rate 1.000e-05"
                        import re
                        iter_match = re.search(r'Iter (\d+)', line)
                        loss_match = re.search(r'Train loss ([\d.]+)', line)
                        lr_match = re.search(r'Learning Rate ([\d.e-]+)', line)
                        
                        if iter_match and loss_match and MLFLOW_AVAILABLE and mlflow.active_run():
                            iteration = int(iter_match.group(1))
                            train_loss = float(loss_match.group(1))
                            
                            mlflow.log_metrics({
                                "train_loss": train_loss,
                                "iteration": iteration
                            }, step=iteration)
                            
                            if lr_match:
                                learning_rate = float(lr_match.group(1))
                                mlflow.log_metric("learning_rate", learning_rate, step=iteration)
                                
                    except (ValueError, AttributeError) as e:
                        logging.debug(f"Could not parse training metrics: {e}")
                
                elif "Validation" in line or "Val" in line:
                    if detection_mode:
                        logging.info(f"🎯 Detection validation: {line}")
                    else:
                        logging.info(f"Validation: {line}")
                        
                    # Parse and log validation loss to MLflow
                    try:
                        # Parse line like "Iter 200: Val loss 2.123, Val ppl 8.456"
                        import re
                        iter_match = re.search(r'Iter (\d+)', line)
                        val_loss_match = re.search(r'Val loss ([\d.]+)', line)
                        val_ppl_match = re.search(r'Val ppl ([\d.]+)', line)
                        
                        if iter_match and val_loss_match and MLFLOW_AVAILABLE and mlflow.active_run():
                            iteration = int(iter_match.group(1))
                            val_loss = float(val_loss_match.group(1))
                            
                            mlflow.log_metrics({
                                "val_loss": val_loss,
                                "validation_iteration": iteration
                            }, step=iteration)
                            
                            if val_ppl_match:
                                val_ppl = float(val_ppl_match.group(1))
                                mlflow.log_metric("val_perplexity", val_ppl, step=iteration)
                        
                        # Track best validation loss for detection
                        try:
                            parts = line.split()
                            for i, part in enumerate(parts):
                                if "loss" in part.lower() and i + 1 < len(parts):
                                    try:
                                        val_loss = float(parts[i + 1])
                                        if val_loss < best_val_loss:
                                            best_val_loss = val_loss
                                            logging.info(f"✨ New best detection validation loss: {val_loss:.4f}")
                                            
                                            # Log best validation loss to MLflow
                                            if MLFLOW_AVAILABLE and mlflow.active_run():
                                                mlflow.log_metric("best_val_loss", best_val_loss)
                                        break
                                    except ValueError:
                                        continue
                        except Exception:
                            pass
                    except (ValueError, AttributeError) as e:
                        logging.debug(f"Could not parse validation metrics: {e}")
                
                elif "Saved" in line:
                    if detection_mode:
                        logging.info(f"💾 Detection checkpoint: {line}")
                    else:
                        logging.info(f"Checkpoint: {line}")
        
        process.wait()
        end_time = time.time()
        
        if process.returncode == 0:
            duration = end_time - start_time
            
            # Log final training metrics to MLflow
            if MLFLOW_AVAILABLE and mlflow.active_run():
                metrics_to_log = {
                    "training_duration_seconds": duration
                }
                
                # Only log validation loss if it's valid
                if best_val_loss != float('inf'):
                    metrics_to_log["final_best_val_loss"] = best_val_loss
                
                mlflow.log_metrics(metrics_to_log)
            
            if detection_mode:
                logging.info(f"🎉 Detection training completed successfully in {duration:.2f} seconds")
                logging.info(f"🎯 Detection adapters saved to: {config['adapter_path']}")
                logging.info(f"📊 Best validation loss: {best_val_loss:.4f}")
                logging.info("💡 Next steps:")
                logging.info("   1. Run detection evaluation: python scripts/grammar_focused_evaluation.py --task-filter DETECT")
                logging.info("   2. Test on sample sentences for quality assessment")
            else:
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
    parser.add_argument(
        "--resume-from",
        type=str,
        help="Path to existing adapter to resume training from"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level, log_file=args.log_file)
    
    try:
        # Load and validate configuration
        config = load_config(args.config)
        if not validate_config(config):
            sys.exit(1)
        
        # Setup MLflow tracking
        mlflow_enabled = setup_mlflow(config) if not args.dry_run else False
        if mlflow_enabled:
            logging.info("MLflow tracking enabled")
        
        # Log training configuration
        logging.info("=== Japanese GEC Training Configuration ===")
        for key, value in config.items():
            logging.info(f"{key}: {value}")
        logging.info("=" * 45)
        
        # Run training
        success = run_training(config, dry_run=args.dry_run, resume_from=args.resume_from)
        
        if success:
            logging.info("Training completed successfully!")
            if not args.dry_run:
                logging.info(f"Fine-tuned adapters available at: {config['adapter_path']}")
                
                # Log success to MLflow
                if mlflow_enabled:
                    mlflow.log_metric("training_success", 1.0)
        else:
            logging.error("Training failed!")
            # Log failure to MLflow
            if mlflow_enabled:
                mlflow.log_metric("training_success", 0.0)
            sys.exit(1)
            
    except KeyboardInterrupt:
        logging.info("Training interrupted by user")
        # Log interruption to MLflow
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_metric("training_interrupted", 1.0)
            mlflow.end_run(status="KILLED")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Training script failed: {e}")
        # Log error to MLflow
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.log_metric("training_error", 1.0)
            mlflow.end_run(status="FAILED")
        sys.exit(1)
    finally:
        # End MLflow run if active
        if MLFLOW_AVAILABLE and mlflow.active_run():
            mlflow.end_run()


if __name__ == "__main__":
    main()