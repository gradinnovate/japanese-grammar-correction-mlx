#!/usr/bin/env python3
"""
Complete Training Pipeline for Japanese Grammar Correction

This script provides an end-to-end training pipeline that combines:
1. Data preprocessing from Japanese GEC corpus
2. MLX LoRA fine-tuning of Qwen3-0.6B-4bit model
3. Automatic evaluation after training completion

The pipeline supports different training configurations through command-line arguments
and provides comprehensive logging and progress monitoring.
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.logging_utils import setup_logging, get_logger
from utils.data_utils import extract_correction_pairs, create_training_prompt, calculate_dataset_stats
from utils.gec_parser import parse_gec_corpus
from utils.dataset_splitter import stratified_split


class TrainingPipeline:
    """Complete training pipeline for Japanese Grammar Correction."""
    
    def __init__(self, config_path: str, output_dir: str = "pipeline_output"):
        """
        Initialize the training pipeline.
        
        Args:
            config_path: Path to training configuration file
            output_dir: Directory for pipeline outputs
        """
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.logger = get_logger(__name__)
        self.config = None
        self.training_start_time = None
        
    def load_config(self) -> Dict[str, Any]:
        """Load and validate training configuration."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            self.logger.info(f"Loaded configuration from {self.config_path}")
            
            # Set default values for pipeline-specific settings
            config.setdefault('corpus_path', 'exclude/japanese_gec_corpus/corpus_v0.txt')
            config.setdefault('train_ratio', 0.8)
            config.setdefault('valid_ratio', 0.1)
            config.setdefault('test_ratio', 0.1)
            config.setdefault('min_samples', 100)
            config.setdefault('max_samples', None)
            config.setdefault('skip_preprocessing', False)
            config.setdefault('skip_training', False)
            config.setdefault('skip_evaluation', False)
            
            self.config = config
            return config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def validate_config(self) -> bool:
        """Validate the training configuration."""
        if not self.config:
            self.logger.error("Configuration not loaded")
            return False
        
        required_keys = [
            'model', 'lora_rank', 'lora_alpha', 'learning_rate', 
            'batch_size', 'iters', 'adapter_path'
        ]
        
        for key in required_keys:
            if key not in self.config:
                self.logger.error(f"Missing required configuration key: {key}")
                return False
        
        # Validate corpus file exists
        corpus_path = self.config['corpus_path']
        if not os.path.exists(corpus_path):
            self.logger.error(f"Corpus file not found: {corpus_path}")
            return False
        
        self.logger.info("Configuration validation passed")
        return True
    
    def preprocess_data(self) -> bool:
        """
        Preprocess the Japanese GEC corpus data.
        
        Returns:
            True if preprocessing successful, False otherwise
        """
        if self.config.get('skip_preprocessing', False):
            self.logger.info("Skipping data preprocessing (skip_preprocessing=True)")
            return self._validate_existing_datasets()
        
        self.logger.info("=== Starting Data Preprocessing ===")
        
        try:
            corpus_path = self.config['corpus_path']
            self.logger.info(f"Processing corpus: {corpus_path}")
            
            # Parse GEC corpus
            self.logger.info("Parsing GEC corpus...")
            correction_pairs = parse_gec_corpus(corpus_path)
            
            if not correction_pairs:
                self.logger.error("No correction pairs extracted from corpus")
                return False
            
            self.logger.info(f"Extracted {len(correction_pairs)} correction pairs")
            
            # Apply sample limits if specified
            max_samples = self.config.get('max_samples')
            if max_samples and len(correction_pairs) > max_samples:
                correction_pairs = correction_pairs[:max_samples]
                self.logger.info(f"Limited to {max_samples} samples")
            
            # Check minimum samples requirement
            min_samples = self.config.get('min_samples', 100)
            if len(correction_pairs) < min_samples:
                self.logger.error(f"Insufficient samples: {len(correction_pairs)} < {min_samples}")
                return False
            
            # Convert to training format
            self.logger.info("Converting to training format...")
            training_data = []
            for pair in correction_pairs:
                prompt_data = create_training_prompt(pair.error_text, pair.correct_text)
                training_data.append(prompt_data)
            
            # Split dataset
            self.logger.info("Splitting dataset...")
            train_ratio = self.config.get('train_ratio', 0.8)
            valid_ratio = self.config.get('valid_ratio', 0.1)
            test_ratio = self.config.get('test_ratio', 0.1)
            
            train_data, valid_data, test_data = stratified_split(
                training_data, train_ratio, valid_ratio, test_ratio
            )
            
            # Save datasets
            datasets_dir = Path("datasets")
            datasets_dir.mkdir(exist_ok=True)
            
            dataset_files = {
                'train': datasets_dir / "train.jsonl",
                'valid': datasets_dir / "valid.jsonl", 
                'test': datasets_dir / "test.jsonl"
            }
            
            for split_name, data in [('train', train_data), ('valid', valid_data), ('test', test_data)]:
                file_path = dataset_files[split_name]
                with open(file_path, 'w', encoding='utf-8') as f:
                    for item in data:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
                stats = calculate_dataset_stats(data)
                self.logger.info(f"{split_name.capitalize()} dataset: {stats['total_samples']} samples")
                self.logger.info(f"  Saved to: {file_path}")
            
            # Update config with dataset paths
            self.config['train'] = str(dataset_files['train'])
            self.config['valid'] = str(dataset_files['valid'])
            self.config['test'] = str(dataset_files['test'])
            
            self.logger.info("Data preprocessing completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Data preprocessing failed: {e}")
            return False
    
    def _validate_existing_datasets(self) -> bool:
        """Validate that existing dataset files are present and valid."""
        dataset_files = ['train', 'valid', 'test']
        
        for dataset in dataset_files:
            if dataset not in self.config:
                self.logger.error(f"Missing dataset path in config: {dataset}")
                return False
            
            file_path = self.config[dataset]
            if not os.path.exists(file_path):
                self.logger.error(f"Dataset file not found: {file_path}")
                return False
            
            # Check file is not empty
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    if not first_line.strip():
                        self.logger.error(f"Dataset file is empty: {file_path}")
                        return False
            except Exception as e:
                self.logger.error(f"Error reading dataset file {file_path}: {e}")
                return False
        
        self.logger.info("Existing datasets validated successfully")
        return True
    
    def run_training(self) -> bool:
        """
        Execute the MLX LoRA training process.
        
        Returns:
            True if training successful, False otherwise
        """
        if self.config.get('skip_training', False):
            self.logger.info("Skipping training (skip_training=True)")
            return self._validate_existing_model()
        
        self.logger.info("=== Starting Model Training ===")
        
        try:
            # Build MLX command
            cmd = self._build_mlx_command()
            self.logger.info(f"Training command: {' '.join(cmd)}")
            
            # Setup output directory
            adapter_path = self.config['adapter_path']
            os.makedirs(adapter_path, exist_ok=True)
            
            # Start training
            self.training_start_time = time.time()
            self.logger.info("Starting MLX LoRA training...")
            
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
                    self.logger.info(f"MLX: {line}")
                    
                    # Log important training metrics
                    if "Iter" in line and "Loss" in line:
                        self.logger.info(f"Training progress: {line}")
                    elif "Validation" in line:
                        self.logger.info(f"Validation: {line}")
                    elif "Saved" in line:
                        self.logger.info(f"Checkpoint: {line}")
            
            process.wait()
            training_duration = time.time() - self.training_start_time
            
            if process.returncode == 0:
                self.logger.info(f"Training completed successfully in {training_duration:.2f} seconds")
                self.logger.info(f"Model adapters saved to: {adapter_path}")
                return True
            else:
                self.logger.error(f"Training failed with return code: {process.returncode}")
                return False
                
        except Exception as e:
            self.logger.error(f"Training execution failed: {e}")
            return False
    
    def _build_mlx_command(self) -> List[str]:
        """Build the MLX LoRA training command."""
        config = self.config
        
        cmd = [
            "python", "-m", "mlx_lm.lora",
            "--model", config['model'],
            "--train", config['train'],
            "--valid", config['valid'],
            "--adapter-path", config['adapter_path'],
            "--lora-layers", str(config.get('lora_layers', 16)),
            "--lora-rank", str(config['lora_rank']),
            "--lora-alpha", str(config['lora_alpha']),
            "--learning-rate", str(config['learning_rate']),
            "--batch-size", str(config['batch_size']),
            "--iters", str(config['iters']),
            "--val-batches", str(config.get('val_batches', 25)),
            "--steps-per-report", str(config.get('steps_per_report', 10)),
            "--steps-per-eval", str(config.get('steps_per_eval', 200)),
            "--steps-per-save", str(config.get('steps_per_save', 400)),
            "--max-seq-len", str(config.get('max_seq_length', 512)),
            "--seed", str(config.get('seed', 42))
        ]
        
        # Add optional parameters
        if config.get('lora_dropout'):
            cmd.extend(["--lora-dropout", str(config['lora_dropout'])])
        
        if config.get('grad_checkpoint', False):
            cmd.append("--grad-checkpoint")
        
        if config.get('warmup_steps'):
            cmd.extend(["--warmup-steps", str(config['warmup_steps'])])
        
        if config.get('weight_decay'):
            cmd.extend(["--weight-decay", str(config['weight_decay'])])
        
        return cmd
    
    def _validate_existing_model(self) -> bool:
        """Validate that existing model adapters are present."""
        adapter_path = self.config['adapter_path']
        
        if not os.path.exists(adapter_path):
            self.logger.error(f"Adapter path not found: {adapter_path}")
            return False
        
        # Check for adapter files
        adapter_file = os.path.join(adapter_path, "adapters.safetensors")
        config_file = os.path.join(adapter_path, "adapter_config.json")
        
        if not os.path.exists(adapter_file):
            self.logger.error(f"Adapter file not found: {adapter_file}")
            return False
        
        if not os.path.exists(config_file):
            self.logger.error(f"Adapter config not found: {config_file}")
            return False
        
        self.logger.info("Existing model adapters validated successfully")
        return True
    
    def run_evaluation(self) -> bool:
        """
        Run automatic evaluation after training completion.
        
        Returns:
            True if evaluation successful, False otherwise
        """
        if self.config.get('skip_evaluation', False):
            self.logger.info("Skipping evaluation (skip_evaluation=True)")
            return True
        
        self.logger.info("=== Starting Model Evaluation ===")
        
        try:
            # First run batch inference on test set
            self.logger.info("Running batch inference on test set...")
            inference_results_file = self.output_dir / "test_predictions.jsonl"
            
            if not self._run_batch_inference(inference_results_file):
                self.logger.error("Batch inference failed")
                return False
            
            # Then run evaluation on the results
            self.logger.info("Calculating evaluation metrics...")
            evaluation_report_file = self.output_dir / "evaluation_report.json"
            
            if not self._run_evaluation_script(inference_results_file, evaluation_report_file):
                self.logger.error("Evaluation script failed")
                return False
            
            # Log evaluation summary
            self._log_evaluation_summary(evaluation_report_file)
            
            self.logger.info("Evaluation completed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Evaluation failed: {e}")
            return False
    
    def _run_batch_inference(self, output_file: Path) -> bool:
        """Run batch inference on test dataset."""
        try:
            cmd = [
                "python", "scripts/batch_inference.py",
                "--model", self.config['model'],
                "--adapter-path", self.config['adapter_path'],
                "--input-file", self.config['test'],
                "--output-file", str(output_file),
                "--batch-size", str(self.config.get('batch_size', 4)),
                "--max-tokens", str(self.config.get('max_tokens', 512)),
                "--temperature", str(self.config.get('temp', 0.1)),
                "--top-p", str(self.config.get('top_p', 0.9))
            ]
            
            self.logger.info(f"Batch inference command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Batch inference completed successfully")
                return True
            else:
                self.logger.error(f"Batch inference failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Batch inference execution failed: {e}")
            return False
    
    def _run_evaluation_script(self, results_file: Path, output_file: Path) -> bool:
        """Run evaluation script on inference results."""
        try:
            cmd = [
                "python", "scripts/evaluate_gec.py",
                "--results-file", str(results_file),
                "--output-file", str(output_file)
            ]
            
            self.logger.info(f"Evaluation command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Evaluation script completed successfully")
                return True
            else:
                self.logger.error(f"Evaluation script failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Evaluation script execution failed: {e}")
            return False
    
    def _log_evaluation_summary(self, report_file: Path) -> None:
        """Log evaluation summary from report file."""
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report = json.load(f)
            
            self.logger.info("=== EVALUATION SUMMARY ===")
            
            # Extract key metrics
            sentence_metrics = report.get('sentence_level_metrics', {})
            token_metrics = report.get('token_level_metrics', {})
            fluency_metrics = report.get('fluency_metrics', {})
            edit_metrics = report.get('edit_distance_metrics', {})
            
            self.logger.info(f"Total Examples: {report.get('evaluation_summary', {}).get('total_examples', 'N/A')}")
            self.logger.info(f"Sentence Accuracy: {sentence_metrics.get('exact_match_accuracy', 'N/A'):.4f}")
            self.logger.info(f"Token F1 Score: {token_metrics.get('f1_score', 'N/A'):.4f}")
            self.logger.info(f"BLEU Score: {fluency_metrics.get('bleu_score', 'N/A'):.4f}")
            self.logger.info(f"Edit Accuracy: {edit_metrics.get('edit_accuracy', 'N/A'):.4f}")
            self.logger.info("=" * 27)
            
        except Exception as e:
            self.logger.warning(f"Could not log evaluation summary: {e}")
    
    def run_complete_pipeline(self) -> bool:
        """
        Run the complete training pipeline.
        
        Returns:
            True if entire pipeline successful, False otherwise
        """
        pipeline_start_time = time.time()
        
        self.logger.info("=== STARTING COMPLETE TRAINING PIPELINE ===")
        self.logger.info(f"Configuration: {self.config_path}")
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info("=" * 50)
        
        try:
            # Step 1: Load and validate configuration
            if not self.load_config():
                return False
            
            if not self.validate_config():
                return False
            
            # Step 2: Data preprocessing
            if not self.preprocess_data():
                return False
            
            # Step 3: Model training
            if not self.run_training():
                return False
            
            # Step 4: Evaluation
            if not self.run_evaluation():
                return False
            
            # Pipeline completed successfully
            pipeline_duration = time.time() - pipeline_start_time
            self.logger.info("=== PIPELINE COMPLETED SUCCESSFULLY ===")
            self.logger.info(f"Total pipeline duration: {pipeline_duration:.2f} seconds")
            self.logger.info(f"Output directory: {self.output_dir}")
            self.logger.info("=" * 40)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return False


def main():
    """Main pipeline script entry point."""
    parser = argparse.ArgumentParser(description="Complete Japanese Grammar Correction Training Pipeline")
    
    # Configuration arguments
    parser.add_argument(
        "--config",
        default="config/lora_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--output-dir",
        default="pipeline_output",
        help="Directory for pipeline outputs"
    )
    
    # Pipeline control arguments
    parser.add_argument(
        "--skip-preprocessing",
        action="store_true",
        help="Skip data preprocessing step"
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip model training step"
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation step"
    )
    
    # Data configuration arguments
    parser.add_argument(
        "--corpus-path",
        default="exclude/japanese_gec_corpus/corpus_v0.txt",
        help="Path to Japanese GEC corpus file"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum number of samples to use (for testing)"
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Training set ratio"
    )
    parser.add_argument(
        "--valid-ratio",
        type=float,
        default=0.1,
        help="Validation set ratio"
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test set ratio"
    )
    
    # Logging arguments
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        default="logs/pipeline.log",
        help="Path to log file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level, log_file=args.log_file)
    logger = get_logger(__name__)
    
    try:
        # Create pipeline instance
        pipeline = TrainingPipeline(args.config, args.output_dir)
        
        # Override config with command line arguments
        if hasattr(pipeline, 'config') and pipeline.config:
            config = pipeline.config
        else:
            config = {}
        
        config['skip_preprocessing'] = args.skip_preprocessing
        config['skip_training'] = args.skip_training
        config['skip_evaluation'] = args.skip_evaluation
        
        # Update pipeline config
        pipeline.config = config
        config['corpus_path'] = args.corpus_path
        config['train_ratio'] = args.train_ratio
        config['valid_ratio'] = args.valid_ratio
        config['test_ratio'] = args.test_ratio
        
        if args.max_samples:
            config['max_samples'] = args.max_samples
        
        # Run complete pipeline
        success = pipeline.run_complete_pipeline()
        
        if success:
            logger.info("Pipeline completed successfully!")
            sys.exit(0)
        else:
            logger.error("Pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline script failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()