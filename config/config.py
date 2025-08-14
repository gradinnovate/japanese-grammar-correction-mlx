"""
Main configuration module for Japanese Grammar Correction system.
Provides centralized access to all configuration settings.
"""

import os
import yaml
from typing import Dict, Any, Optional
from .training_config import TrainingConfig, get_training_config
from .paths import ProjectPaths


class Config:
    """Main configuration class that combines all configuration sources."""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration.
        
        Args:
            config_file: Optional path to YAML configuration file
        """
        self.paths = ProjectPaths()
        self.training = get_training_config()
        
        # Load additional configuration from file if provided
        if config_file and os.path.exists(config_file):
            self._load_config_file(config_file)
        
        # Ensure necessary directories exist
        self.paths.ensure_directories()
    
    def _load_config_file(self, config_file: str) -> None:
        """Load configuration from YAML file."""
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
            
            # Update training configuration if present
            if 'training' in config_data:
                self.training = get_training_config(config_data['training'])
                
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    def get_corpus_path(self) -> str:
        """Get path to the Japanese GEC corpus."""
        return self.paths.get_corpus_path()
    
    def get_data_paths(self) -> Dict[str, str]:
        """Get all dataset paths."""
        return {
            'train': self.paths.get_train_data_path(),
            'valid': self.paths.get_valid_data_path(),
            'test': self.paths.get_test_data_path(),
        }
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'base_model': self.training.model_name,
            'adapters_path': self.paths.get_adapters_path(),
            'lora_rank': self.training.lora_rank,
            'lora_alpha': self.training.lora_alpha,
        }
    
    def get_training_args(self) -> Dict[str, Any]:
        """Get training arguments for MLX LoRA."""
        return {
            'model': self.training.model_name,
            'train': self.paths.get_train_data_path(),
            'valid': self.paths.get_valid_data_path(),
            'adapter_path': self.paths.get_adapters_path(),
            'batch_size': self.training.batch_size,
            'lora_layers': self.training.lora_rank,
            'learning_rate': self.training.learning_rate,
            'iters': self.training.num_epochs * 1000,  # Approximate iterations
            'save_every': self.training.save_steps,
            'val_batches': 25,
            'steps_per_report': self.training.logging_steps,
            'steps_per_eval': self.training.eval_steps,
        }
    
    def validate_setup(self) -> bool:
        """
        Validate that the system is properly configured.
        
        Returns:
            True if configuration is valid, False otherwise
        """
        # Check if corpus file exists
        if not self.paths.validate_corpus_exists():
            print(f"Error: Corpus file not found at {self.paths.get_corpus_path()}")
            return False
        
        # Check if required directories exist
        required_dirs = [
            self.paths.DATA_DIR,
            self.paths.MODELS_DIR,
            self.paths.CONFIG_DIR,
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                print(f"Error: Required directory not found: {directory}")
                return False
        
        return True
    
    def print_config_summary(self) -> None:
        """Print a summary of the current configuration."""
        print("=== Japanese Grammar Correction Configuration ===")
        print(f"Corpus path: {self.get_corpus_path()}")
        print(f"Base model: {self.training.model_name}")
        print(f"LoRA rank: {self.training.lora_rank}")
        print(f"Batch size: {self.training.batch_size}")
        print(f"Learning rate: {self.training.learning_rate}")
        print(f"Output directory: {self.paths.get_adapters_path()}")
        print("=" * 50)


# Global configuration instance
_config_instance = None


def get_config(config_file: Optional[str] = None) -> Config:
    """
    Get the global configuration instance.
    
    Args:
        config_file: Optional path to configuration file
        
    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_file)
    return _config_instance


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config_instance
    _config_instance = None