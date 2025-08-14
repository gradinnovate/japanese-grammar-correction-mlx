"""
Training configuration for Japanese Grammar Correction system.
Contains all parameters and settings for MLX LoRA fine-tuning.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class TrainingConfig:
    """Configuration class for MLX LoRA training parameters."""
    
    # Model configuration
    model_name: str = "mlx-community/Qwen3-0.6B-4bit"
    model_path: Optional[str] = None
    
    # LoRA configuration
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: list = None
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 4
    num_epochs: int = 3
    max_seq_length: int = 512
    
    # Optimization settings
    warmup_steps: int = 100
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Scheduler settings
    lr_scheduler_type: str = "cosine"
    num_warmup_steps: int = 100
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Data settings
    train_data_path: str = "datasets/train.jsonl"
    valid_data_path: str = "datasets/valid.jsonl"
    test_data_path: str = "datasets/test.jsonl"
    
    # Output settings
    output_dir: str = "models/japanese-gec-lora"
    run_name: str = "japanese-gec-qwen3-lora"
    
    # Evaluation settings
    evaluation_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    # Generation settings for evaluation
    generation_max_length: int = 512
    generation_temperature: float = 0.1
    generation_top_p: float = 0.9
    generation_do_sample: bool = True
    
    def __post_init__(self):
        """Set default target modules if not specified."""
        if self.target_modules is None:
            # Target attention and feed-forward layers for Qwen3
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "max_seq_length": self.max_seq_length,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_grad_norm": self.max_grad_norm,
            "lr_scheduler_type": self.lr_scheduler_type,
            "num_warmup_steps": self.num_warmup_steps,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "save_total_limit": self.save_total_limit,
            "train_data_path": self.train_data_path,
            "valid_data_path": self.valid_data_path,
            "test_data_path": self.test_data_path,
            "output_dir": self.output_dir,
            "run_name": self.run_name,
            "evaluation_strategy": self.evaluation_strategy,
            "load_best_model_at_end": self.load_best_model_at_end,
            "metric_for_best_model": self.metric_for_best_model,
            "greater_is_better": self.greater_is_better,
            "generation_max_length": self.generation_max_length,
            "generation_temperature": self.generation_temperature,
            "generation_top_p": self.generation_top_p,
            "generation_do_sample": self.generation_do_sample,
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)


# Default configuration instance
DEFAULT_CONFIG = TrainingConfig()


def get_training_config(custom_config: Optional[Dict[str, Any]] = None) -> TrainingConfig:
    """
    Get training configuration with optional custom overrides.
    
    Args:
        custom_config: Dictionary of custom configuration values
        
    Returns:
        TrainingConfig instance
    """
    if custom_config is None:
        return DEFAULT_CONFIG
    
    # Start with default config and update with custom values
    config_dict = DEFAULT_CONFIG.to_dict()
    config_dict.update(custom_config)
    
    return TrainingConfig.from_dict(config_dict)