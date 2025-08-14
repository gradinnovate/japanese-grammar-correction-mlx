"""
Path configuration for Japanese Grammar Correction system.
Centralized path management for all data, models, and output files.
"""

import os
from pathlib import Path


class ProjectPaths:
    """Centralized path configuration for the project."""
    
    # Project root directory
    ROOT_DIR = Path(__file__).parent.parent
    
    # Data directories
    DATA_DIR = ROOT_DIR / "datasets"
    CORPUS_DIR = ROOT_DIR / "exclude" / "japanese_gec_corpus"
    
    # Model directories
    MODELS_DIR = ROOT_DIR / "models"
    ADAPTERS_DIR = MODELS_DIR / "adapters"
    
    # Configuration directories
    CONFIG_DIR = ROOT_DIR / "config"
    
    # Script directories
    SCRIPTS_DIR = ROOT_DIR / "scripts"
    
    # Utility directories
    UTILS_DIR = ROOT_DIR / "utils"
    
    # Corpus files
    CORPUS_FILE = CORPUS_DIR / "corpus_v0.txt"
    TAGS_FILE = CORPUS_DIR / "tags_v0.txt"
    
    # Dataset files
    TRAIN_DATA = DATA_DIR / "train.jsonl"
    VALID_DATA = DATA_DIR / "valid.jsonl"
    TEST_DATA = DATA_DIR / "test.jsonl"
    RAW_PAIRS = DATA_DIR / "raw_pairs.jsonl"
    
    # Configuration files
    LORA_CONFIG = CONFIG_DIR / "lora_config.yaml"
    
    # Model files
    BASE_MODEL_PATH = "mlx-community/Qwen3-0.6B-4bit"
    LORA_ADAPTERS = ADAPTERS_DIR / "japanese-gec-lora"
    
    # Output directories
    LOGS_DIR = ROOT_DIR / "logs"
    RESULTS_DIR = ROOT_DIR / "results"
    
    @classmethod
    def ensure_directories(cls) -> None:
        """Create all necessary directories if they don't exist."""
        directories = [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.ADAPTERS_DIR,
            cls.LOGS_DIR,
            cls.RESULTS_DIR,
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_corpus_path(cls) -> str:
        """Get the path to the Japanese GEC corpus file."""
        return str(cls.CORPUS_FILE)
    
    @classmethod
    def get_train_data_path(cls) -> str:
        """Get the path to the training data file."""
        return str(cls.TRAIN_DATA)
    
    @classmethod
    def get_valid_data_path(cls) -> str:
        """Get the path to the validation data file."""
        return str(cls.VALID_DATA)
    
    @classmethod
    def get_test_data_path(cls) -> str:
        """Get the path to the test data file."""
        return str(cls.TEST_DATA)
    
    @classmethod
    def get_lora_config_path(cls) -> str:
        """Get the path to the LoRA configuration file."""
        return str(cls.LORA_CONFIG)
    
    @classmethod
    def get_adapters_path(cls) -> str:
        """Get the path to save LoRA adapters."""
        return str(cls.LORA_ADAPTERS)
    
    @classmethod
    def validate_corpus_exists(cls) -> bool:
        """Check if the Japanese GEC corpus file exists."""
        return cls.CORPUS_FILE.exists()
    
    @classmethod
    def get_relative_path(cls, path: Path) -> str:
        """Get relative path from project root."""
        try:
            return str(path.relative_to(cls.ROOT_DIR))
        except ValueError:
            return str(path)