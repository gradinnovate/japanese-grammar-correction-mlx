#!/usr/bin/env python3
"""
Setup script for Japanese Grammar Correction project.
Verifies project structure and initializes necessary components.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.config import get_config
from utils.logging_utils import setup_logging, get_logger


def main():
    """Main setup function."""
    # Setup logging
    setup_logging(log_level="INFO")
    logger = get_logger(__name__)
    
    logger.info("Setting up Japanese Grammar Correction project...")
    
    # Initialize configuration
    config = get_config()
    
    # Validate setup
    if config.validate_setup():
        logger.info("✓ Project setup validation passed")
        config.print_config_summary()
    else:
        logger.error("✗ Project setup validation failed")
        return 1
    
    # Check corpus file
    corpus_path = config.get_corpus_path()
    if Path(corpus_path).exists():
        logger.info(f"✓ Found Japanese GEC corpus at: {corpus_path}")
    else:
        logger.warning(f"⚠ Japanese GEC corpus not found at: {corpus_path}")
        logger.info("Please ensure the corpus file is available before training")
    
    logger.info("Project setup completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())