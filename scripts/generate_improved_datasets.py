#!/usr/bin/env python3
"""
Generate Improved Datasets for Japanese Grammar Correction

This script regenerates datasets with improved prompt formats and better data quality.
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.data_utils import create_training_prompt, validate_gec_pair, clean_japanese_text
from utils.gec_parser import parse_gec_corpus, GECPair
from utils.dataset_splitter import stratified_split
from utils.logging_utils import setup_logging


def create_improved_dataset_entry(error_text: str, correct_text: str, format_version: str = "v2", use_english: bool = False) -> Dict[str, Any]:
    """Create an improved dataset entry with better prompt format."""
    # Clean texts
    error_text = clean_japanese_text(error_text)
    correct_text = clean_japanese_text(correct_text)
    
    # Validate pair
    if not validate_gec_pair(error_text, correct_text):
        return None
    
    # Create training prompt with improved format using global constants
    return create_training_prompt(error_text, correct_text, format_version, use_english)


def generate_improved_datasets(
    corpus_path: str,
    output_dir: str = "datasets",
    format_version: str = "v2",
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    max_samples: int = None,
    use_english: bool = False
):
    """
    Generate improved datasets with better prompt formats.
    
    Args:
        corpus_path: Path to GEC corpus file
        output_dir: Output directory for datasets
        format_version: Prompt format version ("v1", "v2", "messages")
        train_ratio: Training set ratio
        valid_ratio: Validation set ratio  
        test_ratio: Test set ratio
        max_samples: Maximum number of samples to process
    """
    logger = logging.getLogger(__name__)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Parsing GEC corpus from {corpus_path}")
    
    # Parse corpus
    try:
        gec_pairs = parse_gec_corpus(corpus_path)
    except FileNotFoundError:
        logger.error(f"Corpus file not found: {corpus_path}")
        return False
    
    if not gec_pairs:
        logger.error("No valid GEC pairs found in corpus")
        return False
    
    logger.info(f"Found {len(gec_pairs)} GEC pairs")
    
    # Limit samples if specified
    if max_samples and len(gec_pairs) > max_samples:
        gec_pairs = gec_pairs[:max_samples]
        logger.info(f"Limited to {max_samples} samples")
    
    # Convert to training format
    logger.info(f"Converting to training format (version: {format_version})")
    dataset_entries = []
    skipped = 0
    
    for pair in gec_pairs:
        entry = create_improved_dataset_entry(
            pair.error_text, 
            pair.correct_text, 
            format_version,
            use_english=use_english
        )
        
        if entry:
            dataset_entries.append(entry)
        else:
            skipped += 1
    
    logger.info(f"Created {len(dataset_entries)} dataset entries")
    if skipped > 0:
        logger.info(f"Skipped {skipped} invalid pairs")
    
    if not dataset_entries:
        logger.error("No valid dataset entries created")
        return False
    
    # Split dataset
    logger.info("Splitting dataset")
    train_data, valid_data, test_data = stratified_split(
        dataset_entries, 
        train_ratio, 
        valid_ratio, 
        test_ratio
    )
    
    logger.info(f"Dataset split: train={len(train_data)}, valid={len(valid_data)}, test={len(test_data)}")
    
    # Save datasets
    datasets = {
        "train": train_data,
        "valid": valid_data, 
        "test": test_data
    }
    
    for split_name, data in datasets.items():
        output_path = os.path.join(output_dir, f"{split_name}.jsonl")
        logger.info(f"Saving {split_name} dataset to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in data:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    # Generate dataset statistics
    logger.info("Dataset generation completed successfully")
    logger.info(f"Format version: {format_version}")
    logger.info(f"Total samples: {len(dataset_entries)}")
    logger.info(f"Train samples: {len(train_data)}")
    logger.info(f"Valid samples: {len(valid_data)}")
    logger.info(f"Test samples: {len(test_data)}")
    
    return True


def main():
    """Main function to generate improved datasets."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate improved datasets for Japanese GEC")
    parser.add_argument("--corpus-path", 
                       default="exclude/japanese_gec_corpus/corpus_v0.txt",
                       help="Path to GEC corpus file")
    parser.add_argument("--output-dir", 
                       default="datasets",
                       help="Output directory for datasets")
    parser.add_argument("--format-version", 
                       choices=["v1", "v2", "messages"],
                       default="v2",
                       help="Prompt format version")
    parser.add_argument("--train-ratio", 
                       type=float, 
                       default=0.8,
                       help="Training set ratio")
    parser.add_argument("--valid-ratio", 
                       type=float, 
                       default=0.1,
                       help="Validation set ratio")
    parser.add_argument("--test-ratio", 
                       type=float, 
                       default=0.1,
                       help="Test set ratio")
    parser.add_argument("--max-samples", 
                       type=int,
                       help="Maximum number of samples to process")
    parser.add_argument("--log-level", 
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO",
                       help="Logging level")
    parser.add_argument("--use-english", 
                       action="store_true",
                       help="Use English prompts instead of Japanese")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting improved dataset generation")
    logger.info(f"Corpus path: {args.corpus_path}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Format version: {args.format_version}")
    
    # Generate datasets
    success = generate_improved_datasets(
        corpus_path=args.corpus_path,
        output_dir=args.output_dir,
        format_version=args.format_version,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        max_samples=args.max_samples,
        use_english=args.use_english
    )
    
    if success:
        logger.info("Dataset generation completed successfully!")
        print("✅ Improved datasets generated successfully!")
        print(f"📁 Output directory: {args.output_dir}")
        print(f"🔧 Format version: {args.format_version}")
    else:
        logger.error("Dataset generation failed")
        print("❌ Dataset generation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()