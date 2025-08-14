"""
Dataset splitting utilities for Japanese GEC training.

This module provides functions to split the processed GEC data into
training, validation, and test sets with balanced error type distribution.
"""

import random
import os
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from utils.mlx_formatter import save_to_jsonl


def analyze_error_types(formatted_data: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """
    Analyze error types in the dataset for balanced splitting.
    
    This is a simplified analysis based on text patterns.
    In a more sophisticated version, we could use linguistic analysis.
    
    Args:
        formatted_data: List of formatted training examples
        
    Returns:
        Dictionary mapping error types to list of example indices
    """
    error_types = defaultdict(list)
    
    for idx, example in enumerate(formatted_data):
        messages = example['messages']
        
        # Extract the error text from the user message
        user_message = None
        for msg in messages:
            if msg['role'] == 'user':
                user_message = msg['content']
                break
        
        if user_message:
            # Extract error text from the user message
            if '：' in user_message:
                error_text = user_message.split('：', 1)[1].strip()
            else:
                error_text = user_message
            
            # Simple error type classification based on patterns
            error_type = classify_error_type(error_text)
            error_types[error_type].append(idx)
    
    return error_types


def classify_error_type(error_text: str) -> str:
    """
    Classify the type of grammatical error (simplified classification).
    
    Args:
        error_text: Text containing grammatical errors
        
    Returns:
        Error type category as string
    """
    # Simple pattern-based classification
    if any(particle in error_text for particle in ['を', 'が', 'に', 'で', 'と', 'から', 'まで']):
        return 'particle'
    elif any(verb_ending in error_text for verb_ending in ['た', 'だ', 'ます', 'です', 'る', 'う']):
        return 'verb_form'
    elif any(adj_pattern in error_text for adj_pattern in ['い', 'な', 'だ']):
        return 'adjective'
    elif any(dem in error_text for dem in ['この', 'その', 'あの', 'これ', 'それ', 'あれ']):
        return 'demonstrative'
    else:
        return 'other'


def stratified_split(
    data: List[Dict[str, Any]], 
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split dataset into train/validation/test sets with stratified sampling.
    
    Args:
        data: List of formatted training examples
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set  
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
    
    random.seed(random_seed)
    
    # Analyze error types for stratified splitting
    error_types = analyze_error_types(data)
    
    train_data = []
    val_data = []
    test_data = []
    
    # Split each error type proportionally
    for error_type, indices in error_types.items():
        # Shuffle indices for this error type
        random.shuffle(indices)
        
        n_total = len(indices)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        
        # Split indices
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Add examples to respective sets
        for idx in train_indices:
            train_data.append(data[idx])
        for idx in val_indices:
            val_data.append(data[idx])
        for idx in test_indices:
            test_data.append(data[idx])
    
    # Shuffle the final datasets
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)
    
    return train_data, val_data, test_data


def save_dataset_splits(
    train_data: List[Dict[str, Any]],
    val_data: List[Dict[str, Any]], 
    test_data: List[Dict[str, Any]],
    output_dir: str = "datasets"
) -> None:
    """
    Save train/validation/test splits to separate JSONL files.
    
    Args:
        train_data: Training examples
        val_data: Validation examples
        test_data: Test examples
        output_dir: Directory to save the files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save each split
    train_path = os.path.join(output_dir, "train.jsonl")
    val_path = os.path.join(output_dir, "valid.jsonl")
    test_path = os.path.join(output_dir, "test.jsonl")
    
    save_to_jsonl(train_data, train_path)
    save_to_jsonl(val_data, val_path)
    save_to_jsonl(test_data, test_path)
    
    print(f"\nDataset splits saved:")
    print(f"  Training: {len(train_data)} examples -> {train_path}")
    print(f"  Validation: {len(val_data)} examples -> {val_path}")
    print(f"  Test: {len(test_data)} examples -> {test_path}")


def create_dataset_splits_from_corpus(
    corpus_path: str,
    output_dir: str = "datasets",
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> None:
    """
    Complete pipeline to create dataset splits from GEC corpus.
    
    Args:
        corpus_path: Path to the Japanese GEC corpus file
        output_dir: Directory to save the split datasets
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility
    """
    from utils.gec_parser import parse_gec_corpus
    from utils.mlx_formatter import convert_pairs_to_mlx_format
    
    print(f"Processing corpus: {corpus_path}")
    
    # Parse the corpus
    pairs = parse_gec_corpus(corpus_path)
    print(f"Parsed {len(pairs)} GEC pairs")
    
    # Convert to MLX format
    formatted_data = convert_pairs_to_mlx_format(pairs)
    print(f"Converted {len(formatted_data)} pairs to MLX format")
    
    # Create splits
    train_data, val_data, test_data = stratified_split(
        formatted_data, train_ratio, val_ratio, test_ratio, random_seed
    )
    
    # Save splits
    save_dataset_splits(train_data, val_data, test_data, output_dir)
    
    # Print error type distribution
    print("\nError type distribution analysis:")
    error_types = analyze_error_types(formatted_data)
    for error_type, indices in error_types.items():
        print(f"  {error_type}: {len(indices)} examples")