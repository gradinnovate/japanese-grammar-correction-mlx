"""
Data processing utilities for Japanese Grammar Correction system.
Provides functions for data validation, cleaning, and transformation.
"""

import re
import logging
from typing import List, Tuple, Dict, Any, Optional


def clean_japanese_text(text: str) -> str:
    """
    Clean and normalize Japanese text.
    
    Args:
        text: Raw Japanese text
        
    Returns:
        Cleaned and normalized text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Normalize common punctuation
    text = text.replace('　', ' ')  # Full-width space to half-width
    text = text.replace('，', '、')  # Full-width comma to Japanese comma
    text = text.replace('．', '。')  # Full-width period to Japanese period
    
    return text


def validate_japanese_text(text: str) -> bool:
    """
    Validate if text contains Japanese characters.
    
    Args:
        text: Text to validate
        
    Returns:
        True if text contains Japanese characters, False otherwise
    """
    if not text:
        return False
    
    # Check for Hiragana, Katakana, or Kanji characters
    japanese_pattern = r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]'
    return bool(re.search(japanese_pattern, text))


def split_sentences(text: str) -> List[str]:
    """
    Split Japanese text into sentences.
    
    Args:
        text: Japanese text to split
        
    Returns:
        List of sentences
    """
    if not text:
        return []
    
    # Split on Japanese sentence endings
    sentences = re.split(r'[。！？]', text)
    
    # Clean and filter empty sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return sentences


def validate_gec_pair(error_text: str, correct_text: str) -> bool:
    """
    Validate a grammar error correction pair.
    
    Args:
        error_text: Text with errors
        correct_text: Corrected text
        
    Returns:
        True if pair is valid, False otherwise
    """
    if not error_text or not correct_text:
        return False
    
    # Both texts should contain Japanese characters
    if not (validate_japanese_text(error_text) and validate_japanese_text(correct_text)):
        return False
    
    # Texts should not be identical (unless no correction needed)
    # and should have reasonable length
    if len(error_text) > 1000 or len(correct_text) > 1000:
        logging.warning("Text pair exceeds maximum length")
        return False
    
    return True


def create_training_prompt(error_text: str, correct_text: str, format_version: str = "v2", use_english: bool = False) -> Dict[str, Any]:
    """
    Create a training prompt in the format expected by MLX-LM.
    
    Args:
        error_text: Text with grammatical errors
        correct_text: Corrected text
        format_version: Version of prompt format to use ("v1", "v2", "messages")
        use_english: Whether to use English prompts
        
    Returns:
        Dictionary with formatted prompt and response
    """
    from config.prompts import create_messages_format, PromptConfig
    
    if format_version == "messages":
        return create_messages_format(error_text, correct_text, use_english)
    elif format_version == "v2":
        # Improved prompt format using global prompts
        prompt_config = PromptConfig(use_english)
        system_msg = prompt_config.system_prompt
        user_msg = prompt_config.format_user_prompt(error_text)
        
        return {
            "text": f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{correct_text}<|im_end|>"
        }
    else:
        # Original format (v1) using global prompts
        prompt_config = PromptConfig(use_english)
        prompt = prompt_config.format_user_prompt(error_text)
        return {
            "text": f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{correct_text}<|im_end|>"
        }


def extract_correction_pairs(line: str) -> Optional[Tuple[str, str]]:
    """
    Extract error and correction pairs from GEC corpus format.
    Expected format: tab-separated with error markers <> and correction markers ()
    
    Args:
        line: Line from GEC corpus
        
    Returns:
        Tuple of (error_text, correct_text) or None if invalid
    """
    if not line or '\t' not in line:
        return None
    
    try:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            return None
        
        # Extract the marked text (assuming it's in the second column)
        marked_text = parts[1]
        
        # Extract error text (remove correction markers, keep error markers content)
        error_text = re.sub(r'\(([^)]*)\)', '', marked_text)  # Remove corrections
        error_text = re.sub(r'<([^>]*)>', r'\1', error_text)  # Keep error content
        
        # Extract correct text (remove error markers, keep correction markers content)
        correct_text = re.sub(r'<([^>]*)>', '', marked_text)  # Remove errors
        correct_text = re.sub(r'\(([^)]*)\)', r'\1', correct_text)  # Keep corrections
        
        # Clean both texts
        error_text = clean_japanese_text(error_text)
        correct_text = clean_japanese_text(correct_text)
        
        if validate_gec_pair(error_text, correct_text):
            return (error_text, correct_text)
        
        return None
        
    except Exception as e:
        logging.warning(f"Error processing line: {e}")
        return None


def calculate_dataset_stats(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistics for a dataset.
    
    Args:
        data: List of data samples
        
    Returns:
        Dictionary with dataset statistics
    """
    if not data:
        return {"total_samples": 0}
    
    total_samples = len(data)
    
    # Calculate text length statistics if 'text' field exists
    text_lengths = []
    for item in data:
        if 'text' in item and isinstance(item['text'], str):
            text_lengths.append(len(item['text']))
    
    stats = {
        "total_samples": total_samples,
        "avg_text_length": sum(text_lengths) / len(text_lengths) if text_lengths else 0,
        "min_text_length": min(text_lengths) if text_lengths else 0,
        "max_text_length": max(text_lengths) if text_lengths else 0,
    }
    
    return stats