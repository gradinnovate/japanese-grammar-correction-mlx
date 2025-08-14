"""
Japanese GEC corpus parser for processing grammatical error correction data.

This module provides functions to parse the Japanese GEC corpus format
with error markers <> and correction markers ().
"""

import re
import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class GECPair:
    """Represents a grammatical error correction pair."""
    error_text: str
    correct_text: str
    original_line: str


def extract_error_corrections(line: str) -> Tuple[str, str]:
    """
    Extract clean error and correct sentence pairs from marked text.
    
    Args:
        line: Tab-separated line with error text <> and correct text ()
        
    Returns:
        Tuple of (error_text, correct_text) with markers removed
        
    Raises:
        ValueError: If line format is invalid
    """
    if not line.strip():
        raise ValueError("Empty line provided")
    
    parts = line.strip().split('\t')
    if len(parts) != 2:
        raise ValueError(f"Expected 2 tab-separated parts, got {len(parts)}")
    
    error_part, correct_part = parts
    
    # Extract error text by removing <> markers
    error_text = re.sub(r'<([^>]*)>', r'\1', error_part)
    
    # Extract correct text by removing () markers  
    correct_text = re.sub(r'\(([^)]*)\)', r'\1', correct_part)
    
    return error_text, correct_text


def validate_corpus_format(line: str) -> bool:
    """
    Validate that a corpus line follows the expected format.
    
    Args:
        line: Single line from corpus file
        
    Returns:
        True if format is valid, False otherwise
    """
    if not line.strip():
        return False
        
    parts = line.strip().split('\t')
    if len(parts) != 2:
        return False
    
    error_part, correct_part = parts
    
    # Check for error markers <> in first part
    if not re.search(r'<[^>]*>', error_part):
        return False
    
    # Check for correction markers () in second part
    if not re.search(r'\([^)]*\)', correct_part):
        return False
    
    return True


def parse_gec_corpus(corpus_path: str) -> List[GECPair]:
    """
    Parse the Japanese GEC corpus file into error-correction pairs.
    
    Args:
        corpus_path: Path to the tab-separated corpus file
        
    Returns:
        List of GECPair objects with extracted error and correction text
        
    Raises:
        FileNotFoundError: If corpus file doesn't exist
        UnicodeDecodeError: If file encoding is not UTF-8
    """
    pairs = []
    skipped_lines = 0
    
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    if not validate_corpus_format(line):
                        logger.warning(f"Skipping invalid format at line {line_num}: {line.strip()}")
                        skipped_lines += 1
                        continue
                    
                    error_text, correct_text = extract_error_corrections(line)
                    
                    pair = GECPair(
                        error_text=error_text,
                        correct_text=correct_text,
                        original_line=line.strip()
                    )
                    pairs.append(pair)
                    
                except ValueError as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
                    skipped_lines += 1
                    continue
                    
    except FileNotFoundError:
        logger.error(f"Corpus file not found: {corpus_path}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"UTF-8 encoding error in file {corpus_path}: {e}")
        raise
    
    logger.info(f"Parsed {len(pairs)} valid pairs from {corpus_path}")
    if skipped_lines > 0:
        logger.info(f"Skipped {skipped_lines} invalid lines")
    
    return pairs