"""
AutoJQE corpus parser for processing Japanese GEC quality estimation data.

This module provides functions to parse the AutoJQE corpus format
which contains source sentences, GEC system outputs, and quality scores.
"""

import csv
import logging
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QEPair:
    """Represents a quality estimation pair with scores."""
    source_text: str
    corrected_text: str
    individual_scores: List[float]
    average_score: float
    is_improved: bool  # Whether correction improved the text
    
    @property
    def quality_level(self) -> str:
        """Get quality level based on average score."""
        if self.average_score >= 3.5:
            return "excellent"
        elif self.average_score >= 2.5:
            return "good"
        elif self.average_score >= 1.5:
            return "fair"
        else:
            return "poor"


def parse_individual_scores(score_str: str) -> List[float]:
    """
    Parse individual scores from string format.
    
    Args:
        score_str: String like "2,2,2" or "3,4,4"
        
    Returns:
        List of individual scores as floats
        
    Raises:
        ValueError: If score format is invalid
    """
    try:
        # Remove quotes and split by comma
        clean_str = score_str.strip('"')
        scores = [float(s.strip()) for s in clean_str.split(',')]
        
        # Validate score range (1-4)
        for score in scores:
            if not (1.0 <= score <= 4.0):
                raise ValueError(f"Score {score} out of valid range (1-4)")
        
        return scores
    except (ValueError, AttributeError) as e:
        raise ValueError(f"Invalid score format: {score_str}") from e


def calculate_improvement(source: str, corrected: str, avg_score: float) -> bool:
    """
    Determine if the correction improved the text.
    
    Args:
        source: Original source text
        corrected: Corrected text
        avg_score: Average quality score
        
    Returns:
        True if correction improved the text
    """
    # If texts are identical and score is high, it was already good
    if source == corrected:
        return avg_score >= 3.0
    
    # If texts differ and score is high, correction was good
    return avg_score >= 2.5


def parse_autojqe_csv(csv_path: str) -> List[QEPair]:
    """
    Parse AutoJQE CSV file into quality estimation pairs.
    
    Args:
        csv_path: Path to the CSV file
        
    Returns:
        List of QEPair objects with quality scores
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If CSV format is invalid
    """
    pairs = []
    skipped_rows = 0
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            # Use tab delimiter as shown in the data
            reader = csv.DictReader(f, delimiter='\t')
            
            for row_num, row in enumerate(reader, 2):  # Start from 2 (header is row 1)
                try:
                    source_text = row['原文'].strip()
                    corrected_text = row['訂正'].strip()
                    individual_scores_str = row['個別評価']
                    average_score = float(row['評価'])
                    
                    # Parse individual scores
                    individual_scores = parse_individual_scores(individual_scores_str)
                    
                    # Determine if correction improved the text
                    is_improved = calculate_improvement(source_text, corrected_text, average_score)
                    
                    pair = QEPair(
                        source_text=source_text,
                        corrected_text=corrected_text,
                        individual_scores=individual_scores,
                        average_score=average_score,
                        is_improved=is_improved
                    )
                    pairs.append(pair)
                    
                except (KeyError, ValueError) as e:
                    logger.warning(f"Error processing row {row_num}: {e}")
                    skipped_rows += 1
                    continue
                    
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_path}")
        raise
    except UnicodeDecodeError as e:
        logger.error(f"UTF-8 encoding error in file {csv_path}: {e}")
        raise
    
    logger.info(f"Parsed {len(pairs)} valid pairs from {csv_path}")
    if skipped_rows > 0:
        logger.info(f"Skipped {skipped_rows} invalid rows")
    
    return pairs


def parse_all_autojqe_datasets(datasets_dir: str) -> Dict[str, List[QEPair]]:
    """
    Parse all AutoJQE datasets in the directory.
    
    Args:
        datasets_dir: Directory containing AutoJQE CSV files
        
    Returns:
        Dictionary mapping dataset names to QEPair lists
    """
    datasets = {}
    datasets_path = Path(datasets_dir)
    
    if not datasets_path.exists():
        raise FileNotFoundError(f"Datasets directory not found: {datasets_dir}")
    
    # Find all CSV files
    csv_files = list(datasets_path.glob("*.csv"))
    
    for csv_file in csv_files:
        dataset_name = csv_file.stem  # filename without extension
        try:
            pairs = parse_autojqe_csv(str(csv_file))
            datasets[dataset_name] = pairs
            logger.info(f"Loaded {len(pairs)} pairs from {dataset_name}")
        except Exception as e:
            logger.error(f"Failed to parse {csv_file}: {e}")
    
    return datasets


def filter_by_quality(pairs: List[QEPair], min_score: float = 2.0) -> List[QEPair]:
    """
    Filter pairs by minimum quality score.
    
    Args:
        pairs: List of QEPair objects
        min_score: Minimum average score threshold
        
    Returns:
        Filtered list of pairs
    """
    return [pair for pair in pairs if pair.average_score >= min_score]


def get_quality_statistics(pairs: List[QEPair]) -> Dict[str, int]:
    """
    Get statistics about quality levels in the dataset.
    
    Args:
        pairs: List of QEPair objects
        
    Returns:
        Dictionary with quality level counts
    """
    stats = {"excellent": 0, "good": 0, "fair": 0, "poor": 0}
    
    for pair in pairs:
        stats[pair.quality_level] += 1
    
    return stats


def get_improvement_statistics(pairs: List[QEPair]) -> Dict[str, int]:
    """
    Get statistics about correction improvements.
    
    Args:
        pairs: List of QEPair objects
        
    Returns:
        Dictionary with improvement statistics
    """
    improved = sum(1 for pair in pairs if pair.is_improved)
    not_improved = len(pairs) - improved
    
    return {
        "improved": improved,
        "not_improved": not_improved,
        "improvement_rate": improved / len(pairs) if pairs else 0.0
    }