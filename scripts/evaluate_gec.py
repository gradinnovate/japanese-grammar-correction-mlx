#!/usr/bin/env python3
"""
Evaluation Script for Japanese Grammar Correction

This script implements comprehensive evaluation metrics for Japanese Grammar Correction,
including GEC-specific metrics (precision, recall, F1), sentence-level accuracy,
and BLEU score computation. It generates detailed evaluation reports comparing
predictions to ground truth.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import re
from collections import defaultdict

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.logging_utils import setup_logging, get_logger


def load_evaluation_results(results_file: str) -> List[Dict[str, Any]]:
    """Load batch inference results for evaluation."""
    results = []
    try:
        with open(results_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        result = json.loads(line)
                        results.append(result)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Skipping malformed JSON at line {line_num}: {e}")
        
        logging.info(f"Loaded {len(results)} evaluation results from {results_file}")
        return results
    except Exception as e:
        logging.error(f"Failed to load evaluation results: {e}")
        raise


def tokenize_japanese(text: str) -> List[str]:
    """
    Simple tokenization for Japanese text.
    Splits on whitespace and punctuation for basic token-level evaluation.
    """
    if not text:
        return []
    
    # Split on whitespace and common punctuation, keeping individual characters for better granularity
    tokens = []
    current_token = ""
    
    for char in text:
        if char in '、。！？，．\s':
            if current_token:
                tokens.append(current_token)
                current_token = ""
            if char.strip():  # Don't add whitespace as tokens
                tokens.append(char)
        else:
            current_token += char
    
    if current_token:
        tokens.append(current_token)
    
    return [token.strip() for token in tokens if token.strip()]


def calculate_sentence_accuracy(predictions: List[str], references: List[str]) -> float:
    """
    Calculate sentence-level accuracy (exact match).
    
    Args:
        predictions: List of predicted sentences
        references: List of reference sentences
        
    Returns:
        Sentence-level accuracy as a float between 0 and 1
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")
    
    if not predictions:
        return 0.0
    
    exact_matches = sum(1 for pred, ref in zip(predictions, references) 
                       if pred.strip() == ref.strip())
    
    return exact_matches / len(predictions)


def calculate_token_level_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate token-level precision, recall, and F1 score.
    
    Args:
        predictions: List of predicted sentences
        references: List of reference sentences
        
    Returns:
        Dictionary with precision, recall, and f1 scores
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")
    
    total_pred_tokens = 0
    total_ref_tokens = 0
    total_correct_tokens = 0
    
    for pred, ref in zip(predictions, references):
        pred_tokens = set(tokenize_japanese(pred))
        ref_tokens = set(tokenize_japanese(ref))
        
        total_pred_tokens += len(pred_tokens)
        total_ref_tokens += len(ref_tokens)
        total_correct_tokens += len(pred_tokens & ref_tokens)
    
    # Calculate metrics
    precision = total_correct_tokens / total_pred_tokens if total_pred_tokens > 0 else 0.0
    recall = total_correct_tokens / total_ref_tokens if total_ref_tokens > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def calculate_bleu_score(predictions: List[str], references: List[str], n_gram: int = 4) -> float:
    """
    Calculate BLEU score for fluency assessment.
    Simplified implementation focusing on n-gram precision.
    
    Args:
        predictions: List of predicted sentences
        references: List of reference sentences
        n_gram: Maximum n-gram order (default: 4)
        
    Returns:
        BLEU score as a float between 0 and 1
    """
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")
    
    if not predictions:
        return 0.0
    
    # Calculate n-gram precisions
    precisions = []
    
    for n in range(1, min(n_gram + 1, 5)):  # Limit to reasonable n-gram size
        total_ngrams = 0
        matched_ngrams = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = tokenize_japanese(pred)
            ref_tokens = tokenize_japanese(ref)
            
            # Skip if not enough tokens for n-gram
            if len(pred_tokens) < n or len(ref_tokens) < n:
                continue
            
            # Generate n-grams
            pred_ngrams = [tuple(pred_tokens[i:i+n]) for i in range(len(pred_tokens) - n + 1)]
            ref_ngrams = [tuple(ref_tokens[i:i+n]) for i in range(len(ref_tokens) - n + 1)]
            
            # Count matches
            ref_ngram_counts = defaultdict(int)
            for ngram in ref_ngrams:
                ref_ngram_counts[ngram] += 1
            
            for ngram in pred_ngrams:
                if ref_ngram_counts[ngram] > 0:
                    matched_ngrams += 1
                    ref_ngram_counts[ngram] -= 1
                total_ngrams += 1
        
        precision = matched_ngrams / total_ngrams if total_ngrams > 0 else 0.0
        precisions.append(precision)
    
    # Calculate geometric mean of precisions (only use available precisions)
    if precisions and all(p > 0 for p in precisions):
        if len(precisions) == 1:
            bleu = precisions[0]
        elif len(precisions) == 2:
            bleu = (precisions[0] * precisions[1]) ** 0.5
        elif len(precisions) == 3:
            bleu = (precisions[0] * precisions[1] * precisions[2]) ** (1/3)
        else:
            bleu = (precisions[0] * precisions[1] * precisions[2] * precisions[3]) ** 0.25
    else:
        # If any precision is 0, use arithmetic mean of available precisions
        bleu = sum(precisions) / len(precisions) if precisions else 0.0
    
    # Apply brevity penalty
    total_pred_length = sum(len(tokenize_japanese(pred)) for pred in predictions)
    total_ref_length = sum(len(tokenize_japanese(ref)) for ref in references)
    
    if total_pred_length < total_ref_length and total_ref_length > 0:
        brevity_penalty = total_pred_length / total_ref_length
        bleu *= brevity_penalty
    
    return bleu


def calculate_edit_distance_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Calculate edit distance-based metrics for GEC evaluation.
    
    Args:
        predictions: List of predicted sentences
        references: List of reference sentences
        
    Returns:
        Dictionary with edit distance metrics
    """
    def levenshtein_distance(s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    if len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same length")
    
    total_distance = 0
    total_ref_length = 0
    
    for pred, ref in zip(predictions, references):
        distance = levenshtein_distance(pred, ref)
        total_distance += distance
        total_ref_length += len(ref)
    
    # Calculate normalized edit distance
    normalized_edit_distance = total_distance / total_ref_length if total_ref_length > 0 else 0.0
    
    return {
        'total_edit_distance': total_distance,
        'normalized_edit_distance': normalized_edit_distance,
        'edit_accuracy': 1.0 - normalized_edit_distance
    }


def analyze_error_types(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze different types of errors and corrections.
    
    Args:
        results: List of evaluation results
        
    Returns:
        Dictionary with error type analysis
    """
    error_analysis = {
        'total_examples': len(results),
        'perfect_matches': 0,
        'partial_corrections': 0,
        'no_corrections': 0,
        'overcorrections': 0,
        'failed_predictions': 0
    }
    
    for result in results:
        if 'error' in result:
            error_analysis['failed_predictions'] += 1
            continue
        
        input_text = result.get('input', '').strip()
        expected = result.get('expected', '').strip()
        predicted = result.get('predicted', '').strip()
        
        if predicted == expected:
            error_analysis['perfect_matches'] += 1
        elif predicted == input_text:
            error_analysis['no_corrections'] += 1
        elif len(predicted) > len(expected) * 1.5:  # Heuristic for overcorrection
            error_analysis['overcorrections'] += 1
        else:
            error_analysis['partial_corrections'] += 1
    
    return error_analysis


def generate_detailed_report(metrics: Dict[str, Any], error_analysis: Dict[str, Any], 
                           output_file: str) -> None:
    """
    Generate a detailed evaluation report.
    
    Args:
        metrics: Dictionary with all calculated metrics
        error_analysis: Dictionary with error type analysis
        output_file: Path to save the report
    """
    report = {
        'evaluation_summary': {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_examples': error_analysis['total_examples'],
            'successful_predictions': error_analysis['total_examples'] - error_analysis['failed_predictions']
        },
        'sentence_level_metrics': {
            'exact_match_accuracy': metrics['sentence_accuracy'],
            'perfect_matches': error_analysis['perfect_matches'],
            'perfect_match_rate': error_analysis['perfect_matches'] / error_analysis['total_examples']
        },
        'token_level_metrics': {
            'precision': metrics['token_precision'],
            'recall': metrics['token_recall'],
            'f1_score': metrics['token_f1']
        },
        'fluency_metrics': {
            'bleu_score': metrics['bleu_score']
        },
        'edit_distance_metrics': {
            'normalized_edit_distance': metrics['normalized_edit_distance'],
            'edit_accuracy': metrics['edit_accuracy']
        },
        'error_analysis': error_analysis,
        'performance_breakdown': {
            'perfect_match_rate': error_analysis['perfect_matches'] / error_analysis['total_examples'],
            'partial_correction_rate': error_analysis['partial_corrections'] / error_analysis['total_examples'],
            'no_correction_rate': error_analysis['no_corrections'] / error_analysis['total_examples'],
            'overcorrection_rate': error_analysis['overcorrections'] / error_analysis['total_examples'],
            'failure_rate': error_analysis['failed_predictions'] / error_analysis['total_examples']
        }
    }
    
    # Save detailed report as JSON
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Also create a human-readable summary
    summary_file = output_file.replace('.json', '_summary.txt')
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Japanese Grammar Correction Evaluation Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"Evaluation Date: {report['evaluation_summary']['timestamp']}\n")
        f.write(f"Total Examples: {report['evaluation_summary']['total_examples']}\n")
        f.write(f"Successful Predictions: {report['evaluation_summary']['successful_predictions']}\n\n")
        
        f.write("SENTENCE-LEVEL METRICS\n")
        f.write("-" * 25 + "\n")
        f.write(f"Exact Match Accuracy: {report['sentence_level_metrics']['exact_match_accuracy']:.4f}\n")
        f.write(f"Perfect Matches: {report['sentence_level_metrics']['perfect_matches']}\n")
        f.write(f"Perfect Match Rate: {report['sentence_level_metrics']['perfect_match_rate']:.4f}\n\n")
        
        f.write("TOKEN-LEVEL METRICS\n")
        f.write("-" * 20 + "\n")
        f.write(f"Precision: {report['token_level_metrics']['precision']:.4f}\n")
        f.write(f"Recall: {report['token_level_metrics']['recall']:.4f}\n")
        f.write(f"F1 Score: {report['token_level_metrics']['f1_score']:.4f}\n\n")
        
        f.write("FLUENCY METRICS\n")
        f.write("-" * 16 + "\n")
        f.write(f"BLEU Score: {report['fluency_metrics']['bleu_score']:.4f}\n\n")
        
        f.write("EDIT DISTANCE METRICS\n")
        f.write("-" * 22 + "\n")
        f.write(f"Normalized Edit Distance: {report['edit_distance_metrics']['normalized_edit_distance']:.4f}\n")
        f.write(f"Edit Accuracy: {report['edit_distance_metrics']['edit_accuracy']:.4f}\n\n")
        
        f.write("ERROR ANALYSIS\n")
        f.write("-" * 15 + "\n")
        f.write(f"Perfect Corrections: {error_analysis['perfect_matches']} ({report['performance_breakdown']['perfect_match_rate']:.2%})\n")
        f.write(f"Partial Corrections: {error_analysis['partial_corrections']} ({report['performance_breakdown']['partial_correction_rate']:.2%})\n")
        f.write(f"No Corrections: {error_analysis['no_corrections']} ({report['performance_breakdown']['no_correction_rate']:.2%})\n")
        f.write(f"Overcorrections: {error_analysis['overcorrections']} ({report['performance_breakdown']['overcorrection_rate']:.2%})\n")
        f.write(f"Failed Predictions: {error_analysis['failed_predictions']} ({report['performance_breakdown']['failure_rate']:.2%})\n")
    
    logging.info(f"Detailed report saved to {output_file}")
    logging.info(f"Summary report saved to {summary_file}")


def evaluate_gec_results(results: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Evaluate GEC results and calculate all metrics.
    
    Args:
        results: List of evaluation results from batch inference
        
    Returns:
        Tuple of (metrics_dict, error_analysis_dict)
    """
    logger = get_logger(__name__)
    
    if not results:
        logger.error("No results provided for evaluation")
        return {}, {'total_examples': 0, 'failed_predictions': 0, 'perfect_matches': 0, 
                   'partial_corrections': 0, 'no_corrections': 0, 'overcorrections': 0}
    
    logger.info(f"Evaluating {len(results)} results...")
    
    # Filter out failed predictions for metric calculation
    valid_results = [r for r in results if 'error' not in r]
    logger.info(f"Using {len(valid_results)} valid results for metric calculation")
    
    if not valid_results:
        logger.error("No valid results found for evaluation")
        error_analysis = analyze_error_types(results)
        return {}, error_analysis
    
    # Extract predictions and references
    predictions = [r['predicted'] for r in valid_results]
    references = [r['expected'] for r in valid_results]
    
    # Calculate all metrics
    logger.info("Calculating sentence-level accuracy...")
    sentence_accuracy = calculate_sentence_accuracy(predictions, references)
    
    logger.info("Calculating token-level metrics...")
    token_metrics = calculate_token_level_metrics(predictions, references)
    
    logger.info("Calculating BLEU score...")
    bleu_score = calculate_bleu_score(predictions, references)
    
    logger.info("Calculating edit distance metrics...")
    edit_metrics = calculate_edit_distance_metrics(predictions, references)
    
    logger.info("Analyzing error types...")
    error_analysis = analyze_error_types(results)
    
    # Combine all metrics
    all_metrics = {
        'sentence_accuracy': sentence_accuracy,
        'token_precision': token_metrics['precision'],
        'token_recall': token_metrics['recall'],
        'token_f1': token_metrics['f1'],
        'bleu_score': bleu_score,
        'normalized_edit_distance': edit_metrics['normalized_edit_distance'],
        'edit_accuracy': edit_metrics['edit_accuracy']
    }
    
    logger.info("Evaluation completed successfully")
    return all_metrics, error_analysis


def main():
    """Main evaluation script entry point."""
    parser = argparse.ArgumentParser(description="Evaluate Japanese Grammar Correction results")
    parser.add_argument(
        "--results-file",
        required=True,
        help="Path to batch inference results file (JSONL format)"
    )
    parser.add_argument(
        "--output-file",
        default="results/evaluation_report.json",
        help="Path to save evaluation report"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        default="logs/evaluation.log",
        help="Path to log file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level, log_file=args.log_file)
    logger = get_logger(__name__)
    
    try:
        logger.info("=== Japanese Grammar Correction Evaluation ===")
        logger.info(f"Results file: {args.results_file}")
        logger.info(f"Output file: {args.output_file}")
        logger.info("=" * 47)
        
        # Load evaluation results
        results = load_evaluation_results(args.results_file)
        if not results:
            logger.error("No results loaded for evaluation")
            sys.exit(1)
        
        # Evaluate results
        metrics, error_analysis = evaluate_gec_results(results)
        if not metrics:
            logger.error("Evaluation failed")
            sys.exit(1)
        
        # Generate detailed report
        generate_detailed_report(metrics, error_analysis, args.output_file)
        
        # Log summary
        logger.info("=== EVALUATION SUMMARY ===")
        logger.info(f"Total Examples: {len(results)}")
        logger.info(f"Sentence Accuracy: {metrics['sentence_accuracy']:.4f}")
        logger.info(f"Token F1 Score: {metrics['token_f1']:.4f}")
        logger.info(f"BLEU Score: {metrics['bleu_score']:.4f}")
        logger.info(f"Edit Accuracy: {metrics['edit_accuracy']:.4f}")
        logger.info("=" * 27)
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Detailed report available at: {args.output_file}")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()