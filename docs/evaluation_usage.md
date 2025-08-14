# Japanese Grammar Correction Evaluation Usage

This document explains how to use the evaluation metrics calculation system for Japanese Grammar Correction.

## Overview

The evaluation system provides comprehensive metrics for assessing the quality of Japanese grammar correction models, including:

- **Sentence-level accuracy**: Exact match between predicted and expected corrections
- **Token-level metrics**: Precision, recall, and F1 score at the token level
- **BLEU score**: Fluency assessment using n-gram overlap
- **Edit distance metrics**: Character-level edit distance and accuracy
- **Error analysis**: Breakdown of different types of corrections and failures

## Usage

### Basic Usage

```bash
python scripts/evaluate_gec.py --results-file results/batch_inference_results.jsonl
```

### Advanced Usage

```bash
python scripts/evaluate_gec.py \
    --results-file results/batch_inference_results.jsonl \
    --output-file results/my_evaluation_report.json \
    --log-level DEBUG \
    --log-file logs/evaluation.log
```

### Command Line Arguments

- `--results-file`: Path to batch inference results file (JSONL format) - **Required**
- `--output-file`: Path to save evaluation report (default: `results/evaluation_report.json`)
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR) (default: INFO)
- `--log-file`: Path to log file (default: `logs/evaluation.log`)

## Input Format

The evaluation script expects a JSONL file where each line contains a JSON object with the following structure:

```json
{
    "input": "山下先生は母をしています。",
    "expected": "山下先生は母をご存知です。",
    "predicted": "山下先生は母をご存知です。",
    "example_id": 1
}
```

For failed predictions, include an `error` field:

```json
{
    "input": "エラーのある文",
    "expected": "修正された文",
    "predicted": "エラーのある文",
    "example_id": 2,
    "error": "Generation failed"
}
```

## Output

The evaluation script generates two output files:

### 1. Detailed JSON Report

Contains comprehensive metrics and analysis:

```json
{
  "evaluation_summary": {
    "timestamp": "2025-08-14 00:32:46",
    "total_examples": 100,
    "successful_predictions": 95
  },
  "sentence_level_metrics": {
    "exact_match_accuracy": 0.75,
    "perfect_matches": 75,
    "perfect_match_rate": 0.75
  },
  "token_level_metrics": {
    "precision": 0.85,
    "recall": 0.82,
    "f1_score": 0.835
  },
  "fluency_metrics": {
    "bleu_score": 0.68
  },
  "edit_distance_metrics": {
    "normalized_edit_distance": 0.12,
    "edit_accuracy": 0.88
  },
  "error_analysis": {
    "perfect_matches": 75,
    "partial_corrections": 15,
    "no_corrections": 5,
    "overcorrections": 0,
    "failed_predictions": 5
  }
}
```

### 2. Human-Readable Summary

A text file with formatted metrics for easy reading:

```
Japanese Grammar Correction Evaluation Report
==================================================

Evaluation Date: 2025-08-14 00:32:46
Total Examples: 100
Successful Predictions: 95

SENTENCE-LEVEL METRICS
-------------------------
Exact Match Accuracy: 0.7500
Perfect Matches: 75
Perfect Match Rate: 0.7500

TOKEN-LEVEL METRICS
--------------------
Precision: 0.8500
Recall: 0.8200
F1 Score: 0.8350

FLUENCY METRICS
----------------
BLEU Score: 0.6800

EDIT DISTANCE METRICS
----------------------
Normalized Edit Distance: 0.1200
Edit Accuracy: 0.8800

ERROR ANALYSIS
---------------
Perfect Corrections: 75 (75.00%)
Partial Corrections: 15 (15.00%)
No Corrections: 5 (5.00%)
Overcorrections: 0 (0.00%)
Failed Predictions: 5 (5.00%)
```

## Metrics Explanation

### Sentence-Level Metrics

- **Exact Match Accuracy**: Percentage of predictions that exactly match the expected output
- **Perfect Match Rate**: Same as exact match accuracy, but calculated from total examples including failures

### Token-Level Metrics

- **Precision**: Ratio of correct tokens in predictions to total predicted tokens
- **Recall**: Ratio of correct tokens in predictions to total expected tokens
- **F1 Score**: Harmonic mean of precision and recall

### Fluency Metrics

- **BLEU Score**: Measures n-gram overlap between predictions and references, with brevity penalty

### Edit Distance Metrics

- **Normalized Edit Distance**: Levenshtein distance normalized by reference length
- **Edit Accuracy**: 1 - normalized edit distance

### Error Analysis

- **Perfect Corrections**: Predictions that exactly match expected output
- **Partial Corrections**: Predictions that differ from expected but are not identical to input
- **No Corrections**: Predictions identical to input (no changes made)
- **Overcorrections**: Predictions significantly longer than expected (heuristic)
- **Failed Predictions**: Predictions that failed due to errors

## Example

Run the example script to see the evaluation system in action:

```bash
python scripts/run_evaluation_example.py
```

This will create sample data, run evaluation, and generate reports in the `results/` directory.

## Integration with Batch Inference

The evaluation script is designed to work with the output from `scripts/batch_inference.py`:

```bash
# Step 1: Run batch inference
python scripts/batch_inference.py \
    --test-file datasets/test.jsonl \
    --output-file results/batch_inference_results.jsonl

# Step 2: Evaluate results
python scripts/evaluate_gec.py \
    --results-file results/batch_inference_results.jsonl \
    --output-file results/evaluation_report.json
```

## Troubleshooting

### Common Issues

1. **No valid results found**: Check that your results file contains valid JSON objects without errors
2. **File not found**: Ensure the results file path is correct and the file exists
3. **Encoding issues**: Make sure your results file is saved with UTF-8 encoding

### Debugging

Use debug logging to get more detailed information:

```bash
python scripts/evaluate_gec.py \
    --results-file results/batch_inference_results.jsonl \
    --log-level DEBUG
```

## Testing

Run the unit tests to verify the evaluation system:

```bash
python -m unittest tests.test_evaluate_gec -v
```