# Batch Inference Usage Guide

This document explains how to use the batch inference functionality for Japanese Grammar Correction evaluation.

## Overview

The batch inference system processes test datasets to generate predictions for all test examples, saving results in a format suitable for evaluation metrics calculation. This implements task 4.2 from the implementation plan.

## Files

- `scripts/batch_inference.py` - Main batch inference script
- `scripts/run_batch_inference_example.py` - Example usage script
- `tests/test_batch_inference.py` - Unit tests for batch inference functionality

## Prerequisites

1. **Test Dataset**: A JSONL file with test examples (default: `datasets/test.jsonl`)
2. **Configuration**: Training configuration file (default: `config/lora_config.yaml`)
3. **Model**: Base model (mlx-community/Qwen3-0.6B-4bit)
4. **Adapters** (optional): Fine-tuned LoRA adapters (default: `models/japanese-gec-lora`)

## Usage

### Basic Usage

Run batch inference with default settings:

```bash
python scripts/batch_inference.py
```

### Custom Configuration

Specify custom paths and settings:

```bash
python scripts/batch_inference.py \
    --config config/lora_config.yaml \
    --test-file datasets/test.jsonl \
    --model-path mlx-community/Qwen3-0.6B-4bit \
    --adapter-path models/japanese-gec-lora \
    --output-file results/my_batch_results.jsonl \
    --log-level INFO
```

### Example Script

Use the example script to run with default settings:

```bash
# Check prerequisites only
python scripts/run_batch_inference_example.py --check-only

# Run full example
python scripts/run_batch_inference_example.py
```

## Input Format

The test dataset should be in JSONL format with the following structure:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "あなたは日本語の文法修正を専門とするAIアシスタントです。"
    },
    {
      "role": "user",
      "content": "Please correct the grammar in the following Japanese sentence: 山下先生は母をしています。"
    },
    {
      "role": "assistant",
      "content": "山下先生は母をご存知です。"
    }
  ]
}
```

## Output Format

The batch inference generates two output files:

### Results File (`*.jsonl`)

Each line contains a JSON object with:

```json
{
  "input": "山下先生は母をしています。",
  "expected": "山下先生は母をご存知です。",
  "predicted": "山下先生は母をご存知です。",
  "example_id": 1
}
```

For failed predictions, an additional `error` field is included:

```json
{
  "input": "エラーのある文",
  "expected": "修正された文",
  "predicted": "エラーのある文",
  "example_id": 2,
  "error": "Generation failed"
}
```

### Summary File (`*_summary.json`)

Contains overall statistics:

```json
{
  "total_examples": 100,
  "successful_predictions": 95,
  "failed_predictions": 5,
  "output_file": "results/batch_inference_results.jsonl",
  "timestamp": "2025-08-13 23:43:34"
}
```

## Configuration Options

The batch inference uses the same configuration file as training (`config/lora_config.yaml`). Key parameters:

- `model`: Base model path
- `adapter_path`: Path to LoRA adapters
- `prompt_template`: Template for formatting input prompts
- `max_tokens`: Maximum tokens to generate
- `temp`: Temperature for generation (lower = more deterministic)
- `top_p`: Nucleus sampling parameter

## Error Handling

The script handles various error conditions gracefully:

- **Missing files**: Clear error messages for missing datasets or models
- **Malformed data**: Skips invalid JSON entries with warnings
- **Generation failures**: Falls back to original text and logs errors
- **Memory issues**: Processes examples one at a time to minimize memory usage

## Performance

The batch inference processes examples sequentially to ensure stable memory usage. Progress is logged every 10 examples with timing information:

```
Processed 50/100 examples (avg: 2.34s/example, ETA: 117.0s)
```

## Testing

Run the unit tests to verify functionality:

```bash
python -m unittest tests.test_batch_inference -v
```

The tests cover:
- Loading test datasets
- Extracting input/output pairs
- Saving results and summaries
- Error handling for malformed data

## Integration with Evaluation

The output format is designed to be compatible with evaluation metrics calculation (task 4.3). The results file can be directly used by evaluation scripts to calculate:

- Sentence-level accuracy
- Token-level precision, recall, and F1
- BLEU scores
- Other GEC-specific metrics

## Troubleshooting

### Common Issues

1. **MLX import errors**: Ensure MLX and mlx-lm are installed
2. **Model not found**: Check model path and internet connection for downloads
3. **Memory issues**: Reduce batch size or use gradient checkpointing
4. **Encoding errors**: Ensure all files use UTF-8 encoding

### Debug Mode

Enable debug logging for detailed information:

```bash
python scripts/batch_inference.py --log-level DEBUG
```

This will show detailed information about data processing, model loading, and generation steps.