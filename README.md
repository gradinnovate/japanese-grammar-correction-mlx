# Japanese Grammar Correction System

A machine learning system for Japanese grammatical error correction using MLX LoRA fine-tuning of the Qwen3-0.6B-4bit model. This system processes Japanese GEC corpus data to train a specialized model that can automatically detect and correct grammatical errors in Japanese text.

## Features

- **End-to-End Pipeline**: Complete training pipeline from data preprocessing to model evaluation
- **MLX LoRA Fine-tuning**: Efficient fine-tuning using LoRA (Low-Rank Adaptation) with MLX framework
- **Japanese GEC Corpus Support**: Built-in support for Japanese Grammar Error Correction corpus format
- **Comprehensive Evaluation**: Multiple evaluation metrics including BLEU, F1, and edit distance
- **Interactive Interface**: Command-line tools for both batch processing and interactive correction
- **Flexible Configuration**: YAML-based configuration system for easy customization

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Format](#data-format)
- [Training Process](#training-process)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Evaluation](#evaluation)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)

## Installation

### Prerequisites

- Python 3.8 or higher
- macOS with Apple Silicon (for MLX support)
- At least 8GB of RAM
- 10GB of free disk space

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd japanese-grammar-correction
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the project structure:**
   ```bash
   python setup_project.py
   ```

5. **Download and prepare models:**
   ```bash
   bash scripts/setup_models.sh
   ```

## Quick Start

### 1. Prepare Your Data

Place your Japanese GEC corpus file in the `exclude/japanese_gec_corpus/` directory:

```bash
# Example corpus format (tab-separated):
# ID    Marked_Text    Error_Type
1	私は<昨日>（昨日）映画を見ました。	Correct
2	彼は<学校に行く>（学校に行きます）。	Grammar
3	<これは>（これが）私の本です。	Particle
```

### 2. Run Complete Training Pipeline

```bash
# Run the complete pipeline with default settings
python scripts/complete_training_pipeline.py

# Or with custom configuration
python scripts/complete_training_pipeline.py --config config/custom_config.yaml
```

### 3. Interactive Grammar Correction

```bash
# Start interactive correction session
python scripts/interactive_correction.py

# Example usage:
# Input: 私は学校に行く。
# Output: 私は学校に行きます。
```

### 4. Batch File Processing

```bash
# Process a text file
python scripts/batch_file_correction.py --input input.txt --output corrected.txt
```

## Data Format

### Japanese GEC Corpus Format

The system expects tab-separated corpus files with the following format:

```
ID	Marked_Text	Error_Type
1	私は<error_text>（correct_text）残りのテキスト。	Grammar
```

**Markers:**
- `<error_text>`: Text containing grammatical errors
- `(correct_text)`: Corrected version of the error text

**Example:**
```
1	私は<学校に行く>（学校に行きます）。	Grammar
2	<これは>（これが）私の本です。	Particle
3	彼女は<きれい>（きれいです）。	Adjective
```

### Training Data Format

The system converts GEC corpus to MLX-compatible JSONL format:

```json
{"text": "<|im_start|>user\n以下の日本語文の文法を修正してください：\n私は学校に行く。<|im_end|>\n<|im_start|>assistant\n私は学校に行きます。<|im_end|>"}
```

## Training Process

### 1. Data Preprocessing

```bash
# Preprocess corpus data
python utils/gec_parser.py --input exclude/japanese_gec_corpus/corpus_v0.txt --output datasets/
```

### 2. Model Training

```bash
# Train with default configuration
python scripts/train_japanese_gec.py

# Train with custom parameters
python scripts/train_japanese_gec.py --config config/custom_lora_config.yaml
```

### 3. Model Evaluation

```bash
# Run evaluation on test set
python scripts/evaluate_gec.py --results-file results/test_predictions.jsonl
```

## Usage Examples

### Complete Pipeline Example

```bash
# Run complete pipeline with custom settings
python scripts/complete_training_pipeline.py \
    --config config/lora_config.yaml \
    --output-dir my_training_run \
    --max-samples 1000 \
    --train-ratio 0.8 \
    --valid-ratio 0.1 \
    --test-ratio 0.1
```

### Training Only Example

```bash
# Skip preprocessing and evaluation, only train
python scripts/complete_training_pipeline.py \
    --skip-preprocessing \
    --skip-evaluation \
    --config config/lora_config.yaml
```

### Batch Inference Example

```bash
# Process test dataset
python scripts/batch_inference.py \
    --model mlx-community/Qwen3-0.6B-4bit \
    --adapter-path models/japanese-gec-lora \
    --input-file datasets/test.jsonl \
    --output-file results/predictions.jsonl \
    --batch-size 4
```

### Interactive Correction Example

```bash
# Start interactive session
python scripts/interactive_correction.py \
    --model mlx-community/Qwen3-0.6B-4bit \
    --adapter-path models/japanese-gec-lora
```

### Evaluation Example

```bash
# Evaluate model performance
python scripts/evaluate_gec.py \
    --results-file results/predictions.jsonl \
    --output-file results/evaluation_report.json
```

## Configuration

### Main Configuration File (`config/lora_config.yaml`)

```yaml
# Model configuration
model: "mlx-community/Qwen3-0.6B-4bit"

# LoRA parameters
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05

# Training parameters
learning_rate: 0.0001
batch_size: 4
iters: 2000
val_batches: 50

# Data paths
train: "datasets/train.jsonl"
valid: "datasets/valid.jsonl"
test: "datasets/test.jsonl"

# Output configuration
adapter_path: "models/japanese-gec-lora"

# Generation parameters
max_tokens: 512
temp: 0.1
top_p: 0.9
```

### Pipeline Configuration Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `corpus_path` | Path to GEC corpus file | `exclude/japanese_gec_corpus/corpus_v0.txt` |
| `train_ratio` | Training set ratio | `0.8` |
| `valid_ratio` | Validation set ratio | `0.1` |
| `test_ratio` | Test set ratio | `0.1` |
| `max_samples` | Maximum samples to use | `None` (all) |
| `min_samples` | Minimum samples required | `100` |
| `skip_preprocessing` | Skip data preprocessing | `False` |
| `skip_training` | Skip model training | `False` |
| `skip_evaluation` | Skip evaluation | `False` |

## Evaluation

### Metrics

The system provides comprehensive evaluation metrics:

- **Sentence-level Accuracy**: Exact match between prediction and reference
- **Token-level Metrics**: Precision, Recall, and F1 score at token level
- **BLEU Score**: Fluency assessment using n-gram overlap
- **Edit Distance**: Normalized Levenshtein distance and edit accuracy

### Evaluation Methods

The system provides multiple evaluation approaches:

#### 1. Flexible Evaluation (Similarity-Based)
```bash
# Run flexible evaluation with similarity metrics
python scripts/flexible_evaluation.py
```

#### 2. Grammar-Focused Evaluation (Strict)
```bash
# Run strict grammar-focused evaluation
python scripts/grammar_focused_evaluation.py
```

#### 3. Standard Evaluation
```bash
# Generate detailed evaluation report
python scripts/evaluate_gec.py \
    --results-file results/predictions.jsonl \
    --output-file results/evaluation_report.json
```

### Current Model Performance

Based on evaluation with 50 test samples:

#### japanese-gec-lora-simple Model:
- **Exact Match Accuracy**: 20% (10/50 samples)
- **Flexible Accuracy**: 80% (40/50 samples with partial matching)
- **Average Similarity**: 82.74%
- **Correction Rate**: 62% (31/50 samples modified)

**Error Breakdown:**
- Perfect corrections: 20%
- Missed corrections: 38% (should have corrected but didn't)
- Incorrect corrections: 42% (made wrong corrections)

#### Performance Analysis:
- **Strengths**: Good at simple grammatical fixes (particles, verb forms)
- **Weaknesses**: Struggles with complex sentence restructuring and context-dependent corrections
- **Recommendation**: Model needs improvement for production use (target: 60-80% accuracy)

### Example Evaluation Output

```
🎯 GRAMMAR CORRECTION ANALYSIS:
Total examples: 50
Exact matches: 10 (20.0%)

📊 Error Breakdown:
✅ Perfect corrections: 10 (20.0%)
✅ Correct no-change: 0 (0.0%)
❌ Missed corrections: 19 (38.0%)
❌ Incorrect corrections: 21 (42.0%)
⚠️  Unnecessary changes: 0 (0.0%)

🔍 EXAMPLE ANALYSIS:
❌ Missed Corrections:
  1. 'コーヒーは苦く美味しくない。' → should be 'コーヒーは苦くて美味しくない。'
  2. '母と父もアメリカ人です。' → should be '母と父はアメリカ人です。'

❌ Incorrect Corrections:
  1. '私の父はおおきくです。' → expected '私の父はおおきいです。' but got '私の父はおおきです。'
  2. 'りょうで友だちに会た。' → expected 'りょうで友だちができた。' but got 'りょうで友だちに会った。'
```

## API Reference

### Core Classes

#### `TrainingPipeline`

Main pipeline class for end-to-end training.

```python
from scripts.complete_training_pipeline import TrainingPipeline

pipeline = TrainingPipeline("config/lora_config.yaml", "output_dir")
success = pipeline.run_complete_pipeline()
```

#### Data Processing Functions

```python
from utils.gec_parser import parse_gec_corpus
from utils.data_utils import create_training_prompt

# Parse GEC corpus
pairs = parse_gec_corpus("corpus.txt")

# Create training prompt
prompt = create_training_prompt("error_text", "correct_text")
```

### Command Line Tools

#### Complete Pipeline
```bash
python scripts/complete_training_pipeline.py [OPTIONS]
```

#### Training Only
```bash
python scripts/train_japanese_gec.py [OPTIONS]
```

#### Batch Inference
```bash
python scripts/batch_inference.py [OPTIONS]
```

#### Interactive Correction
```bash
python scripts/interactive_correction.py [OPTIONS]
```

#### Evaluation
```bash
python scripts/evaluate_gec.py [OPTIONS]
```

## Testing

### Run Integration Tests

```bash
# Run all integration tests
python scripts/run_integration_tests.py

# Run smoke tests only
python scripts/run_integration_tests.py --smoke-tests

# Run specific test class
python scripts/run_integration_tests.py --test-class TestIntegrationPipeline

# Generate test report
python scripts/run_integration_tests.py --report-file results/test_report.txt
```

### Run Unit Tests

```bash
# Run all unit tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_data_preprocessing.py -v

# Run with coverage
python -m pytest tests/ --cov=utils --cov=scripts
```

## Troubleshooting

### Common Issues

#### 1. MLX Installation Issues
```bash
# Ensure you're on Apple Silicon Mac
uname -m  # Should show "arm64"

# Reinstall MLX
pip uninstall mlx-lm
pip install mlx-lm
```

#### 2. Memory Issues During Training
```yaml
# Reduce batch size in config
batch_size: 2  # or 1

# Enable gradient checkpointing
grad_checkpoint: true
```

#### 3. Corpus Parsing Errors
```bash
# Check corpus file encoding
file -I exclude/japanese_gec_corpus/corpus_v0.txt

# Should be UTF-8 encoded
```

#### 4. Model Loading Errors
```bash
# Verify model files exist
ls -la models/japanese-gec-lora/

# Should contain:
# - adapters.safetensors
# - adapter_config.json
```

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
python scripts/complete_training_pipeline.py --log-level DEBUG
```

### Performance Optimization

#### For Faster Training:
- Reduce `iters` in configuration
- Increase `batch_size` if memory allows
- Use smaller `lora_rank`

#### For Better Quality:
- Increase `iters` for more training
- Use higher `lora_rank` (16-32)
- Lower `learning_rate` for stability

### Model Improvement Recommendations

Based on current evaluation results (20% exact accuracy), consider these improvements:

#### 1. Data Quality Enhancement
```bash
# Improve prompt format and regenerate datasets
# Focus on clearer error-correction pairs
# Add more diverse grammatical error types
```

#### 2. Training Configuration Optimization
```bash
# Use enhanced configuration for better performance
python scripts/train_japanese_gec.py --config config/enhanced_config.yaml
```

#### 3. Evaluation Strategy
```bash
# Use strict grammar-focused evaluation for accurate assessment
python scripts/grammar_focused_evaluation.py

# Compare with flexible evaluation for development insights
python scripts/flexible_evaluation.py
```

#### 4. Iterative Improvement Process
1. **Analyze errors** using grammar-focused evaluation
2. **Adjust training data** based on common error patterns
3. **Retrain model** with improved configuration
4. **Re-evaluate** and repeat until target accuracy (60-80%) is achieved

## Project Structure

```
japanese-grammar-correction/
├── .kiro/                          # Kiro IDE configuration
│   └── specs/                      # Feature specifications
├── .venv/                          # Python virtual environment
├── .vscode/                        # VS Code settings
│   └── settings.json
├── config/                         # Configuration files
│   ├── __init__.py
│   ├── config.py                   # Configuration loader
│   ├── paths.py                    # Path configurations
│   ├── training_config.py          # Training configuration class
│   ├── simple_config.yaml          # Simple LoRA configuration
│   ├── improved_config.yaml        # Improved training settings
│   ├── enhanced_config.yaml        # Enhanced configuration
│   ├── optimized_config.yaml       # Optimized parameters
│   ├── optimized_v2_config.yaml    # Version 2 optimized
│   ├── better_config.yaml          # Better performance config
│   └── lora_config.yaml           # Main LoRA configuration
├── datasets/                       # Training datasets
│   ├── .gitkeep
│   ├── train.jsonl                 # Training data (JSONL format)
│   ├── valid.jsonl                 # Validation data
│   └── test.jsonl                  # Test data
├── docs/                          # Documentation
│   ├── training_guide.md          # Training guide
│   ├── usage_examples.md          # Usage examples
│   ├── batch_inference_usage.md   # Batch inference guide
│   └── evaluation_usage.md        # Evaluation guide
├── exclude/                       # External data (not in git)
│   ├── japanese_gec_corpus/       # GEC corpus data
│   └── mlx-lm/                    # MLX-LM source (if needed)
├── logs/                          # Log files
│   ├── training.log               # Training logs
│   ├── evaluation.log             # Evaluation logs
│   ├── batch_inference.log        # Batch inference logs
│   ├── fast_batch_inference.log   # Fast inference logs
│   ├── interactive_correction.log # Interactive session logs
│   └── pipeline.log               # Pipeline execution logs
├── models/                        # Model files and adapters
│   ├── .gitkeep
│   ├── adapters/                  # Base adapter directory
│   ├── japanese-gec-lora-simple/  # Simple LoRA model
│   ├── japanese-gec-lora-improved/ # Improved model
│   ├── japanese-gec-lora-better/  # Better performance model
│   ├── japanese-gec-lora-optimized/ # Optimized model
│   └── japanese-gec-lora-optimized-v2/ # Version 2 optimized
├── optimized_training/            # Optimized training experiments
├── results/                       # Evaluation results and outputs
│   ├── test_predictions_simple.jsonl      # Simple model predictions
│   ├── test_predictions_simple_summary.json # Summary
│   ├── fast_test_predictions.jsonl        # Fast inference results
│   ├── fast_test_predictions_summary.json # Fast summary
│   ├── fast_evaluation_report.json        # Fast evaluation report
│   ├── fast_evaluation_report_summary.txt # Fast summary text
│   ├── example_evaluation_report.json     # Example evaluation
│   └── example_evaluation_report_summary.txt # Example summary
├── scripts/                       # Main execution scripts
│   ├── __init__.py
│   ├── complete_training_pipeline.py      # End-to-end pipeline
│   ├── train_japanese_gec.py             # Model training
│   ├── batch_inference.py               # Batch processing
│   ├── fast_batch_inference.py          # Fast batch inference
│   ├── flexible_evaluation.py           # Flexible evaluation metrics
│   ├── grammar_focused_evaluation.py    # Grammar-focused evaluation
│   ├── interactive_correction.py        # Interactive correction tool
│   ├── batch_file_correction.py         # File batch correction
│   ├── evaluate_gec.py                  # Model evaluation
│   ├── prepare_model.py                 # Model preparation
│   ├── run_batch_inference_example.py   # Batch inference example
│   ├── run_evaluation_example.py        # Evaluation example
│   ├── run_integration_tests.py         # Integration testing
│   └── setup_models.sh                  # Model setup script
├── test_training/                 # Test training experiments
├── tests/                         # Unit and integration tests
│   ├── __init__.py
│   ├── test_batch_inference.py    # Batch inference tests
│   ├── test_data_preprocessing.py # Data preprocessing tests
│   ├── test_evaluate_gec.py       # Evaluation tests
│   └── test_integration_pipeline.py # Integration tests
├── utils/                         # Utility modules
│   ├── __init__.py
│   ├── data_utils.py              # Data processing utilities
│   ├── dataset_splitter.py        # Dataset splitting utilities
│   ├── file_utils.py              # File handling utilities
│   ├── gec_parser.py              # GEC corpus parser
│   ├── logging_utils.py           # Logging configuration
│   └── mlx_formatter.py           # MLX format utilities
├── requirements.txt               # Python dependencies
├── setup_project.py              # Project setup script
└── README.md                     # This documentation
```

### Key Directories Explained

#### Configuration (`config/`)
- **Multiple config files**: Different training configurations for various experiments
- **YAML format**: Easy-to-modify training parameters
- **Modular design**: Separate configs for different model versions

#### Models (`models/`)
- **Multiple model versions**: Different LoRA adaptations with varying performance
- **Organized by experiment**: Each model has its own directory
- **Adapter format**: MLX-compatible LoRA adapters

#### Scripts (`scripts/`)
- **Complete pipeline**: End-to-end training and evaluation
- **Modular tools**: Individual scripts for specific tasks
- **Evaluation tools**: Multiple evaluation approaches (flexible, grammar-focused)
- **Interactive tools**: Real-time correction interface

#### Results (`results/`)
- **Structured outputs**: JSON and JSONL format results
- **Summary reports**: Human-readable evaluation summaries
- **Prediction files**: Model outputs for analysis

#### Utils (`utils/`)
- **Data processing**: GEC corpus parsing and formatting
- **File handling**: Robust file I/O operations
- **Logging**: Centralized logging configuration
- **MLX integration**: MLX-specific formatting utilities

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Run tests before committing
python scripts/run_integration_tests.py --smoke-tests
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MLX framework by Apple for efficient model training on Apple Silicon
- Qwen3 model by Alibaba for the base language model
- Japanese GEC corpus contributors for the training data

## Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the documentation in the `docs/` directory

---

**Note**: This system is designed for educational and research purposes. For production use, additional validation and testing are recommended.