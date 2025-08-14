# Usage Examples for Japanese Grammar Correction

This document provides comprehensive examples of how to use the Japanese Grammar Correction system for various scenarios and use cases.

## Table of Contents

- [Basic Usage](#basic-usage)
- [Training Examples](#training-examples)
- [Inference Examples](#inference-examples)
- [Evaluation Examples](#evaluation-examples)
- [Advanced Usage](#advanced-usage)
- [Integration Examples](#integration-examples)
- [Troubleshooting Examples](#troubleshooting-examples)

## Basic Usage

### Quick Start Example

```bash
# 1. Set up the environment
python setup_project.py

# 2. Run complete pipeline with sample data
python scripts/complete_training_pipeline.py \
    --max-samples 100 \
    --config config/lora_config.yaml

# 3. Test interactive correction
python scripts/interactive_correction.py
```

### Interactive Grammar Correction

```bash
# Start interactive session
python scripts/interactive_correction.py \
    --model mlx-community/Qwen3-0.6B-4bit \
    --adapter-path models/japanese-gec-lora

# Example interaction:
# Input: 私は学校に行く。
# Output: 私は学校に行きます。

# Input: これは私の本。
# Output: これは私の本です。

# Input: 彼女はとてもきれい。
# Output: 彼女はとてもきれいです。
```

### Batch File Processing

```bash
# Create input file
cat > input.txt << EOF
私は昨日映画を見る。
彼は学校に行く。
これは私の本。
今日は天気がいい。
EOF

# Process the file
python scripts/batch_file_correction.py \
    --input input.txt \
    --output corrected.txt \
    --model mlx-community/Qwen3-0.6B-4bit \
    --adapter-path models/japanese-gec-lora

# View results
cat corrected.txt
# Output:
# 私は昨日映画を見ました。
# 彼は学校に行きます。
# これは私の本です。
# 今日は天気がいいです。
```

## Training Examples

### Complete Pipeline Training

```bash
# Basic training with default settings
python scripts/complete_training_pipeline.py

# Training with custom configuration
python scripts/complete_training_pipeline.py \
    --config config/custom_config.yaml \
    --output-dir my_training_$(date +%Y%m%d) \
    --log-level DEBUG

# Training with limited data for testing
python scripts/complete_training_pipeline.py \
    --max-samples 500 \
    --train-ratio 0.7 \
    --valid-ratio 0.2 \
    --test-ratio 0.1
```

### Custom Configuration Training

Create a custom configuration file:

```yaml
# config/custom_config.yaml
model: "mlx-community/Qwen3-0.6B-4bit"
lora_rank: 32          # Higher rank for better quality
lora_alpha: 64
learning_rate: 0.00005 # Lower learning rate for stability
batch_size: 2          # Smaller batch for memory efficiency
iters: 3000           # More iterations for better convergence
adapter_path: "models/custom-gec-lora"
```

```bash
# Train with custom configuration
python scripts/complete_training_pipeline.py \
    --config config/custom_config.yaml
```

### Incremental Training

```bash
# Phase 1: Train on basic grammar errors
python scripts/complete_training_pipeline.py \
    --config config/phase1_config.yaml \
    --corpus-path exclude/japanese_gec_corpus/basic_errors.txt \
    --output-dir phase1_training

# Phase 2: Continue training on complex errors
python scripts/train_japanese_gec.py \
    --config config/phase2_config.yaml \
    --resume-from phase1_training/adapters.safetensors
```

### Training with Different Data Splits

```bash
# 90% training, 5% validation, 5% test
python scripts/complete_training_pipeline.py \
    --train-ratio 0.9 \
    --valid-ratio 0.05 \
    --test-ratio 0.05

# 60% training, 20% validation, 20% test (for extensive validation)
python scripts/complete_training_pipeline.py \
    --train-ratio 0.6 \
    --valid-ratio 0.2 \
    --test-ratio 0.2
```

## Inference Examples

### Batch Inference on Test Data

```bash
# Run inference on test dataset
python scripts/batch_inference.py \
    --model mlx-community/Qwen3-0.6B-4bit \
    --adapter-path models/japanese-gec-lora \
    --input-file datasets/test.jsonl \
    --output-file results/test_predictions.jsonl \
    --batch-size 4 \
    --max-tokens 512 \
    --temperature 0.1
```

### Custom Input Format Processing

```bash
# Create custom input file (plain text, one sentence per line)
cat > custom_input.txt << EOF
私は昨日友達と映画を見る。
彼女は毎日日本語を勉強する。
今日は雨が降った。
私たちは来週旅行に行く予定。
EOF

# Convert to JSONL format for batch processing
python -c "
import json
with open('custom_input.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

with open('custom_input.jsonl', 'w', encoding='utf-8') as f:
    for i, line in enumerate(lines):
        if line.strip():
            data = {
                'id': i,
                'input': line.strip(),
                'expected': ''  # Empty for inference-only
            }
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
"

# Run batch inference
python scripts/batch_inference.py \
    --model mlx-community/Qwen3-0.6B-4bit \
    --adapter-path models/japanese-gec-lora \
    --input-file custom_input.jsonl \
    --output-file custom_predictions.jsonl
```

### Real-time Processing Example

```python
# real_time_correction.py
import sys
sys.path.append('.')

from scripts.interactive_correction import GrammarCorrector

# Initialize corrector
corrector = GrammarCorrector(
    model_name="mlx-community/Qwen3-0.6B-4bit",
    adapter_path="models/japanese-gec-lora"
)

# Example sentences to correct
sentences = [
    "私は毎日学校に行く。",
    "彼女はとてもきれい。",
    "今日は天気がいい。",
    "私たちは友達と遊ぶ。"
]

print("Real-time Grammar Correction Examples:")
print("=" * 40)

for sentence in sentences:
    corrected = corrector.correct_text(sentence)
    print(f"Original:  {sentence}")
    print(f"Corrected: {corrected}")
    print("-" * 40)
```

## Evaluation Examples

### Comprehensive Evaluation

```bash
# Run complete evaluation pipeline
python scripts/evaluate_gec.py \
    --results-file results/test_predictions.jsonl \
    --output-file results/comprehensive_evaluation.json \
    --log-level INFO

# View evaluation summary
cat results/comprehensive_evaluation_summary.txt
```

### Comparative Evaluation

```bash
# Evaluate baseline model (without fine-tuning)
python scripts/batch_inference.py \
    --model mlx-community/Qwen3-0.6B-4bit \
    --input-file datasets/test.jsonl \
    --output-file results/baseline_predictions.jsonl

# Evaluate fine-tuned model
python scripts/batch_inference.py \
    --model mlx-community/Qwen3-0.6B-4bit \
    --adapter-path models/japanese-gec-lora \
    --input-file datasets/test.jsonl \
    --output-file results/finetuned_predictions.jsonl

# Compare results
python scripts/evaluate_gec.py \
    --results-file results/baseline_predictions.jsonl \
    --output-file results/baseline_evaluation.json

python scripts/evaluate_gec.py \
    --results-file results/finetuned_predictions.jsonl \
    --output-file results/finetuned_evaluation.json

# Generate comparison report
python -c "
import json

with open('results/baseline_evaluation.json', 'r') as f:
    baseline = json.load(f)
with open('results/finetuned_evaluation.json', 'r') as f:
    finetuned = json.load(f)

print('Model Comparison Results:')
print('=' * 30)
print(f'Baseline F1 Score: {baseline[\"token_level_metrics\"][\"f1_score\"]:.4f}')
print(f'Fine-tuned F1 Score: {finetuned[\"token_level_metrics\"][\"f1_score\"]:.4f}')
print(f'Improvement: {finetuned[\"token_level_metrics\"][\"f1_score\"] - baseline[\"token_level_metrics\"][\"f1_score\"]:.4f}')
"
```

### Error Analysis

```bash
# Generate detailed error analysis
python scripts/evaluate_gec.py \
    --results-file results/test_predictions.jsonl \
    --output-file results/error_analysis.json

# Extract specific error types
python -c "
import json

with open('results/error_analysis.json', 'r') as f:
    data = json.load(f)

error_analysis = data['error_analysis']
print('Error Type Analysis:')
print('=' * 25)
print(f'Perfect Matches: {error_analysis[\"perfect_matches\"]}')
print(f'Partial Corrections: {error_analysis[\"partial_corrections\"]}')
print(f'No Corrections: {error_analysis[\"no_corrections\"]}')
print(f'Overcorrections: {error_analysis[\"overcorrections\"]}')
print(f'Failed Predictions: {error_analysis[\"failed_predictions\"]}')
"
```

## Advanced Usage

### Multi-Model Ensemble

```python
# ensemble_correction.py
import sys
sys.path.append('.')

from scripts.interactive_correction import GrammarCorrector
import statistics

class EnsembleCorrector:
    def __init__(self, model_paths):
        self.correctors = []
        for model_path in model_paths:
            corrector = GrammarCorrector(
                model_name="mlx-community/Qwen3-0.6B-4bit",
                adapter_path=model_path
            )
            self.correctors.append(corrector)
    
    def correct_text(self, text):
        corrections = []
        for corrector in self.correctors:
            correction = corrector.correct_text(text)
            corrections.append(correction)
        
        # Simple voting mechanism (return most common correction)
        if len(set(corrections)) == 1:
            return corrections[0]
        else:
            # If no consensus, return first correction
            return corrections[0]

# Usage
ensemble = EnsembleCorrector([
    "models/japanese-gec-lora-v1",
    "models/japanese-gec-lora-v2",
    "models/japanese-gec-lora-v3"
])

text = "私は学校に行く。"
corrected = ensemble.correct_text(text)
print(f"Ensemble correction: {corrected}")
```

### Custom Data Processing Pipeline

```python
# custom_pipeline.py
import sys
sys.path.append('.')

from utils.gec_parser import parse_gec_corpus
from utils.data_utils import create_training_prompt
import json

def create_custom_dataset(corpus_file, output_file, filter_func=None):
    """Create custom dataset with filtering."""
    
    # Parse corpus
    pairs = parse_gec_corpus(corpus_file)
    
    # Apply custom filtering
    if filter_func:
        pairs = [pair for pair in pairs if filter_func(pair)]
    
    # Convert to training format
    training_data = []
    for error_text, correct_text in pairs:
        prompt_data = create_training_prompt(error_text, correct_text)
        training_data.append(prompt_data)
    
    # Save dataset
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in training_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Created dataset with {len(training_data)} samples")

# Example: Filter for particle errors only
def particle_error_filter(pair):
    error_text, correct_text = pair
    # Simple heuristic: look for particle differences
    particles = ['は', 'が', 'を', 'に', 'で', 'と', 'から', 'まで']
    for particle in particles:
        if particle in error_text and particle in correct_text:
            return True
    return False

# Create particle-specific dataset
create_custom_dataset(
    "exclude/japanese_gec_corpus/corpus_v0.txt",
    "datasets/particle_errors.jsonl",
    particle_error_filter
)
```

### Automated Training Pipeline

```bash
#!/bin/bash
# automated_training.sh

# Configuration
CORPUS_PATH="exclude/japanese_gec_corpus/corpus_v0.txt"
OUTPUT_BASE="automated_training_$(date +%Y%m%d_%H%M%S)"
CONFIG_FILE="config/lora_config.yaml"

echo "Starting automated training pipeline..."
echo "Output directory: $OUTPUT_BASE"

# Step 1: Data preprocessing
echo "Step 1: Data preprocessing..."
python scripts/complete_training_pipeline.py \
    --skip-training \
    --skip-evaluation \
    --corpus-path "$CORPUS_PATH" \
    --output-dir "$OUTPUT_BASE" \
    --config "$CONFIG_FILE"

if [ $? -ne 0 ]; then
    echo "Data preprocessing failed!"
    exit 1
fi

# Step 2: Model training
echo "Step 2: Model training..."
python scripts/complete_training_pipeline.py \
    --skip-preprocessing \
    --skip-evaluation \
    --output-dir "$OUTPUT_BASE" \
    --config "$CONFIG_FILE"

if [ $? -ne 0 ]; then
    echo "Model training failed!"
    exit 1
fi

# Step 3: Evaluation
echo "Step 3: Model evaluation..."
python scripts/complete_training_pipeline.py \
    --skip-preprocessing \
    --skip-training \
    --output-dir "$OUTPUT_BASE" \
    --config "$CONFIG_FILE"

if [ $? -ne 0 ]; then
    echo "Model evaluation failed!"
    exit 1
fi

echo "Automated training pipeline completed successfully!"
echo "Results available in: $OUTPUT_BASE"
```

## Integration Examples

### Web API Integration

```python
# web_api.py
from flask import Flask, request, jsonify
import sys
sys.path.append('.')

from scripts.interactive_correction import GrammarCorrector

app = Flask(__name__)

# Initialize corrector
corrector = GrammarCorrector(
    model_name="mlx-community/Qwen3-0.6B-4bit",
    adapter_path="models/japanese-gec-lora"
)

@app.route('/correct', methods=['POST'])
def correct_grammar():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        corrected = corrector.correct_text(text)
        
        return jsonify({
            'original': text,
            'corrected': corrected,
            'status': 'success'
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
```

```bash
# Start the web API
python web_api.py

# Test the API
curl -X POST http://localhost:5000/correct \
    -H "Content-Type: application/json" \
    -d '{"text": "私は学校に行く。"}'

# Response:
# {
#   "original": "私は学校に行く。",
#   "corrected": "私は学校に行きます。",
#   "status": "success"
# }
```

### Batch Processing Script

```python
# batch_processor.py
import argparse
import json
import sys
from pathlib import Path

sys.path.append('.')
from scripts.interactive_correction import GrammarCorrector

def process_file(input_file, output_file, model_path):
    """Process a file with Japanese text for grammar correction."""
    
    corrector = GrammarCorrector(
        model_name="mlx-community/Qwen3-0.6B-4bit",
        adapter_path=model_path
    )
    
    results = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            text = line.strip()
            if text:
                try:
                    corrected = corrector.correct_text(text)
                    results.append({
                        'line_number': line_num,
                        'original': text,
                        'corrected': corrected
                    })
                except Exception as e:
                    results.append({
                        'line_number': line_num,
                        'original': text,
                        'error': str(e)
                    })
    
    # Save results
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(results)} lines")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process Japanese text for grammar correction")
    parser.add_argument("--input", required=True, help="Input text file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--model-path", default="models/japanese-gec-lora", help="Model adapter path")
    
    args = parser.parse_args()
    
    process_file(args.input, args.output, args.model_path)
```

## Troubleshooting Examples

### Memory Issues

```bash
# Check available memory
python -c "
import mlx.core as mx
print(f'Available memory: {mx.metal.get_active_memory() / 1024**3:.2f} GB')
"

# Reduce memory usage
python scripts/complete_training_pipeline.py \
    --config config/low_memory_config.yaml

# config/low_memory_config.yaml
# batch_size: 1
# grad_checkpoint: true
# max_seq_length: 256
# lora_rank: 8
```

### Data Format Issues

```bash
# Validate corpus format
python -c "
import sys
sys.path.append('.')
from utils.gec_parser import parse_gec_corpus

try:
    pairs = parse_gec_corpus('exclude/japanese_gec_corpus/corpus_v0.txt')
    print(f'Successfully parsed {len(pairs)} correction pairs')
    
    # Show first few examples
    for i, (error, correct) in enumerate(pairs[:3]):
        print(f'Example {i+1}:')
        print(f'  Error: {error}')
        print(f'  Correct: {correct}')
        print()
        
except Exception as e:
    print(f'Error parsing corpus: {e}')
"
```

### Model Loading Issues

```bash
# Test model loading
python -c "
try:
    from mlx_lm import load
    model, tokenizer = load('mlx-community/Qwen3-0.6B-4bit')
    print('Base model loaded successfully')
    
    # Test with adapters
    model, tokenizer = load(
        'mlx-community/Qwen3-0.6B-4bit',
        adapter_path='models/japanese-gec-lora'
    )
    print('Model with adapters loaded successfully')
    
except Exception as e:
    print(f'Model loading error: {e}')
"
```

### Performance Optimization

```bash
# Profile training performance
python -m cProfile -o training_profile.prof scripts/train_japanese_gec.py \
    --config config/lora_config.yaml \
    --dry-run

# Analyze profile
python -c "
import pstats
p = pstats.Stats('training_profile.prof')
p.sort_stats('cumulative').print_stats(20)
"

# Benchmark inference speed
python -c "
import time
import sys
sys.path.append('.')

from scripts.interactive_correction import GrammarCorrector

corrector = GrammarCorrector(
    model_name='mlx-community/Qwen3-0.6B-4bit',
    adapter_path='models/japanese-gec-lora'
)

test_sentences = [
    '私は学校に行く。',
    '彼女はとてもきれい。',
    '今日は天気がいい。'
] * 10  # 30 sentences total

start_time = time.time()
for sentence in test_sentences:
    corrected = corrector.correct_text(sentence)

end_time = time.time()
total_time = end_time - start_time
avg_time = total_time / len(test_sentences)

print(f'Total time: {total_time:.2f} seconds')
print(f'Average time per sentence: {avg_time:.3f} seconds')
print(f'Sentences per second: {1/avg_time:.2f}')
"
```

These examples provide comprehensive coverage of the Japanese Grammar Correction system usage, from basic operations to advanced integration scenarios. Use them as templates for your specific use cases and modify as needed.