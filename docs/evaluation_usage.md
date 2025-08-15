# Grammar-Focused Evaluation Usage

The updated `grammar_focused_evaluation.py` script now uses the centralized prompt management system.

## Features

- **Consistent Prompts**: Uses `config/prompts.py` and `config/prompt_config.yaml`
- **Multi-task Support**: Can filter by task type (FIX, DETECT, CORRECT, ASSESS)
- **Flexible Configuration**: Command-line arguments for all settings
- **Detailed Analysis**: Categorizes different types of errors

## Usage

### Basic Usage
```bash
python scripts/grammar_focused_evaluation.py
```

### Custom Configuration
```bash
python scripts/grammar_focused_evaluation.py \
  --model-path models/my-model \
  --test-data datasets/combined/test.jsonl \
  --max-examples 100 \
  --task-filter FIX \
  --prompt-config config/prompt_config.yaml
```

### Available Options

- `--model-path`: Path to trained model adapters (default: `models/japanese-gec-v1`)
- `--base-model`: Base model to use (default: `mlx-community/Qwen3-0.6B-4bit`)
- `--test-data`: Path to test data (default: `datasets/combined/test.jsonl`)
- `--max-examples`: Maximum examples to evaluate (default: 50)
- `--task-filter`: Task type to filter - FIX, DETECT, CORRECT, ASSESS (default: FIX)
- `--prompt-config`: Path to prompt configuration (default: `config/prompt_config.yaml`)

## Task Types and Metrics

### **FIX** (End-to-end Grammar Correction)
- **Primary Metric**: Exact match accuracy
- **Analysis**: Error categorization (perfect corrections, missed corrections, etc.)
- **Focus**: Whether the model produces the exact expected correction

### **DETECT** (Error Detection with Marking)
- **Primary Metrics**: Precision, Recall, F1-Score
- **Analysis**: Error detection performance at token level
- **Focus**: How well the model identifies and marks errors with `<>` brackets

### **CORRECT** (Correction of Marked Errors)
- **Primary Metric**: Exact match accuracy
- **Analysis**: Correction completion rate (removing markers)
- **Focus**: Whether the model correctly fixes marked errors

### **ASSESS** (Quality Assessment)
- **Primary Metrics**: Score accuracy, Mean Absolute Error (MAE)
- **Analysis**: Close predictions (±0.5 score difference)
- **Focus**: How accurately the model predicts quality scores (1-4)

## Output

The script provides task-specific analysis:

### FIX Task Output
- Exact match accuracy
- Error breakdown by category
- Examples of missed/incorrect corrections

### DETECT Task Output  
- Average Precision, Recall, F1-Score
- Perfect detection rate
- Error marking analysis

### CORRECT Task Output
- Correction completion rate
- Marker removal success
- Final output accuracy

### ASSESS Task Output
- Score prediction accuracy
- Mean Absolute Error
- Close prediction rate (±0.5)

## Integration with Prompt System

The script automatically:
1. Loads prompt configuration from YAML
2. Determines language (English/Japanese) from config
3. Uses task-specific system prompts and user prompts for each evaluation type:

   **System Prompts:**
   - **FIX**: "You are a Japanese grammar correction specialist. Correct grammatical errors..."
   - **DETECT**: "You are a Japanese grammar error detection specialist. Mark grammatical errors..."
   - **CORRECT**: "You are a Japanese grammar correction specialist. Correct the grammatical errors marked..."
   - **ASSESS**: "You are a Japanese grammar correction quality assessor. Evaluate the quality..."

   **User Prompts (with task prefixes):**
   - **FIX**: "[FIX] Correct the grammar in this Japanese sentence: {input_text}"
   - **DETECT**: "[DETECT] Mark the grammatical errors in this Japanese sentence: {input_text}"
   - **CORRECT**: "[CORRECT] Correct the marked errors in this Japanese sentence: {input_text}"
   - **ASSESS**: "[ASSESS] Assess the quality of this correction:\nOriginal: {source_text}\nCorrected: {corrected_text}"
4. Handles multi-task data format with task prefixes
5. Maintains consistency with training data prompts

## Prompt Consistency

The evaluation now uses the exact same system prompts as the training data, ensuring:
- **Accurate evaluation**: Model sees the same instructions during evaluation as training
- **Task-specific behavior**: Each task gets appropriate specialized prompts
- **Consistent results**: Eliminates prompt mismatch as a source of evaluation error