# Japanese Grammar Error Detection Training Guide

This guide explains how to train a model specifically for the DETECT task using the corrected LoRA parameters.

## Problem Fixed

The previous configuration had LoRA parameters that weren't being passed to MLX LoRA. The training script now:

1. **Creates MLX-compatible config files** with proper LoRA parameters
2. **Uses the `--config` flag** instead of individual command-line arguments
3. **Maps LoRA parameters correctly**:
   - `lora_alpha` → `scale` (MLX terminology)
   - `lora_rank` → `rank`
   - `lora_dropout` → `dropout`

## Updated Configuration

The `config/conservative_config.yaml` now includes:

```yaml
# LoRA parameters for detection task
lora_rank: 16          # Standard rank for detection
lora_alpha: 32         # 2x rank (MLX uses this as 'scale')
lora_dropout: 0.1
lora_layers: 16        # Number of layers to apply LoRA to

# Training parameters optimized for detection task
batch_size: 4
learning_rate: 2e-5    # Higher learning rate for detection task
iters: 2000           # More iterations for complex detection task
```

## Training Command

```bash
python scripts/train_japanese_gec.py --config config/conservative_config.yaml
```

## What Happens Now

1. **Config File Generation**: The script creates a temporary MLX config file with proper LoRA parameters
2. **MLX Training**: Uses `mlx_lm lora --config <generated_config.yaml>`
3. **Parameter Mapping**: 
   - `lora_rank: 16` → `lora_parameters.rank: 16`
   - `lora_alpha: 32` → `lora_parameters.scale: 32`
   - `lora_dropout: 0.1` → `lora_parameters.dropout: 0.1`

## Expected Improvements

With proper LoRA parameters, the DETECT task should show:
- Better error detection precision and recall
- Improved F1-Score (target: >0.6)
- More accurate error marking with `<>` brackets

## Monitoring Training

The training will create:
- Model adapters in `models/japanese-gec-detect-v1/`
- MLX config file: `models/japanese-gec-detect-v1_mlx_config.yaml`
- Training logs with proper LoRA parameter reporting

## Evaluation

After training, evaluate with:

```bash
python scripts/grammar_focused_evaluation.py \
  --model-path models/japanese-gec-detect-v1 \
  --task-filter DETECT \
  --max-examples 100
```

The evaluation should now show significantly improved detection metrics compared to the previous 21.1% F1-Score.