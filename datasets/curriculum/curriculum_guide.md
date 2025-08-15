# Curriculum Learning Guide

## Training Stages

Train the model in the following order:

1. **stage1_basic_correction**: gec_end_to_end
2. **stage2_error_detection**: gec_error_detection
3. **stage3_precise_correction**: gec_error_correction
4. **stage4_quality_assessment**: quality_assessment

## Usage

```bash
# Stage 1: Basic correction ability
mlx_lm.lora --data datasets/curriculum/stage1_basic_correction

# Stage 2: Error detection ability (continue from stage 1)
mlx_lm.lora --data datasets/curriculum/stage2_error_detection --resume

# Continue with stages 3 and 4...
```
