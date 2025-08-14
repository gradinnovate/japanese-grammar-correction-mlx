# Implementation Plan

- [x] 1. Set up project structure and data processing utilities
  - Create directory structure for datasets, scripts, and configuration files
  - Set up basic Python utilities for data handling and file management
  - Create configuration files for MLX-LM training parameters
  - _Requirements: 2.2, 3.1_

- [x] 2. Implement Japanese GEC corpus data preprocessing
- [x] 2.1 Create corpus parser for Japanese GEC format
  - Write function to parse tab-separated corpus file with error/correction markers `<>` and `()`
  - Implement extraction of clean error and correct sentence pairs from marked text
  - Add validation for corpus format and UTF-8 encoding handling
  - _Requirements: 2.1, 2.2_

- [x] 2.2 Convert GEC data to MLX-LM compatible format
  - Transform GEC pairs into JSONL format expected by mlx_lm.lora command
  - Create proper prompt-response format for instruction tuning
  - Implement data cleaning and normalization for Japanese text
  - _Requirements: 2.2, 2.3_

- [x] 2.3 Create train/validation/test dataset splits
  - Split the processed corpus into training, validation, and test sets
  - Ensure balanced distribution of error types across splits
  - Save datasets in separate JSONL files for MLX-LM consumption
  - _Requirements: 2.2, 4.1_

- [x] 2.4 Create unit tests for data preprocessing
  - Write tests for corpus parsing with sample Japanese GEC data
  - Test dataset formatting and splitting functions
  - Validate error handling for malformed corpus entries
  - _Requirements: 2.1, 2.2, 3.2_

- [x] 3. Create MLX-LM training configuration and scripts
- [x] 3.1 Create LoRA training configuration file
  - Set up YAML configuration for mlx_lm.lora with appropriate parameters
  - Configure LoRA rank, learning rate, and batch size for Qwen3-0.6B-4bit
  - Add training hyperparameters optimized for Japanese GEC task
  - _Requirements: 2.3, 2.4_

- [x] 3.2 Create training execution script
  - Write Python script that calls mlx_lm.lora with proper arguments
  - Add data path configuration and model output management
  - Implement training progress monitoring and logging
  - _Requirements: 2.3, 2.4, 2.5_

- [x] 3.3 Create model conversion and preparation utilities
  - Download and prepare mlx-community/Qwen3-0.6B-4bit model if needed
  - Create utilities to verify model compatibility and format
  - Add scripts to manage model versions and adapter files
  - _Requirements: 2.4, 3.4_

- [ ] 4. Implement inference and evaluation utilities
- [ ] 4.1 Create inference script using mlx_lm.generate
  - Write script that uses mlx_lm.generate with fine-tuned adapters
  - Implement proper prompt formatting for grammar correction task
  - Add input preprocessing and output postprocessing for Japanese text
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 4.2 Create batch inference for evaluation
  - Implement batch processing of test dataset using inference script
  - Generate predictions for all test examples
  - Save results in format suitable for evaluation metrics calculation
  - _Requirements: 4.1, 4.2_

- [x] 4.3 Implement evaluation metrics calculation
  - Create script to calculate GEC-specific metrics (precision, recall, F1)
  - Add sentence-level accuracy and BLEU score computation
  - Generate detailed evaluation reports comparing predictions to ground truth
  - _Requirements: 4.2, 4.3, 4.4_

- [x] 5. Create user interface and application scripts
- [x] 5.1 Create interactive grammar correction script
  - Implement command-line interface for single text correction
  - Add user-friendly prompts and formatted output display
  - Include error handling for invalid input and model failures
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [x] 5.2 Create batch processing script for file input
  - Implement script to process text files with multiple Japanese sentences
  - Add support for various input formats (txt, csv, json)
  - Generate corrected output files with original/corrected comparisons
  - _Requirements: 1.1, 1.2, 1.3, 1.4_

- [x] 6. Create end-to-end pipeline and validation
- [x] 6.1 Create complete training pipeline script
  - Combine data preprocessing, training, and evaluation into single workflow
  - Add command-line arguments for different training configurations
  - Implement automatic evaluation after training completion
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 6.2 Create integration tests with sample data
  - Test complete pipeline with small subset of Japanese GEC corpus
  - Validate data processing, training execution, and inference quality
  - Create automated tests for different error scenarios and edge cases
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 3.1, 3.2, 3.3, 3.4_

- [x] 6.3 Create documentation and usage examples
  - Write README with setup instructions and usage examples
  - Document the data format requirements and training process
  - Add example commands for training, evaluation, and inference
  - _Requirements: 5.1, 5.2, 5.3, 5.4_