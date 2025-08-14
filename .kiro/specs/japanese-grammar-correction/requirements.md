# Requirements Document

## Introduction

This feature implements a Japanese grammar correction system that uses MLX LoRA fine-tuning to adapt the Qwen3-0.6B-4bit model for Japanese grammatical error correction (GEC). The system will leverage the Japanese GEC corpus data to train a model that can automatically detect and correct grammatical errors in Japanese text input from users.

## Requirements

### Requirement 1

**User Story:** As a Japanese language learner, I want to input Japanese text and receive grammatically corrected output, so that I can improve my Japanese writing skills.

#### Acceptance Criteria

1. WHEN a user inputs Japanese text THEN the system SHALL process the text through the fine-tuned model
2. WHEN the model processes the text THEN the system SHALL return grammatically corrected Japanese text
3. WHEN the input text has no grammatical errors THEN the system SHALL return the original text unchanged
4. WHEN the input text contains multiple errors THEN the system SHALL correct all detectable errors in a single pass

### Requirement 2

**User Story:** As a developer, I want to fine-tune the Qwen3-0.6B-4bit model using the Japanese GEC corpus, so that the model can perform accurate Japanese grammar correction.

#### Acceptance Criteria

1. WHEN the fine-tuning process starts THEN the system SHALL load the Japanese GEC corpus data from the exclude/japanese_gec_corpus directory
2. WHEN processing the corpus data THEN the system SHALL format it appropriately for MLX LoRA training
3. WHEN training begins THEN the system SHALL use mlx_lm.lora to fine-tune the Qwen3-0.6B-4bit model
4. WHEN training completes THEN the system SHALL save the fine-tuned model weights for inference
5. IF training fails THEN the system SHALL provide clear error messages and recovery options

### Requirement 3

**User Story:** As a system administrator, I want the grammar correction system to handle various input formats and edge cases, so that it provides reliable service to users.

#### Acceptance Criteria

1. WHEN a user inputs empty text THEN the system SHALL return an appropriate message indicating no input was provided
2. WHEN a user inputs non-Japanese text THEN the system SHALL handle it gracefully without crashing
3. WHEN a user inputs extremely long text THEN the system SHALL process it within reasonable memory and time constraints
4. WHEN the model is not available or fails to load THEN the system SHALL provide clear error messages

### Requirement 4

**User Story:** As a developer, I want to evaluate the model's performance on grammar correction tasks, so that I can assess the quality of the fine-tuned model.

#### Acceptance Criteria

1. WHEN evaluation is requested THEN the system SHALL use a held-out test set from the Japanese GEC corpus
2. WHEN running evaluation THEN the system SHALL calculate standard GEC metrics (precision, recall, F1)
3. WHEN evaluation completes THEN the system SHALL provide detailed performance reports
4. WHEN comparing models THEN the system SHALL support before/after fine-tuning comparisons

### Requirement 5

**User Story:** As a user, I want to interact with the grammar correction system through a simple interface, so that I can easily correct my Japanese text.

#### Acceptance Criteria

1. WHEN the system starts THEN it SHALL provide a clear interface for text input
2. WHEN a user submits text THEN the system SHALL display the corrected output clearly
3. WHEN processing takes time THEN the system SHALL show appropriate loading indicators
4. WHEN errors occur THEN the system SHALL display user-friendly error messages