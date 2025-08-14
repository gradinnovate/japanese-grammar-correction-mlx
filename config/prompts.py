"""
Global prompt templates for Japanese Grammar Correction system.

This module contains all system and user prompt templates used across
the training, evaluation, and inference scripts.
"""

# System prompt for Japanese grammar correction
SYSTEM_PROMPT = "あなたは日本語の文法修正を専門とするAIアシスタントです。与えられた文章の文法エラーを正確に修正してください。"

# Alternative English system prompt (for comparison)
SYSTEM_PROMPT_EN = "You are a Japanese grammar correction assistant. Correct grammatical errors in Japanese sentences."

# User prompt template for grammar correction
USER_PROMPT_TEMPLATE = "次の日本語文を文法的に正しく修正してください：{input_text}"

# Alternative English user prompt template
USER_PROMPT_TEMPLATE_EN = "Please correct the grammar in the following Japanese sentence: {input_text}"

# Chat format templates
CHAT_TEMPLATE = "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

# Messages format for training data
def create_messages_format(input_text: str, output_text: str, use_english: bool = False):
    """
    Create messages format for training data.
    
    Args:
        input_text: Input text with grammatical errors
        output_text: Corrected text
        use_english: Whether to use English prompts
        
    Returns:
        Dictionary with messages format
    """
    if use_english:
        system_prompt = SYSTEM_PROMPT_EN
        user_prompt = USER_PROMPT_TEMPLATE_EN.format(input_text=input_text)
    else:
        system_prompt = SYSTEM_PROMPT
        user_prompt = USER_PROMPT_TEMPLATE.format(input_text=input_text)
    
    return {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            },
            {
                "role": "assistant",
                "content": output_text
            }
        ]
    }

# Chat format for inference
def create_chat_prompt(input_text: str, use_english: bool = False):
    """
    Create chat format prompt for inference.
    
    Args:
        input_text: Input text with grammatical errors
        use_english: Whether to use English prompts
        
    Returns:
        Formatted chat prompt string
    """
    if use_english:
        system_message = SYSTEM_PROMPT_EN
        user_message = USER_PROMPT_TEMPLATE_EN.format(input_text=input_text)
    else:
        system_message = SYSTEM_PROMPT
        user_message = USER_PROMPT_TEMPLATE.format(input_text=input_text)
    
    return CHAT_TEMPLATE.format(
        system_message=system_message,
        user_message=user_message
    )

# Prompt configuration
class PromptConfig:
    """Configuration class for prompt settings."""
    
    def __init__(self, use_english: bool = False):
        self.use_english = use_english
        
    @property
    def system_prompt(self):
        return SYSTEM_PROMPT_EN if self.use_english else SYSTEM_PROMPT
    
    @property
    def user_template(self):
        return USER_PROMPT_TEMPLATE_EN if self.use_english else USER_PROMPT_TEMPLATE
    
    def format_user_prompt(self, input_text: str):
        return self.user_template.format(input_text=input_text)
    
    def create_messages(self, input_text: str, output_text: str):
        return create_messages_format(input_text, output_text, self.use_english)
    
    def create_chat_prompt(self, input_text: str):
        return create_chat_prompt(input_text, self.use_english)