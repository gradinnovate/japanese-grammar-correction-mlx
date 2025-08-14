#!/usr/bin/env python3
"""
Example usage of global prompt constants.

This script demonstrates how to use the global prompt templates
for different scenarios.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.prompts import (
    SYSTEM_PROMPT, SYSTEM_PROMPT_EN,
    USER_PROMPT_TEMPLATE, USER_PROMPT_TEMPLATE_EN,
    create_messages_format, create_chat_prompt, PromptConfig
)

def main():
    # Example input text
    input_text = "私は学校に行く。"
    output_text = "私は学校に行きます。"
    
    print("=== Global Prompt Constants Demo ===\n")
    
    # 1. Using individual constants
    print("1. Individual Constants:")
    print(f"Japanese System Prompt: {SYSTEM_PROMPT}")
    print(f"English System Prompt: {SYSTEM_PROMPT_EN}")
    print(f"Japanese User Template: {USER_PROMPT_TEMPLATE}")
    print(f"English User Template: {USER_PROMPT_TEMPLATE_EN}")
    print()
    
    # 2. Using helper functions
    print("2. Helper Functions:")
    
    # Messages format (for training data)
    messages_jp = create_messages_format(input_text, output_text, use_english=False)
    messages_en = create_messages_format(input_text, output_text, use_english=True)
    
    print("Japanese Messages Format:")
    print(messages_jp)
    print("\nEnglish Messages Format:")
    print(messages_en)
    print()
    
    # Chat format (for inference)
    chat_jp = create_chat_prompt(input_text, use_english=False)
    chat_en = create_chat_prompt(input_text, use_english=True)
    
    print("Japanese Chat Format:")
    print(chat_jp)
    print("\nEnglish Chat Format:")
    print(chat_en)
    print()
    
    # 3. Using PromptConfig class
    print("3. PromptConfig Class:")
    
    # Japanese configuration
    config_jp = PromptConfig(use_english=False)
    print(f"Japanese Config - System: {config_jp.system_prompt}")
    print(f"Japanese Config - User: {config_jp.format_user_prompt(input_text)}")
    
    # English configuration
    config_en = PromptConfig(use_english=True)
    print(f"English Config - System: {config_en.system_prompt}")
    print(f"English Config - User: {config_en.format_user_prompt(input_text)}")
    print()
    
    # 4. Creating training data
    print("4. Creating Training Data:")
    training_data_jp = config_jp.create_messages(input_text, output_text)
    training_data_en = config_en.create_messages(input_text, output_text)
    
    print("Japanese Training Data:")
    print(training_data_jp)
    print("\nEnglish Training Data:")
    print(training_data_en)
    print()
    
    # 5. Creating inference prompts
    print("5. Creating Inference Prompts:")
    inference_prompt_jp = config_jp.create_chat_prompt(input_text)
    inference_prompt_en = config_en.create_chat_prompt(input_text)
    
    print("Japanese Inference Prompt:")
    print(inference_prompt_jp)
    print("\nEnglish Inference Prompt:")
    print(inference_prompt_en)

if __name__ == "__main__":
    main()