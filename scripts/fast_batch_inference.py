#!/usr/bin/env python3
"""
Fast Batch Inference for Japanese Grammar Correction

This script runs batch inference with the correct prompt format matching training data.
"""

import json
import sys
import time
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from mlx_lm import load, generate


def load_test_data(file_path: str, max_examples: int = 10):
    """Load test data from JSONL file."""
    examples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_examples:
                break
            data = json.loads(line.strip())
            messages = data['messages']
            
            # Extract input and expected output
            user_msg = None
            assistant_msg = None
            for msg in messages:
                if msg['role'] == 'user':
                    user_msg = msg['content']
                elif msg['role'] == 'assistant':
                    assistant_msg = msg['content']
            
            if user_msg and assistant_msg:
                # Extract Japanese sentence from user message
                if ': ' in user_msg:
                    input_text = user_msg.split(': ', 1)[1]
                    examples.append({
                        'input': input_text,
                        'expected': assistant_msg,
                        'original_user_msg': user_msg
                    })
    
    return examples


def generate_correction(model, tokenizer, input_text: str, use_english: bool = False):
    """Generate correction using the same format as training data."""
    from config.prompts import create_chat_prompt
    
    # Create the prompt using global constants
    prompt = create_chat_prompt(input_text, use_english)
    
    try:
        response = generate(
            model, 
            tokenizer, 
            prompt=prompt,
            max_tokens=100,
            verbose=False
        )
        
        # Extract the generated text (remove the prompt)
        if response.startswith(prompt):
            correction = response[len(prompt):].strip()
        else:
            correction = response.strip()
        
        # Remove <think>...</think> tags for think models
        import re
        correction = re.sub(r'<think>.*?</think>', '', correction, flags=re.DOTALL).strip()
        
        # Remove any end tokens
        if '<|im_end|>' in correction:
            correction = correction.split('<|im_end|>')[0].strip()
        
        return correction
            
    except Exception as e:
        print(f"Error generating correction: {e}")
        return input_text  # Return original if generation fails


def main():
    print("🔄 Loading model and adapters...")
    
    # Load model and adapters
    model, tokenizer = load(
        "mlx-community/Qwen3-0.6B-4bit",
        adapter_path="models/japanese-gec-lora-simple"
    )
    
    print("✅ Model loaded successfully!")
    
    # Load test examples
    print("📖 Loading test examples...")
    examples = load_test_data("datasets/test.jsonl", max_examples=50)
    print(f"Loaded {len(examples)} test examples")
    
    print("\n" + "="*80)
    print("🧪 TESTING JAPANESE GRAMMAR CORRECTION")
    print("="*80)
    
    correct_predictions = 0
    total_examples = len(examples)
    
    for i, example in enumerate(examples, 1):
        input_text = example['input']
        expected = example['expected']
        
        print(f"\n📝 Example {i}/{total_examples}")
        print(f"Input:    {input_text}")
        print(f"Expected: {expected}")
        
        # Generate correction
        start_time = time.time()
        predicted = generate_correction(model, tokenizer, input_text)
        generation_time = time.time() - start_time
        
        print(f"Predicted: {predicted}")
        print(f"Time: {generation_time:.2f}s")
        
        # Check if prediction matches expected
        is_correct = predicted.strip() == expected.strip()
        if is_correct:
            correct_predictions += 1
            print("✅ CORRECT")
        else:
            print("❌ INCORRECT")
        
        print("-" * 80)
    
    # Final results
    accuracy = correct_predictions / total_examples if total_examples > 0 else 0
    print(f"\n🎯 FINAL RESULTS:")
    print(f"Correct predictions: {correct_predictions}/{total_examples}")
    print(f"Accuracy: {accuracy:.2%}")
    
    if accuracy > 0.5:
        print("🎉 Model is performing well!")
    else:
        print("⚠️  Model needs improvement")


if __name__ == "__main__":
    main()