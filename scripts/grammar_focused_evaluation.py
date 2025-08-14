#!/usr/bin/env python3
"""
Grammar-Focused Evaluation for Japanese Grammar Correction

This script provides strict evaluation focused on grammatical correctness.
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from mlx_lm import load, generate


def load_test_data(file_path: str, max_examples: int = 50):
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
                # Handle both old and new prompt formats
                if '：' in user_msg:  # New format with Japanese colon
                    input_text = user_msg.split('：', 1)[1]
                elif ': ' in user_msg:  # Old format with English colon
                    input_text = user_msg.split(': ', 1)[1]
                else:
                    input_text = user_msg
                
                examples.append({
                    'input': input_text,
                    'expected': assistant_msg,
                    'original_user_msg': user_msg
                })
    
    return examples


def generate_correction(model, tokenizer, input_text: str, use_english: bool = True):
    """Generate correction using the same format as training data."""
    from config.prompts import create_chat_prompt
    
    # Create the prompt using global constants
    prompt = create_chat_prompt(input_text, use_english)
    
    try:
        response = generate(
            model, 
            tokenizer, 
            prompt=prompt,
            max_tokens=400,
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


def analyze_grammar_correction(input_text: str, predicted: str, expected: str) -> Dict:
    """Analyze grammar correction with focus on correctness."""
    
    # Check if model made any changes
    made_correction = predicted != input_text
    
    # Check exact match (most important for grammar)
    exact_match = predicted.strip() == expected.strip()
    
    # Analyze the type of error
    error_analysis = {
        'no_change_needed': input_text == expected,  # Input was already correct
        'model_made_change': made_correction,
        'correct_change': exact_match and made_correction,
        'incorrect_change': made_correction and not exact_match,
        'missed_correction': not made_correction and input_text != expected,
        'unnecessary_change': made_correction and input_text == expected
    }
    
    return {
        'exact_match': exact_match,
        'made_correction': made_correction,
        'error_type': error_analysis
    }


def categorize_errors(examples_with_results: List[Dict]) -> Dict:
    """Categorize different types of errors."""
    categories = {
        'perfect_corrections': [],      # Exact match and needed correction
        'correct_no_change': [],        # No change needed, model didn't change
        'missed_corrections': [],       # Should have corrected but didn't
        'incorrect_corrections': [],    # Made wrong correction
        'unnecessary_changes': []       # Changed something that was already correct
    }
    
    for result in examples_with_results:
        analysis = result['analysis']
        error_type = analysis['error_type']
        
        if analysis['exact_match'] and error_type['model_made_change']:
            categories['perfect_corrections'].append(result)
        elif error_type['no_change_needed'] and not error_type['model_made_change']:
            categories['correct_no_change'].append(result)
        elif error_type['missed_correction']:
            categories['missed_corrections'].append(result)
        elif error_type['incorrect_change']:
            categories['incorrect_corrections'].append(result)
        elif error_type['unnecessary_change']:
            categories['unnecessary_changes'].append(result)
    
    return categories


def main():
    model_path = "models/japanese-gec-lora-conservative"
    
    print(f"🔄 Loading model: {model_path}")
    
    # Load model and adapters
    model, tokenizer = load(
        "mlx-community/Qwen3-0.6B-4bit",
        adapter_path=model_path
    )
    
    print("✅ Model loaded successfully!")
    
    # Load test examples
    print("📖 Loading test examples...")
    examples = load_test_data("datasets/test.jsonl", max_examples=50)
    print(f"Loaded {len(examples)} test examples")
    
    print("\n" + "="*80)
    print("📝 GRAMMAR-FOCUSED EVALUATION")
    print("="*80)
    
    results = []
    exact_matches = 0
    
    for i, example in enumerate(examples, 1):
        input_text = example['input']
        expected = example['expected']
        
        print(f"\n📝 Example {i}/{len(examples)}")
        print(f"Input:     {input_text}")
        print(f"Expected:  {expected}")
        
        # Generate correction
        predicted = generate_correction(model, tokenizer, input_text)
        print(f"Predicted: {predicted}")
        
        # Analyze correction
        analysis = analyze_grammar_correction(input_text, predicted, expected)
        
        result = {
            'input': input_text,
            'expected': expected,
            'predicted': predicted,
            'analysis': analysis
        }
        results.append(result)
        
        # Show result
        if analysis['exact_match']:
            exact_matches += 1
            print("✅ CORRECT")
        else:
            error_type = analysis['error_type']
            if error_type['missed_correction']:
                print("❌ MISSED CORRECTION (should have fixed but didn't)")
            elif error_type['incorrect_change']:
                print("❌ WRONG CORRECTION (made incorrect change)")
            elif error_type['unnecessary_change']:
                print("⚠️  UNNECESSARY CHANGE (changed correct sentence)")
            else:
                print("❌ INCORRECT")
        
        print("-" * 80)
    
    # Categorize errors
    categories = categorize_errors(results)
    
    # Final analysis
    total = len(examples)
    print(f"\n🎯 GRAMMAR CORRECTION ANALYSIS:")
    print(f"Total examples: {total}")
    print(f"Exact matches: {exact_matches} ({exact_matches/total:.1%})")
    print()
    print("📊 Error Breakdown:")
    print(f"✅ Perfect corrections: {len(categories['perfect_corrections'])} ({len(categories['perfect_corrections'])/total:.1%})")
    print(f"✅ Correct no-change: {len(categories['correct_no_change'])} ({len(categories['correct_no_change'])/total:.1%})")
    print(f"❌ Missed corrections: {len(categories['missed_corrections'])} ({len(categories['missed_corrections'])/total:.1%})")
    print(f"❌ Incorrect corrections: {len(categories['incorrect_corrections'])} ({len(categories['incorrect_corrections'])/total:.1%})")
    print(f"⚠️  Unnecessary changes: {len(categories['unnecessary_changes'])} ({len(categories['unnecessary_changes'])/total:.1%})")
    
    # Show some examples of each category
    print(f"\n🔍 EXAMPLE ANALYSIS:")
    
    if categories['missed_corrections']:
        print(f"\n❌ Missed Corrections (showing first 3):")
        for i, result in enumerate(categories['missed_corrections'][:3], 1):
            print(f"  {i}. '{result['input']}' → should be '{result['expected']}' but got '{result['predicted']}'")
    
    if categories['incorrect_corrections']:
        print(f"\n❌ Incorrect Corrections (showing first 3):")
        for i, result in enumerate(categories['incorrect_corrections'][:3], 1):
            print(f"  {i}. '{result['input']}' → expected '{result['expected']}' but got '{result['predicted']}'")
    
    # Overall assessment
    good_performance = exact_matches / total
    if good_performance >= 0.8:
        print(f"\n🎉 Excellent performance! ({good_performance:.1%} accuracy)")
    elif good_performance >= 0.6:
        print(f"\n👍 Good performance ({good_performance:.1%} accuracy)")
    elif good_performance >= 0.4:
        print(f"\n⚠️  Moderate performance ({good_performance:.1%} accuracy) - needs improvement")
    else:
        print(f"\n❌ Poor performance ({good_performance:.1%} accuracy) - significant improvement needed")


if __name__ == "__main__":
    main()