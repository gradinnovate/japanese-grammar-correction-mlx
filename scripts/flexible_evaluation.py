#!/usr/bin/env python3
"""
Flexible Evaluation for Japanese Grammar Correction

This script provides multiple evaluation metrics beyond exact match.
"""

import json
import sys
import time
from pathlib import Path
import difflib
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


def calculate_similarity_score(predicted: str, expected: str) -> float:
    """Calculate similarity score using difflib."""
    return difflib.SequenceMatcher(None, predicted, expected).ratio()


def evaluate_partial_match(predicted: str, expected: str) -> bool:
    """Check if prediction contains key corrections from expected."""
    # Simple heuristic: if predicted shares significant content with expected
    similarity = calculate_similarity_score(predicted, expected)
    return similarity > 0.7  # 70% similarity threshold


def analyze_correction_quality(input_text: str, predicted: str, expected: str) -> Dict:
    """Analyze the quality of correction."""
    analysis = {
        'exact_match': predicted.strip() == expected.strip(),
        'similarity_score': calculate_similarity_score(predicted, expected),
        'partial_match': evaluate_partial_match(predicted, expected),
        'length_diff': abs(len(predicted) - len(expected)),
        'changed_from_input': predicted != input_text
    }
    return analysis


def main():
    model_path = "models/japanese-gec-lora-simple"  # Change this to test different models
    
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
    print("🧪 FLEXIBLE EVALUATION - JAPANESE GRAMMAR CORRECTION")
    print("="*80)
    
    # Metrics tracking
    exact_matches = 0
    partial_matches = 0
    similarity_scores = []
    corrections_made = 0
    
    detailed_results = []
    
    for i, example in enumerate(examples, 1):
        input_text = example['input']
        expected = example['expected']
        
        print(f"\n📝 Example {i}/{len(examples)}")
        print(f"Input:    {input_text}")
        print(f"Expected: {expected}")
        
        # Generate correction
        start_time = time.time()
        predicted = generate_correction(model, tokenizer, input_text)
        generation_time = time.time() - start_time
        
        print(f"Predicted: {predicted}")
        print(f"Time: {generation_time:.2f}s")
        
        # Analyze correction quality
        analysis = analyze_correction_quality(input_text, predicted, expected)
        detailed_results.append({
            'input': input_text,
            'expected': expected,
            'predicted': predicted,
            'analysis': analysis
        })
        
        # Update metrics
        if analysis['exact_match']:
            exact_matches += 1
            print("✅ EXACT MATCH")
        elif analysis['partial_match']:
            partial_matches += 1
            print(f"🔶 PARTIAL MATCH (similarity: {analysis['similarity_score']:.2%})")
        else:
            print(f"❌ NO MATCH (similarity: {analysis['similarity_score']:.2%})")
        
        if analysis['changed_from_input']:
            corrections_made += 1
        
        similarity_scores.append(analysis['similarity_score'])
        print("-" * 80)
    
    # Final results
    total_examples = len(examples)
    exact_accuracy = exact_matches / total_examples
    partial_accuracy = (exact_matches + partial_matches) / total_examples
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    correction_rate = corrections_made / total_examples
    
    print(f"\n🎯 EVALUATION RESULTS:")
    print(f"Model: {model_path}")
    print(f"Total examples: {total_examples}")
    print(f"Exact matches: {exact_matches} ({exact_accuracy:.2%})")
    print(f"Partial matches: {partial_matches}")
    print(f"Combined accuracy: {exact_matches + partial_matches} ({partial_accuracy:.2%})")
    print(f"Average similarity: {avg_similarity:.2%}")
    print(f"Correction rate: {corrections_made} ({correction_rate:.2%})")
    
    # Performance assessment
    if exact_accuracy > 0.4:
        print("🎉 Model shows good exact matching performance!")
    elif partial_accuracy > 0.6:
        print("🔶 Model shows reasonable partial matching performance")
    else:
        print("⚠️  Model needs significant improvement")
    
    # Save detailed results
    results_file = f"evaluation_results_{model_path.replace('/', '_')}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'model_path': model_path,
            'metrics': {
                'exact_accuracy': exact_accuracy,
                'partial_accuracy': partial_accuracy,
                'avg_similarity': avg_similarity,
                'correction_rate': correction_rate
            },
            'detailed_results': detailed_results
        }, f, ensure_ascii=False, indent=2)
    
    print(f"📊 Detailed results saved to: {results_file}")


if __name__ == "__main__":
    main()