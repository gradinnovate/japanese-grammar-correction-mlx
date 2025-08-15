#!/usr/bin/env python3
"""
Analyze DETECT task training data quality.
"""

import json
import re
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))


def analyze_detect_example(example):
    """Analyze a single DETECT training example."""
    messages = example['messages']
    user_msg = messages[1]['content']
    assistant_msg = messages[2]['content']
    
    # Extract input sentence
    input_text = user_msg.replace('[DETECT] Mark the grammatical errors in this Japanese sentence: ', '')
    
    # Extract marked errors
    marked_errors = re.findall(r'<([^>]*)>', assistant_msg)
    
    # Check if entire sentence is marked as error
    entire_sentence_marked = assistant_msg.startswith('<') and assistant_msg.endswith('>。')
    
    # Check if no errors are marked
    no_errors_marked = '<' not in assistant_msg
    
    # Count error spans
    error_count = len(marked_errors)
    
    return {
        'input': input_text,
        'output': assistant_msg,
        'marked_errors': marked_errors,
        'entire_sentence_marked': entire_sentence_marked,
        'no_errors_marked': no_errors_marked,
        'error_count': error_count
    }


def main():
    print("=== DETECT Task Data Quality Analysis ===\n")
    
    # Load training data
    train_file = "datasets/gec_error_detection/train.jsonl"
    examples = []
    
    with open(train_file, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} training examples")
    
    # Analyze examples
    analysis_results = []
    for example in examples:
        result = analyze_detect_example(example)
        analysis_results.append(result)
    
    # Statistics
    total = len(analysis_results)
    entire_sentence_marked = sum(1 for r in analysis_results if r['entire_sentence_marked'])
    no_errors_marked = sum(1 for r in analysis_results if r['no_errors_marked'])
    single_error = sum(1 for r in analysis_results if r['error_count'] == 1)
    multiple_errors = sum(1 for r in analysis_results if r['error_count'] > 1)
    
    print(f"\n📊 Data Quality Statistics:")
    print(f"Total examples: {total}")
    print(f"Entire sentence marked as error: {entire_sentence_marked} ({entire_sentence_marked/total:.1%})")
    print(f"No errors marked: {no_errors_marked} ({no_errors_marked/total:.1%})")
    print(f"Single error marked: {single_error} ({single_error/total:.1%})")
    print(f"Multiple errors marked: {multiple_errors} ({multiple_errors/total:.1%})")
    
    # Show problematic examples
    print(f"\n🔍 Problematic Examples (Entire Sentence Marked):")
    problematic = [r for r in analysis_results if r['entire_sentence_marked']][:5]
    for i, result in enumerate(problematic, 1):
        print(f"\n{i}. Input: {result['input']}")
        print(f"   Output: {result['output']}")
        print(f"   Issue: Entire sentence marked as error")
    
    # Show good examples
    print(f"\n✅ Good Examples (Specific Error Marking):")
    good_examples = [r for r in analysis_results if r['error_count'] == 1 and not r['entire_sentence_marked']][:5]
    for i, result in enumerate(good_examples, 1):
        print(f"\n{i}. Input: {result['input']}")
        print(f"   Output: {result['output']}")
        print(f"   Marked error: {result['marked_errors'][0]}")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    if entire_sentence_marked > total * 0.1:
        print(f"- High rate of entire sentence marking ({entire_sentence_marked/total:.1%}) may confuse the model")
        print(f"- Consider filtering out examples where entire sentence is marked")
    
    if no_errors_marked > total * 0.05:
        print(f"- Some examples have no errors marked ({no_errors_marked/total:.1%})")
        print(f"- These might be correct sentences that should be left unchanged")
    
    print(f"- Focus training on examples with specific error marking")
    print(f"- Consider data augmentation to balance error types")


if __name__ == "__main__":
    main()