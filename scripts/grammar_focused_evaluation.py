#!/usr/bin/env python3
"""
Grammar-Focused Evaluation for Japanese Grammar Correction

This script provides strict evaluation focused on grammatical correctness.
Uses prompts.py and prompt_config.yaml for consistent prompt management.
"""

import json
import sys
import time
import yaml
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from mlx_lm import load, generate
from config.prompts import PromptConfig


def load_prompt_config(config_path: str = "config/prompt_config.yaml") -> Dict:
    """Load prompt configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Warning: Could not load prompt config: {e}")
        return {"default_prompt_language": "english"}


def load_test_data(file_path: str, max_examples: int = 3000, task_filter: str = None):
    """Load test data from JSONL file."""
    examples = []
    task_counts = {'FIX': 0, 'DETECT': 0, 'CORRECT': 0, 'ASSESS': 0, 'OTHER': 0}
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if len(examples) >= max_examples:
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
                # Determine task type from user message
                detected_task_type = 'OTHER'
                if user_msg.startswith('[FIX]'):
                    detected_task_type = 'FIX'
                elif user_msg.startswith('[DETECT]'):
                    detected_task_type = 'DETECT'
                elif user_msg.startswith('[CORRECT]'):
                    detected_task_type = 'CORRECT'
                elif user_msg.startswith('[ASSESS]'):
                    detected_task_type = 'ASSESS'
                
                # Filter by task type if specified
                if task_filter and task_filter != 'ALL':
                    if detected_task_type != task_filter:
                        continue
                
                # Extract Japanese sentence from user message
                # Handle task prefixes and different prompt formats
                input_text = user_msg
                
                # Remove task prefix if present
                if input_text.startswith('[') and ']' in input_text:
                    input_text = input_text.split(']', 1)[1].strip()
                
                # Extract sentence from prompt template
                if '：' in input_text:  # Japanese colon
                    input_text = input_text.split('：', 1)[1].strip()
                elif ': ' in input_text:  # English colon
                    input_text = input_text.split(': ', 1)[1].strip()
                elif 'sentence: ' in input_text:
                    input_text = input_text.split('sentence: ', 1)[1].strip()
                
                examples.append({
                    'input': input_text,
                    'expected': assistant_msg,
                    'original_user_msg': user_msg,
                    'task_type': detected_task_type
                })
                
                task_counts[detected_task_type] += 1
    
    # Print task distribution
    if not task_filter or task_filter == 'ALL':
        print(f"📊 Task Distribution:")
        for task, count in task_counts.items():
            if count > 0:
                print(f"  {task}: {count} examples")
    
    return examples


def generate_correction(model, tokenizer, input_text: str, prompt_config: PromptConfig, task_type: str = "FIX", **kwargs):
    """Generate correction using task-specific prompts from configuration."""
    
    # Create the prompt using PromptConfig with task type
    prompt = prompt_config.create_chat_prompt(input_text, task_type=task_type, **kwargs)
    
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


def analyze_fix_task(input_text: str, predicted: str, expected: str) -> Dict:
    """Analyze FIX task (end-to-end correction)."""
    made_correction = predicted != input_text
    exact_match = predicted.strip() == expected.strip()
    
    error_analysis = {
        'no_change_needed': input_text == expected,
        'model_made_change': made_correction,
        'correct_change': exact_match and made_correction,
        'incorrect_change': made_correction and not exact_match,
        'missed_correction': not made_correction and input_text != expected,
        'unnecessary_change': made_correction and input_text == expected
    }
    
    return {
        'exact_match': exact_match,
        'made_correction': made_correction,
        'error_type': error_analysis,
        'task_type': 'FIX'
    }


def analyze_detect_task(input_text: str, predicted: str, expected: str) -> Dict:
    """Analyze DETECT task (error detection with <> markers)."""
    import re
    
    # Extract marked errors from predicted and expected
    predicted_errors = re.findall(r'<([^>]*)>', predicted)
    expected_errors = re.findall(r'<([^>]*)>', expected)
    
    # Check if the base text (without markers) is the same
    predicted_base = re.sub(r'<([^>]*)>', r'\1', predicted)
    expected_base = re.sub(r'<([^>]*)>', r'\1', expected)
    base_text_match = predicted_base.strip() == expected_base.strip()
    
    # Calculate exact match precision, recall, F1
    predicted_set = set(predicted_errors)
    expected_set = set(expected_errors)
    
    exact_true_positives = len(predicted_set & expected_set)
    exact_false_positives = len(predicted_set - expected_set)
    exact_false_negatives = len(expected_set - predicted_set)
    
    exact_precision = exact_true_positives / (exact_true_positives + exact_false_positives) if (exact_true_positives + exact_false_positives) > 0 else 0
    exact_recall = exact_true_positives / (exact_true_positives + exact_false_negatives) if (exact_true_positives + exact_false_negatives) > 0 else 0
    exact_f1 = 2 * exact_precision * exact_recall / (exact_precision + exact_recall) if (exact_precision + exact_recall) > 0 else 0
    
    # Calculate inclusive match (predicted contains expected errors)
    inclusive_true_positives = 0
    inclusive_false_negatives = 0
    
    for expected_error in expected_errors:
        # Check if any predicted error contains this expected error
        found_in_predicted = any(expected_error in predicted_error for predicted_error in predicted_errors)
        if found_in_predicted:
            inclusive_true_positives += 1
        else:
            inclusive_false_negatives += 1
    
    # For inclusive false positives, count predicted errors that don't contain any expected error
    inclusive_false_positives = 0
    for predicted_error in predicted_errors:
        contains_expected = any(expected_error in predicted_error for expected_error in expected_errors)
        if not contains_expected:
            inclusive_false_positives += 1
    
    inclusive_precision = inclusive_true_positives / (inclusive_true_positives + inclusive_false_positives) if (inclusive_true_positives + inclusive_false_positives) > 0 else 0
    inclusive_recall = inclusive_true_positives / (inclusive_true_positives + inclusive_false_negatives) if (inclusive_true_positives + inclusive_false_negatives) > 0 else 0
    inclusive_f1 = 2 * inclusive_precision * inclusive_recall / (inclusive_precision + inclusive_recall) if (inclusive_precision + inclusive_recall) > 0 else 0
    
    exact_match = predicted.strip() == expected.strip()
    
    return {
        'exact_match': exact_match,
        'base_text_match': base_text_match,
        'precision': exact_precision,
        'recall': exact_recall,
        'f1': exact_f1,
        'inclusive_precision': inclusive_precision,
        'inclusive_recall': inclusive_recall,
        'inclusive_f1': inclusive_f1,
        'predicted_errors': predicted_errors,
        'expected_errors': expected_errors,
        'true_positives': exact_true_positives,
        'false_positives': exact_false_positives,
        'false_negatives': exact_false_negatives,
        'inclusive_true_positives': inclusive_true_positives,
        'inclusive_false_positives': inclusive_false_positives,
        'inclusive_false_negatives': inclusive_false_negatives,
        'task_type': 'DETECT'
    }


def analyze_correct_task(input_text: str, predicted: str, expected: str) -> Dict:
    """Analyze CORRECT task (correcting marked errors)."""
    import re
    
    # Extract the input errors (should be marked with <>)
    input_errors = re.findall(r'<([^>]*)>', input_text)
    
    # Check if the correction is exact
    exact_match = predicted.strip() == expected.strip()
    
    # Check if model attempted to correct the marked errors
    has_markers_in_prediction = '<' in predicted and '>' in predicted
    
    return {
        'exact_match': exact_match,
        'input_errors': input_errors,
        'has_markers_in_prediction': has_markers_in_prediction,
        'task_type': 'CORRECT'
    }


def analyze_assess_task(input_text: str, predicted: str, expected: str) -> Dict:
    """Analyze ASSESS task (quality assessment)."""
    import re
    
    # Extract scores from predicted and expected
    predicted_score = extract_score_from_text(predicted)
    expected_score = extract_score_from_text(expected)
    
    # Calculate score difference
    score_diff = abs(predicted_score - expected_score) if predicted_score is not None and expected_score is not None else None
    
    # Check if score is within acceptable range (±0.5)
    score_close = score_diff is not None and score_diff <= 0.5
    exact_score_match = predicted_score == expected_score
    
    return {
        'predicted_score': predicted_score,
        'expected_score': expected_score,
        'score_diff': score_diff,
        'score_close': score_close,
        'exact_score_match': exact_score_match,
        'task_type': 'ASSESS'
    }


def extract_score_from_text(text: str) -> float:
    """Extract numerical score from assessment text."""
    import re
    
    # Look for patterns like "4 (excellent)" or just "3.5"
    score_patterns = [
        r'^(\d+(?:\.\d+)?)',  # Score at the beginning
        r'(\d+(?:\.\d+)?)\s*\(',  # Score followed by parentheses
        r'Score:\s*(\d+(?:\.\d+)?)',  # "Score: X"
        r'(\d+(?:\.\d+)?)/4',  # "X/4" format
    ]
    
    for pattern in score_patterns:
        match = re.search(pattern, text.strip())
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                continue
    
    return None


def analyze_task_result(input_text: str, predicted: str, expected: str, task_type: str) -> Dict:
    """Analyze result based on task type."""
    if task_type == 'FIX':
        return analyze_fix_task(input_text, predicted, expected)
    elif task_type == 'DETECT':
        return analyze_detect_task(input_text, predicted, expected)
    elif task_type == 'CORRECT':
        return analyze_correct_task(input_text, predicted, expected)
    elif task_type == 'ASSESS':
        return analyze_assess_task(input_text, predicted, expected)
    else:
        # Fallback to FIX analysis
        return analyze_fix_task(input_text, predicted, expected)


def calculate_task_metrics(results: List[Dict], task_type: str) -> Dict:
    """Calculate metrics specific to each task type."""
    if task_type == 'FIX':
        return calculate_fix_metrics(results)
    elif task_type == 'DETECT':
        return calculate_detect_metrics(results)
    elif task_type == 'CORRECT':
        return calculate_correct_metrics(results)
    elif task_type == 'ASSESS':
        return calculate_assess_metrics(results)
    else:
        return calculate_fix_metrics(results)


def calculate_fix_metrics(results: List[Dict]) -> Dict:
    """Calculate metrics for FIX task."""
    total = len(results)
    exact_matches = sum(1 for r in results if r['analysis']['exact_match'])
    
    categories = {
        'perfect_corrections': [],
        'correct_no_change': [],
        'missed_corrections': [],
        'incorrect_corrections': [],
        'unnecessary_changes': []
    }
    
    for result in results:
        analysis = result['analysis']
        error_type = analysis.get('error_type', {})
        
        if analysis['exact_match'] and error_type.get('model_made_change'):
            categories['perfect_corrections'].append(result)
        elif error_type.get('no_change_needed') and not error_type.get('model_made_change'):
            categories['correct_no_change'].append(result)
        elif error_type.get('missed_correction'):
            categories['missed_corrections'].append(result)
        elif error_type.get('incorrect_change'):
            categories['incorrect_corrections'].append(result)
        elif error_type.get('unnecessary_change'):
            categories['unnecessary_changes'].append(result)
    
    return {
        'accuracy': exact_matches / total if total > 0 else 0,
        'exact_matches': exact_matches,
        'total': total,
        'categories': categories
    }


def calculate_detect_metrics(results: List[Dict]) -> Dict:
    """Calculate metrics for DETECT task."""
    total = len(results)
    exact_matches = sum(1 for r in results if r['analysis']['exact_match'])
    
    # Calculate average exact precision, recall, F1
    precisions = [r['analysis']['precision'] for r in results]
    recalls = [r['analysis']['recall'] for r in results]
    f1_scores = [r['analysis']['f1'] for r in results]
    
    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0
    
    # Calculate average inclusive precision, recall, F1
    inclusive_precisions = [r['analysis']['inclusive_precision'] for r in results]
    inclusive_recalls = [r['analysis']['inclusive_recall'] for r in results]
    inclusive_f1_scores = [r['analysis']['inclusive_f1'] for r in results]
    
    avg_inclusive_precision = sum(inclusive_precisions) / len(inclusive_precisions) if inclusive_precisions else 0
    avg_inclusive_recall = sum(inclusive_recalls) / len(inclusive_recalls) if inclusive_recalls else 0
    avg_inclusive_f1 = sum(inclusive_f1_scores) / len(inclusive_f1_scores) if inclusive_f1_scores else 0
    
    # Count perfect detections (F1 = 1.0) and good coverage (inclusive F1 > 0.8)
    perfect_detections = sum(1 for f1 in f1_scores if f1 == 1.0)
    good_coverage = sum(1 for f1 in inclusive_f1_scores if f1 > 0.8)
    
    return {
        'accuracy': exact_matches / total if total > 0 else 0,
        'exact_matches': exact_matches,
        'total': total,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'avg_f1': avg_f1,
        'avg_inclusive_precision': avg_inclusive_precision,
        'avg_inclusive_recall': avg_inclusive_recall,
        'avg_inclusive_f1': avg_inclusive_f1,
        'perfect_detections': perfect_detections,
        'good_coverage': good_coverage
    }


def calculate_correct_metrics(results: List[Dict]) -> Dict:
    """Calculate metrics for CORRECT task."""
    total = len(results)
    exact_matches = sum(1 for r in results if r['analysis']['exact_match'])
    
    # Count how many had markers in prediction (attempted correction)
    attempted_corrections = sum(1 for r in results if not r['analysis']['has_markers_in_prediction'])
    
    return {
        'accuracy': exact_matches / total if total > 0 else 0,
        'exact_matches': exact_matches,
        'total': total,
        'attempted_corrections': attempted_corrections,
        'correction_rate': attempted_corrections / total if total > 0 else 0
    }


def calculate_assess_metrics(results: List[Dict]) -> Dict:
    """Calculate metrics for ASSESS task."""
    total = len(results)
    
    # Filter out results where score extraction failed
    valid_results = [r for r in results if r['analysis']['predicted_score'] is not None and r['analysis']['expected_score'] is not None]
    valid_total = len(valid_results)
    
    if valid_total == 0:
        return {
            'accuracy': 0,
            'total': total,
            'valid_predictions': 0,
            'mae': 0,
            'close_predictions': 0
        }
    
    exact_score_matches = sum(1 for r in valid_results if r['analysis']['exact_score_match'])
    close_predictions = sum(1 for r in valid_results if r['analysis']['score_close'])
    
    # Calculate Mean Absolute Error
    score_diffs = [r['analysis']['score_diff'] for r in valid_results if r['analysis']['score_diff'] is not None]
    mae = sum(score_diffs) / len(score_diffs) if score_diffs else 0
    
    return {
        'accuracy': exact_score_matches / valid_total if valid_total > 0 else 0,
        'exact_matches': exact_score_matches,
        'total': total,
        'valid_predictions': valid_total,
        'mae': mae,
        'close_predictions': close_predictions,
        'close_accuracy': close_predictions / valid_total if valid_total > 0 else 0
    }


def main():
    parser = argparse.ArgumentParser(description="Grammar-focused evaluation for Japanese GEC")
    parser.add_argument("--model-path", default="models/japanese-gec-detect-v4", 
                       help="Path to the trained model adapters")
    parser.add_argument("--base-model", default="mlx-community/Qwen3-0.6B-4bit",
                       help="Base model to use")
    parser.add_argument("--test-data", default="datasets/combined/test.jsonl",
                       help="Path to test data")
    parser.add_argument("--max-examples", type=int, default=600,
                       help="Maximum number of examples to evaluate")
    parser.add_argument("--task-filter", default="ALL",
                       help="Task type to filter (FIX, DETECT, CORRECT, ASSESS, ALL)")
    parser.add_argument("--prompt-config", default="config/prompt_config.yaml",
                       help="Path to prompt configuration file")
    
    args = parser.parse_args()
    
    model_path = args.model_path
    
    print(f"🔄 Loading model: {model_path}")
    
    # Load model and adapters
    model, tokenizer = load(
        args.base_model,
        adapter_path=model_path
    )
    
    print("✅ Model loaded successfully!")
    
    # Load prompt configuration
    print("📋 Loading prompt configuration...")
    config_data = load_prompt_config(args.prompt_config)
    use_english = config_data.get("default_prompt_language", "english") == "english"
    prompt_config = PromptConfig(use_english=use_english)
    
    print(f"Using {'English' if use_english else 'Japanese'} prompts")
    
    # Load test examples
    print("📖 Loading test examples...")
    examples = load_test_data(args.test_data, max_examples=args.max_examples, task_filter=args.task_filter)
    
    if args.task_filter == 'ALL':
        print(f"Loaded {len(examples)} test examples (ALL task types)")
    else:
        print(f"Loaded {len(examples)} test examples ({args.task_filter} tasks)")
    
    print("\n" + "="*80)
    print("📝 GRAMMAR-FOCUSED EVALUATION")
    print("="*80)
    
    results = []
    exact_matches = 0
    
    for i, example in enumerate(examples, 1):
        input_text = example['input']
        expected = example['expected']
        current_task_type = example['task_type']
        
        print(f"\n📝 Example {i}/{len(examples)} [{current_task_type}]")
        print(f"Input:     {input_text}")
        print(f"Expected:  {expected}")
        
        # Generate correction using task-specific prompt
        if current_task_type == 'ASSESS':
            # For ASSESS task, we need to extract source and corrected text from the input
            # The input format should be like "Original: ... Corrected: ..."
            if 'Original:' in input_text and 'Corrected:' in input_text:
                parts = input_text.split('Corrected:', 1)
                source_text = parts[0].replace('Original:', '').strip()
                corrected_text = parts[1].strip()
                predicted = generate_correction(model, tokenizer, input_text, prompt_config, 
                                              task_type=current_task_type, 
                                              source_text=source_text, 
                                              corrected_text=corrected_text)
            else:
                predicted = generate_correction(model, tokenizer, input_text, prompt_config, 
                                              task_type=current_task_type)
        else:
            predicted = generate_correction(model, tokenizer, input_text, prompt_config, 
                                          task_type=current_task_type)
        print(f"Predicted: {predicted}")
        
        # Analyze result based on task type
        analysis = analyze_task_result(input_text, predicted, expected, current_task_type)
        
        result = {
            'input': input_text,
            'expected': expected,
            'predicted': predicted,
            'analysis': analysis,
            'task_type': current_task_type
        }
        results.append(result)
        
        # Show result based on task type
        if current_task_type == 'FIX':
            if analysis['exact_match']:
                exact_matches += 1
                print("✅ CORRECT")
            else:
                error_type = analysis.get('error_type', {})
                if error_type.get('missed_correction'):
                    print("❌ MISSED CORRECTION")
                elif error_type.get('incorrect_change'):
                    print("❌ WRONG CORRECTION")
                elif error_type.get('unnecessary_change'):
                    print("⚠️  UNNECESSARY CHANGE")
                else:
                    print("❌ INCORRECT")
        
        elif current_task_type == 'DETECT':
            if analysis['exact_match']:
                exact_matches += 1
                print("✅ PERFECT DETECTION")
            elif analysis['inclusive_f1'] > 0.8:  # High inclusive F1 means good coverage
                print(f"🟡 GOOD COVERAGE - Exact: P:{analysis['precision']:.2f} R:{analysis['recall']:.2f} F1:{analysis['f1']:.2f}")
                print(f"                  Inclusive: P:{analysis['inclusive_precision']:.2f} R:{analysis['inclusive_recall']:.2f} F1:{analysis['inclusive_f1']:.2f}")
            else:
                print(f"📊 Exact: P:{analysis['precision']:.2f} R:{analysis['recall']:.2f} F1:{analysis['f1']:.2f}")
                print(f"   Inclusive: P:{analysis['inclusive_precision']:.2f} R:{analysis['inclusive_recall']:.2f} F1:{analysis['inclusive_f1']:.2f}")
        
        elif current_task_type == 'CORRECT':
            if analysis['exact_match']:
                exact_matches += 1
                print("✅ CORRECT")
            else:
                if analysis['has_markers_in_prediction']:
                    print("❌ STILL HAS MARKERS (didn't complete correction)")
                else:
                    print("❌ INCORRECT CORRECTION")
        
        elif current_task_type == 'ASSESS':
            if analysis['exact_score_match']:
                exact_matches += 1
                print("✅ EXACT SCORE MATCH")
            elif analysis['score_close']:
                print(f"🟡 CLOSE (predicted: {analysis['predicted_score']}, expected: {analysis['expected_score']})")
            else:
                print(f"❌ WRONG SCORE (predicted: {analysis['predicted_score']}, expected: {analysis['expected_score']})")
        
        else:  # OTHER task type
            if analysis['exact_match']:
                exact_matches += 1
                print("✅ CORRECT")
            else:
                print("❌ INCORRECT")
        
        print("-" * 80)
    
    # Final analysis
    total = len(examples)
    
    if args.task_filter == 'ALL':
        # Group results by task type for multi-task analysis
        task_results = {}
        for result in results:
            task_type = result['task_type']
            if task_type not in task_results:
                task_results[task_type] = []
            task_results[task_type].append(result)
        
        print(f"\n🎯 MULTI-TASK ANALYSIS:")
        print(f"Total examples: {total}")
        print()
        
        overall_correct = 0
        for task_type, task_examples in task_results.items():
            if not task_examples:
                continue
                
            print(f"📋 {task_type} TASK ({len(task_examples)} examples):")
            metrics = calculate_task_metrics(task_examples, task_type)
            
            if task_type == 'FIX':
                print(f"  Exact matches: {metrics['exact_matches']} ({metrics['accuracy']:.1%})")
                overall_correct += metrics['exact_matches']
            elif task_type == 'DETECT':
                print(f"  Exact matches: {metrics['exact_matches']} ({metrics['accuracy']:.1%})")
                print(f"  Good coverage: {metrics['good_coverage']} ({metrics['good_coverage']/len(task_examples):.1%})")
                print(f"  Exact F1: {metrics['avg_f1']:.3f} | Inclusive F1: {metrics['avg_inclusive_f1']:.3f}")
                # Use good coverage for overall score (more lenient)
                overall_correct += metrics['good_coverage']
            elif task_type == 'CORRECT':
                print(f"  Exact matches: {metrics['exact_matches']} ({metrics['accuracy']:.1%})")
                overall_correct += metrics['exact_matches']
            elif task_type == 'ASSESS':
                if metrics['valid_predictions'] > 0:
                    print(f"  Exact matches: {metrics['exact_matches']} ({metrics['accuracy']:.1%})")
                    print(f"  Close predictions: {metrics['close_predictions']} ({metrics['close_accuracy']:.1%})")
                    overall_correct += metrics['close_predictions']  # Use close predictions for overall
                else:
                    print("  No valid predictions")
            print()
        
        overall_accuracy = overall_correct / total if total > 0 else 0
        print(f"📈 Overall Performance: {overall_correct}/{total} ({overall_accuracy:.1%})")
        
        if overall_accuracy >= 0.8:
            print(f"🎉 Excellent overall performance!")
        elif overall_accuracy >= 0.6:
            print(f"👍 Good overall performance")
        elif overall_accuracy >= 0.4:
            print(f"⚠️  Moderate overall performance - needs improvement")
        else:
            print(f"❌ Poor overall performance - significant improvement needed")
            
    else:
        # Single task analysis
        metrics = calculate_task_metrics(results, args.task_filter)
        print(f"\n🎯 {args.task_filter} TASK ANALYSIS:")
        print(f"Total examples: {total}")
    
        if args.task_filter == 'FIX':
            print(f"Exact matches: {metrics['exact_matches']} ({metrics['accuracy']:.1%})")
            print()
            print("📊 Error Breakdown:")
            categories = metrics['categories']
            print(f"✅ Perfect corrections: {len(categories['perfect_corrections'])} ({len(categories['perfect_corrections'])/total:.1%})")
            print(f"✅ Correct no-change: {len(categories['correct_no_change'])} ({len(categories['correct_no_change'])/total:.1%})")
            print(f"❌ Missed corrections: {len(categories['missed_corrections'])} ({len(categories['missed_corrections'])/total:.1%})")
            print(f"❌ Incorrect corrections: {len(categories['incorrect_corrections'])} ({len(categories['incorrect_corrections'])/total:.1%})")
            print(f"⚠️  Unnecessary changes: {len(categories['unnecessary_changes'])} ({len(categories['unnecessary_changes'])/total:.1%})")
            
            # Show examples
            if categories['missed_corrections']:
                print(f"\n❌ Missed Corrections (first 3):")
                for i, result in enumerate(categories['missed_corrections'][:3], 1):
                    print(f"  {i}. '{result['input']}' → should be '{result['expected']}' but got '{result['predicted']}'")
        
        elif args.task_filter == 'DETECT':
            print(f"Exact matches: {metrics['exact_matches']} ({metrics['accuracy']:.1%})")
            print(f"Perfect detections: {metrics['perfect_detections']} ({metrics['perfect_detections']/total:.1%})")
            print(f"Good coverage (inclusive F1>0.8): {metrics['good_coverage']} ({metrics['good_coverage']/total:.1%})")
            print()
            print("📊 Exact Detection Metrics:")
            print(f"Average Precision: {metrics['avg_precision']:.3f}")
            print(f"Average Recall: {metrics['avg_recall']:.3f}")
            print(f"Average F1-Score: {metrics['avg_f1']:.3f}")
            print()
            print("📊 Inclusive Detection Metrics (allows broader error spans):")
            print(f"Average Precision: {metrics['avg_inclusive_precision']:.3f}")
            print(f"Average Recall: {metrics['avg_inclusive_recall']:.3f}")
            print(f"Average F1-Score: {metrics['avg_inclusive_f1']:.3f}")
        
        elif args.task_filter == 'CORRECT':
            print(f"Exact matches: {metrics['exact_matches']} ({metrics['accuracy']:.1%})")
            print(f"Attempted corrections: {metrics['attempted_corrections']} ({metrics['correction_rate']:.1%})")
            print()
            print("📊 Correction Analysis:")
            print(f"Successfully removed markers: {metrics['attempted_corrections']}/{total}")
            print(f"Correct final output: {metrics['exact_matches']}/{total}")
        
        elif args.task_filter == 'ASSESS':
            print(f"Valid predictions: {metrics['valid_predictions']}/{total}")
            if metrics['valid_predictions'] > 0:
                print(f"Exact score matches: {metrics['exact_matches']} ({metrics['accuracy']:.1%})")
                print(f"Close predictions (±0.5): {metrics['close_predictions']} ({metrics['close_accuracy']:.1%})")
                print()
                print("📊 Assessment Metrics:")
                print(f"Mean Absolute Error: {metrics['mae']:.3f}")
            else:
                print("❌ No valid score predictions found")
    
        # Overall assessment for single task
        primary_metric = metrics['accuracy']
        if args.task_filter == 'DETECT':
            # Use inclusive F1 for DETECT as it's more practical
            primary_metric = metrics.get('avg_inclusive_f1', metrics['avg_f1'])
        elif args.task_filter == 'ASSESS' and metrics['valid_predictions'] > 0:
            primary_metric = metrics['close_accuracy']  # Use close accuracy for assessment
        
        print(f"\n📈 Overall Performance:")
        if primary_metric >= 0.8:
            print(f"🎉 Excellent performance! ({primary_metric:.1%})")
        elif primary_metric >= 0.6:
            print(f"👍 Good performance ({primary_metric:.1%})")
        elif primary_metric >= 0.4:
            print(f"⚠️  Moderate performance ({primary_metric:.1%}) - needs improvement")
        else:
            print(f"❌ Poor performance ({primary_metric:.1%}) - significant improvement needed")


if __name__ == "__main__":
    main()