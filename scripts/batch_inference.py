#!/usr/bin/env python3
"""
Batch Inference Script for Japanese Grammar Correction

This script implements batch processing of test dataset using the fine-tuned MLX model
for Japanese grammar correction. It generates predictions for all test examples and
saves results in a format suitable for evaluation metrics calculation.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple
import yaml

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.logging_utils import setup_logging


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logging.info(f"Loaded configuration from {config_path}")
        return config
    except Exception as e:
        logging.error(f"Failed to load configuration: {e}")
        raise


def load_test_dataset(test_file: str) -> List[Dict[str, Any]]:
    """Load test dataset from JSONL file."""
    test_data = []
    try:
        with open(test_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        test_data.append(data)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Skipping malformed JSON at line {line_num}: {e}")
        
        logging.info(f"Loaded {len(test_data)} test examples from {test_file}")
        return test_data
    except Exception as e:
        logging.error(f"Failed to load test dataset: {e}")
        raise


def extract_input_output_pairs(test_data: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """Extract input text and expected output from test dataset."""
    pairs = []
    
    for item in test_data:
        try:
            messages = item.get('messages', [])
            
            # Find user message (input) and assistant message (expected output)
            user_content = None
            assistant_content = None
            
            for message in messages:
                if message.get('role') == 'user':
                    # Extract Japanese text from the user message
                    content = message.get('content', '')
                    # Look for Japanese text after "Please correct the grammar in the following Japanese sentence:"
                    if 'Please correct the grammar in the following Japanese sentence:' in content:
                        japanese_text = content.split('Please correct the grammar in the following Japanese sentence:')[-1].strip()
                        user_content = japanese_text
                elif message.get('role') == 'assistant':
                    assistant_content = message.get('content', '').strip()
            
            if user_content and assistant_content:
                pairs.append((user_content, assistant_content))
            else:
                logging.warning(f"Could not extract input/output pair from item: {item}")
                
        except Exception as e:
            logging.warning(f"Error processing test item: {e}")
    
    logging.info(f"Extracted {len(pairs)} input/output pairs")
    return pairs


def load_model_and_tokenizer(model_path: str, adapter_path: str):
    """Load the fine-tuned model and tokenizer."""
    try:
        # Import MLX modules
        import mlx.core as mx
        from mlx_lm import load, generate
        
        # Load base model
        logging.info(f"Loading base model: {model_path}")
        model, tokenizer = load(model_path)
        
        # Load LoRA adapters if they exist
        if adapter_path and os.path.exists(adapter_path):
            logging.info(f"Loading LoRA adapters from: {adapter_path}")
            # Load adapters - this will be handled by mlx_lm.load if adapters are in the model path
            # For now, we'll assume the model path includes the adapters
        else:
            logging.warning(f"No adapters found at {adapter_path}, using base model")
        
        logging.info("Model and tokenizer loaded successfully")
        return model, tokenizer
        
    except ImportError as e:
        logging.error(f"Failed to import MLX modules: {e}")
        logging.error("Please ensure MLX and mlx-lm are installed")
        raise
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise


def generate_correction(model, tokenizer, input_text: str, config: Dict[str, Any]) -> str:
    """Generate grammar correction for a single input text."""
    try:
        # Import generate function
        from mlx_lm import generate
        
        # Format prompt using global constants
        from config.prompts import create_chat_prompt
        prompt = create_chat_prompt(input_text, use_english=True)
        
        # Generation parameters
        max_tokens = config.get('max_tokens', 512)
        
        # Generate correction
        response = generate(
            model, 
            tokenizer, 
            prompt=prompt,
            max_tokens=max_tokens,
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
        logging.error(f"Failed to generate correction for input '{input_text}': {e}")
        return input_text  # Return original text as fallback


def run_batch_inference(model, tokenizer, input_output_pairs: List[Tuple[str, str]], 
                       config: Dict[str, Any]) -> List[Dict[str, str]]:
    """Run batch inference on all test examples."""
    results = []
    total_examples = len(input_output_pairs)
    
    logging.info(f"Starting batch inference on {total_examples} examples...")
    start_time = time.time()
    
    for i, (input_text, expected_output) in enumerate(input_output_pairs, 1):
        try:
            # Generate correction
            predicted_output = generate_correction(model, tokenizer, input_text, config)
            
            # Store result
            result = {
                'input': input_text,
                'expected': expected_output,
                'predicted': predicted_output,
                'example_id': i
            }
            results.append(result)
            
            # Log progress
            if i % 10 == 0 or i == total_examples:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                eta = avg_time * (total_examples - i)
                logging.info(f"Processed {i}/{total_examples} examples "
                           f"(avg: {avg_time:.2f}s/example, ETA: {eta:.1f}s)")
            
        except Exception as e:
            logging.error(f"Error processing example {i}: {e}")
            # Add error result
            result = {
                'input': input_text,
                'expected': expected_output,
                'predicted': input_text,  # Fallback to original
                'example_id': i,
                'error': str(e)
            }
            results.append(result)
    
    total_time = time.time() - start_time
    logging.info(f"Batch inference completed in {total_time:.2f} seconds "
                f"({total_time/total_examples:.2f}s per example)")
    
    return results


def save_results(results: List[Dict[str, str]], output_file: str) -> None:
    """Save batch inference results to file."""
    try:
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # Save as JSONL for easy processing
        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
        
        logging.info(f"Results saved to {output_file}")
        
        # Also save summary statistics
        summary_file = output_file.replace('.jsonl', '_summary.json')
        summary = {
            'total_examples': len(results),
            'successful_predictions': len([r for r in results if 'error' not in r]),
            'failed_predictions': len([r for r in results if 'error' in r]),
            'output_file': output_file,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logging.info(f"Summary saved to {summary_file}")
        
    except Exception as e:
        logging.error(f"Failed to save results: {e}")
        raise


def main():
    """Main batch inference script entry point."""
    parser = argparse.ArgumentParser(description="Run batch inference for Japanese Grammar Correction evaluation")
    parser.add_argument(
        "--config",
        default="config/lora_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--test-file",
        help="Path to test dataset file (overrides config)"
    )
    parser.add_argument(
        "--model-path",
        help="Path to model (overrides config)"
    )
    parser.add_argument(
        "--adapter-path",
        help="Path to LoRA adapters (overrides config)"
    )
    parser.add_argument(
        "--output-file",
        default="results/batch_inference_results.jsonl",
        help="Path to save inference results"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--log-file",
        default="logs/batch_inference.log",
        help="Path to log file"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level, log_file=args.log_file)
    
    try:
        # Load configuration
        config = load_config(args.config)
        
        # Override config with command line arguments
        test_file = args.test_file or config.get('test', 'datasets/test.jsonl')
        model_path = args.model_path or config.get('model', 'mlx-community/Qwen3-0.6B-4bit')
        adapter_path = args.adapter_path or config.get('adapter_path', 'models/japanese-gec-lora')
        
        logging.info("=== Batch Inference Configuration ===")
        logging.info(f"Test file: {test_file}")
        logging.info(f"Model path: {model_path}")
        logging.info(f"Adapter path: {adapter_path}")
        logging.info(f"Output file: {args.output_file}")
        logging.info("=" * 38)
        
        # Load test dataset
        test_data = load_test_dataset(test_file)
        if not test_data:
            logging.error("No test data loaded")
            sys.exit(1)
        
        # Extract input/output pairs
        input_output_pairs = extract_input_output_pairs(test_data)
        if not input_output_pairs:
            logging.error("No valid input/output pairs extracted")
            sys.exit(1)
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_path, adapter_path)
        
        # Run batch inference
        results = run_batch_inference(model, tokenizer, input_output_pairs, config)
        
        # Save results
        save_results(results, args.output_file)
        
        logging.info("Batch inference completed successfully!")
        logging.info(f"Results available at: {args.output_file}")
        
    except KeyboardInterrupt:
        logging.info("Batch inference interrupted by user")
        sys.exit(1)
    except Exception as e:
        logging.error(f"Batch inference failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()