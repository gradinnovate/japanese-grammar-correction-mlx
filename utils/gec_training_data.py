"""
Training data preparation for Japanese GEC using the corpus.

This module converts the Japanese GEC corpus into various training formats
for different LLM training approaches.
"""

import json
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from .gec_parser import parse_gec_corpus, GECPair


@dataclass
class TrainingExample:
    """Represents a training example for GEC."""
    input_text: str
    target_text: str
    error_spans: List[Tuple[int, int, str]]  # (start, end, correction)
    instruction: str


class GECTrainingDataGenerator:
    """Generate training data for different GEC training approaches."""
    
    def __init__(self, corpus_path: str):
        self.corpus_path = corpus_path
        self.gec_pairs = parse_gec_corpus(corpus_path)
    
    def generate_error_detection_data(self) -> List[TrainingExample]:
        """
        Generate data for training error detection (marking errors with <>).
        
        Returns:
            List of training examples where model learns to mark errors
        """
        examples = []
        
        for pair in self.gec_pairs:
            # Input: clean error text (without markers)
            # Target: error text with <> markers
            input_text = pair.error_text
            target_text = self._reconstruct_error_markers(pair.original_line.split('\t')[0])
            
            instruction = "請標記出句子中的語法錯誤部分，使用 <> 包圍錯誤的地方。"
            
            examples.append(TrainingExample(
                input_text=input_text,
                target_text=target_text,
                error_spans=self._extract_error_spans(pair.original_line.split('\t')[0]),
                instruction=instruction
            ))
        
        return examples
    
    def generate_error_correction_data(self) -> List[TrainingExample]:
        """
        Generate data for training error correction.
        
        Returns:
            List of training examples where model learns to correct errors
        """
        examples = []
        
        for pair in self.gec_pairs:
            # Input: error text with <> markers
            # Target: corrected text
            input_text = pair.original_line.split('\t')[0]
            target_text = pair.correct_text
            
            instruction = "請修正句子中 <> 標記的錯誤部分。"
            
            examples.append(TrainingExample(
                input_text=input_text,
                target_text=target_text,
                error_spans=self._extract_error_spans(input_text),
                instruction=instruction
            ))
        
        return examples
    
    def generate_end_to_end_data(self) -> List[TrainingExample]:
        """
        Generate data for end-to-end error correction (no markers in input).
        
        Returns:
            List of training examples for direct error correction
        """
        examples = []
        
        for pair in self.gec_pairs:
            # Input: clean error text
            # Target: clean correct text
            input_text = pair.error_text
            target_text = pair.correct_text
            
            instruction = "請修正句子中的語法錯誤。"
            
            examples.append(TrainingExample(
                input_text=input_text,
                target_text=target_text,
                error_spans=self._extract_error_spans(pair.original_line.split('\t')[0]),
                instruction=instruction
            ))
        
        return examples
    
    def generate_explanation_data(self) -> List[TrainingExample]:
        """
        Generate data for training error explanation.
        
        Returns:
            List of training examples with error explanations
        """
        examples = []
        
        for pair in self.gec_pairs:
            input_text = f"錯誤句子：{pair.error_text}\n正確句子：{pair.correct_text}"
            
            # Generate explanation based on error patterns
            explanation = self._generate_error_explanation(pair)
            target_text = f"語法錯誤說明：{explanation}"
            
            instruction = "請解釋句子中的語法錯誤並說明如何修正。"
            
            examples.append(TrainingExample(
                input_text=input_text,
                target_text=target_text,
                error_spans=[],
                instruction=instruction
            ))
        
        return examples
    
    def _reconstruct_error_markers(self, marked_text: str) -> str:
        """Reconstruct the original marked text format."""
        return marked_text
    
    def _extract_error_spans(self, marked_text: str) -> List[Tuple[int, int, str]]:
        """Extract error span positions and corrections."""
        spans = []
        pattern = r'<([^>]*)>'
        
        for match in re.finditer(pattern, marked_text):
            start = match.start()
            end = match.end()
            error_text = match.group(1)
            spans.append((start, end, error_text))
        
        return spans
    
    def _generate_error_explanation(self, pair: GECPair) -> str:
        """Generate simple error explanation based on patterns."""
        error_part = re.search(r'<([^>]*)>', pair.original_line.split('\t')[0])
        correct_part = re.search(r'\(([^)]*)\)', pair.original_line.split('\t')[1])
        
        if error_part and correct_part:
            error_text = error_part.group(1)
            correct_text = correct_part.group(1)
            return f"「{error_text}」應該改為「{correct_text}」"
        
        return "語法錯誤需要修正"
    
    def export_to_jsonl(self, examples: List[TrainingExample], output_path: str):
        """Export training examples to JSONL format for LLM training."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                data = {
                    "instruction": example.instruction,
                    "input": example.input_text,
                    "output": example.target_text,
                    "error_spans": example.error_spans
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    def export_to_chat_format(self, examples: List[TrainingExample], output_path: str):
        """Export to chat format for instruction tuning."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                chat_data = {
                    "messages": [
                        {"role": "system", "content": "你是一個日文語法錯誤修正助手。"},
                        {"role": "user", "content": f"{example.instruction}\n\n{example.input_text}"},
                        {"role": "assistant", "content": example.target_text}
                    ]
                }
                f.write(json.dumps(chat_data, ensure_ascii=False) + '\n')


def create_training_datasets(corpus_path: str, output_dir: str = "training_data"):
    """
    Create various training datasets from the GEC corpus.
    
    Args:
        corpus_path: Path to the GEC corpus file
        output_dir: Directory to save training data files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    generator = GECTrainingDataGenerator(corpus_path)
    
    # Generate different types of training data
    datasets = {
        "error_detection": generator.generate_error_detection_data(),
        "error_correction": generator.generate_error_correction_data(),
        "end_to_end": generator.generate_end_to_end_data(),
        "explanation": generator.generate_explanation_data()
    }
    
    # Export to different formats
    for dataset_name, examples in datasets.items():
        # JSONL format
        jsonl_path = os.path.join(output_dir, f"{dataset_name}.jsonl")
        generator.export_to_jsonl(examples, jsonl_path)
        
        # Chat format
        chat_path = os.path.join(output_dir, f"{dataset_name}_chat.jsonl")
        generator.export_to_chat_format(examples, chat_path)
        
        print(f"Generated {len(examples)} examples for {dataset_name}")
        print(f"  - JSONL: {jsonl_path}")
        print(f"  - Chat: {chat_path}")