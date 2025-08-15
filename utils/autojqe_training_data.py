"""
Training data preparation for AutoJQE quality estimation dataset.

This module converts the AutoJQE corpus into various training formats
for different machine learning tasks related to GEC quality estimation.
"""

import json
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from .autojqe_parser import parse_all_autojqe_datasets, QEPair, filter_by_quality


@dataclass
class QETrainingExample:
    """Represents a training example for quality estimation tasks."""
    input_text: str
    target: str  # Can be score, label, or text
    task_type: str
    metadata: Dict


class AutoJQETrainingDataGenerator:
    """Generate training data for various QE-related tasks."""
    
    def __init__(self, datasets_dir: str):
        self.datasets_dir = datasets_dir
        self.datasets = parse_all_autojqe_datasets(datasets_dir)
        self.all_pairs = []
        for pairs in self.datasets.values():
            self.all_pairs.extend(pairs)
    
    def generate_quality_scoring_data(self) -> List[QETrainingExample]:
        """
        Generate data for training quality scoring models.
        Task: Given source + correction, predict quality score (1-4).
        """
        examples = []
        
        for pair in self.all_pairs:
            input_text = f"原文: {pair.source_text}\n修正: {pair.corrected_text}"
            target = str(pair.average_score)
            
            examples.append(QETrainingExample(
                input_text=input_text,
                target=target,
                task_type="quality_scoring",
                metadata={
                    "individual_scores": pair.individual_scores,
                    "quality_level": pair.quality_level
                }
            ))
        
        return examples
    
    def generate_quality_classification_data(self) -> List[QETrainingExample]:
        """
        Generate data for training quality classification models.
        Task: Classify correction quality as excellent/good/fair/poor.
        """
        examples = []
        
        for pair in self.all_pairs:
            input_text = f"原文: {pair.source_text}\n修正: {pair.corrected_text}"
            target = pair.quality_level
            
            examples.append(QETrainingExample(
                input_text=input_text,
                target=target,
                task_type="quality_classification",
                metadata={
                    "score": pair.average_score,
                    "is_improved": pair.is_improved
                }
            ))
        
        return examples
    
    def generate_improvement_detection_data(self) -> List[QETrainingExample]:
        """
        Generate data for training improvement detection models.
        Task: Determine if correction improved the original text.
        """
        examples = []
        
        for pair in self.all_pairs:
            input_text = f"原文: {pair.source_text}\n修正: {pair.corrected_text}"
            target = "改善" if pair.is_improved else "未改善"
            
            examples.append(QETrainingExample(
                input_text=input_text,
                target=target,
                task_type="improvement_detection",
                metadata={
                    "score": pair.average_score,
                    "texts_identical": pair.source_text == pair.corrected_text
                }
            ))
        
        return examples
    
    def generate_correction_ranking_data(self) -> List[QETrainingExample]:
        """
        Generate data for training correction ranking models.
        Task: Given multiple corrections, rank them by quality.
        """
        examples = []
        
        # Group by source text to find multiple corrections
        source_groups = {}
        for pair in self.all_pairs:
            if pair.source_text not in source_groups:
                source_groups[pair.source_text] = []
            source_groups[pair.source_text].append(pair)
        
        # Create ranking examples for sources with multiple corrections
        for source_text, corrections in source_groups.items():
            if len(corrections) < 2:
                continue
            
            # Sort by quality score
            sorted_corrections = sorted(corrections, key=lambda x: x.average_score, reverse=True)
            
            # Create pairwise ranking examples
            for i in range(len(sorted_corrections)):
                for j in range(i + 1, len(sorted_corrections)):
                    better = sorted_corrections[i]
                    worse = sorted_corrections[j]
                    
                    input_text = f"原文: {source_text}\n修正A: {better.corrected_text}\n修正B: {worse.corrected_text}"
                    target = "A"  # A is better
                    
                    examples.append(QETrainingExample(
                        input_text=input_text,
                        target=target,
                        task_type="correction_ranking",
                        metadata={
                            "score_a": better.average_score,
                            "score_b": worse.average_score,
                            "score_diff": better.average_score - worse.average_score
                        }
                    ))
        
        return examples
    
    def generate_error_severity_data(self) -> List[QETrainingExample]:
        """
        Generate data for training error severity assessment.
        Task: Assess how severe the errors in original text are.
        """
        examples = []
        
        for pair in self.all_pairs:
            # Infer error severity from correction quality and text differences
            if pair.source_text == pair.corrected_text:
                severity = "無錯誤" if pair.average_score >= 3.5 else "輕微錯誤"
            else:
                if pair.average_score >= 3.0:
                    severity = "輕微錯誤"
                elif pair.average_score >= 2.0:
                    severity = "中等錯誤"
                else:
                    severity = "嚴重錯誤"
            
            input_text = f"請評估此句子的錯誤嚴重程度: {pair.source_text}"
            target = severity
            
            examples.append(QETrainingExample(
                input_text=input_text,
                target=target,
                task_type="error_severity",
                metadata={
                    "corrected_text": pair.corrected_text,
                    "quality_score": pair.average_score,
                    "has_changes": pair.source_text != pair.corrected_text
                }
            ))
        
        return examples
    
    def generate_confidence_estimation_data(self) -> List[QETrainingExample]:
        """
        Generate data for training confidence estimation models.
        Task: Estimate confidence in correction quality based on annotator agreement.
        """
        examples = []
        
        for pair in self.all_pairs:
            # Calculate annotator agreement (lower std = higher confidence)
            import statistics
            std_dev = statistics.stdev(pair.individual_scores) if len(pair.individual_scores) > 1 else 0
            
            # Convert to confidence level
            if std_dev <= 0.5:
                confidence = "高信心"
            elif std_dev <= 1.0:
                confidence = "中信心"
            else:
                confidence = "低信心"
            
            input_text = f"原文: {pair.source_text}\n修正: {pair.corrected_text}\n評分: {pair.individual_scores}"
            target = confidence
            
            examples.append(QETrainingExample(
                input_text=input_text,
                target=target,
                task_type="confidence_estimation",
                metadata={
                    "std_dev": std_dev,
                    "individual_scores": pair.individual_scores,
                    "average_score": pair.average_score
                }
            ))
        
        return examples
    
    def generate_feedback_generation_data(self) -> List[QETrainingExample]:
        """
        Generate data for training feedback generation models.
        Task: Generate explanatory feedback about correction quality.
        """
        examples = []
        
        for pair in self.all_pairs:
            input_text = f"原文: {pair.source_text}\n修正: {pair.corrected_text}\n評分: {pair.average_score}"
            
            # Generate feedback based on quality and changes
            if pair.source_text == pair.corrected_text:
                if pair.average_score >= 3.5:
                    feedback = "原文已經很好，不需要修正。"
                else:
                    feedback = "雖然沒有修正，但原文仍有改善空間。"
            else:
                if pair.average_score >= 3.5:
                    feedback = "修正效果很好，大幅改善了原文的語法和表達。"
                elif pair.average_score >= 2.5:
                    feedback = "修正有一定效果，但仍有進一步改善的空間。"
                else:
                    feedback = "修正效果不佳，可能引入了新的錯誤或未能解決原有問題。"
            
            examples.append(QETrainingExample(
                input_text=input_text,
                target=feedback,
                task_type="feedback_generation",
                metadata={
                    "quality_level": pair.quality_level,
                    "is_improved": pair.is_improved
                }
            ))
        
        return examples
    
    def export_to_jsonl(self, examples: List[QETrainingExample], output_path: str):
        """Export training examples to JSONL format."""
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                data = {
                    "input": example.input_text,
                    "output": example.target,
                    "task_type": example.task_type,
                    "metadata": example.metadata
                }
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    def export_to_chat_format(self, examples: List[QETrainingExample], output_path: str):
        """Export to chat format for instruction tuning."""
        task_instructions = {
            "quality_scoring": "你是一個日文語法修正品質評估專家。請對給定的修正結果評分（1-4分，4分最好）。",
            "quality_classification": "你是一個日文語法修正品質分類專家。請將修正品質分類為：excellent、good、fair、poor。",
            "improvement_detection": "你是一個日文語法修正改善檢測專家。請判斷修正是否改善了原文。",
            "correction_ranking": "你是一個日文語法修正排序專家。請選擇品質更好的修正版本。",
            "error_severity": "你是一個日文語法錯誤嚴重程度評估專家。請評估句子的錯誤嚴重程度。",
            "confidence_estimation": "你是一個修正品質信心度評估專家。請評估對修正品質判斷的信心程度。",
            "feedback_generation": "你是一個日文語法修正回饋生成專家。請對修正結果提供詳細回饋。"
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                instruction = task_instructions.get(example.task_type, "請完成以下任務。")
                
                chat_data = {
                    "messages": [
                        {"role": "system", "content": instruction},
                        {"role": "user", "content": example.input_text},
                        {"role": "assistant", "content": example.target}
                    ],
                    "task_type": example.task_type,
                    "metadata": example.metadata
                }
                f.write(json.dumps(chat_data, ensure_ascii=False) + '\n')


def create_autojqe_training_datasets(datasets_dir: str, output_dir: str = "autojqe_training_data"):
    """
    Create various training datasets from AutoJQE corpus.
    
    Args:
        datasets_dir: Directory containing AutoJQE CSV files
        output_dir: Directory to save training data files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    generator = AutoJQETrainingDataGenerator(datasets_dir)
    
    # Generate different types of training data
    task_generators = {
        "quality_scoring": generator.generate_quality_scoring_data,
        "quality_classification": generator.generate_quality_classification_data,
        "improvement_detection": generator.generate_improvement_detection_data,
        "correction_ranking": generator.generate_correction_ranking_data,
        "error_severity": generator.generate_error_severity_data,
        "confidence_estimation": generator.generate_confidence_estimation_data,
        "feedback_generation": generator.generate_feedback_generation_data
    }
    
    for task_name, generator_func in task_generators.items():
        examples = generator_func()
        
        if not examples:
            print(f"No examples generated for {task_name}")
            continue
        
        # Export to different formats
        jsonl_path = os.path.join(output_dir, f"{task_name}.jsonl")
        chat_path = os.path.join(output_dir, f"{task_name}_chat.jsonl")
        
        generator.export_to_jsonl(examples, jsonl_path)
        generator.export_to_chat_format(examples, chat_path)
        
        print(f"Generated {len(examples)} examples for {task_name}")
        print(f"  - JSONL: {jsonl_path}")
        print(f"  - Chat: {chat_path}")
    
    # Print dataset statistics
    print(f"\n=== 數據集統計 ===")
    print(f"總數據點: {len(generator.all_pairs)}")
    for dataset_name, pairs in generator.datasets.items():
        print(f"{dataset_name}: {len(pairs)} 個樣本")