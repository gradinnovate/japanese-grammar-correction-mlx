"""
Unified dataset generator for Japanese GEC multi-task training.

This module generates training datasets in MLX LoRA format using prompts
from prompt_config.yaml and data from GEC and AutoJQE corpora.
"""

import json
import random
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass

from .gec_parser import parse_gec_corpus, GECPair
from .autojqe_parser import parse_all_autojqe_datasets, QEPair


@dataclass
class TrainingMessage:
    """Represents a training message in MLX format."""
    messages: List[Dict[str, str]]
    task_type: str
    metadata: Dict[str, Any]


class DatasetGenerator:
    """Generate multi-task training datasets for Japanese GEC."""
    
    def __init__(self, config_path: str = None):
        """Initialize with prompt configuration."""
        # Use the unified prompts API instead of loading YAML directly
        from config.prompts import PromptConfig
        self.prompt_config = PromptConfig(use_english=True)
        
        # Task mapping for dataset generation
        self.task_mapping = {
            "gec_error_detection": "DETECT",
            "gec_error_correction": "CORRECT", 
            "gec_end_to_end": "FIX",
            "quality_assessment": "ASSESS"
        }
        

        
    def _get_task_prompts(self, task_key: str) -> Dict[str, str]:
        """Get system prompt and user template for a task."""
        task_type = self.task_mapping.get(task_key, "FIX")
        return {
            "system_prompt": self.prompt_config.get_system_prompt(task_type),
            "user_template": self.prompt_config.get_user_template(task_type)
        }

    
    def _create_message(self, system_prompt: str, user_content: str, assistant_content: str, 
                       task_type: str, metadata: Dict[str, Any] = None) -> TrainingMessage:
        """Create a training message in MLX format."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
        
        return TrainingMessage(
            messages=messages,
            task_type=task_type,
            metadata=metadata or {}
        )
    
    def generate_gec_error_detection_data(self, gec_pairs: List[GECPair]) -> List[TrainingMessage]:
        """Generate error detection training data from GEC corpus."""
        prompts = self._get_task_prompts("gec_error_detection")
        system_prompt = prompts["system_prompt"]
        user_template = prompts["user_template"]
        
        messages = []
        skipped_count = 0
        
        for pair in gec_pairs:
            # Get the marked error text from original line
            marked_error_text = pair.original_line.split('\t')[0]
            
            # Skip examples where the entire sentence is marked as error
            # This happens when the marked text starts with < and ends with >。
            if self._is_entire_sentence_marked(marked_error_text):
                skipped_count += 1
                continue
            
            # Skip examples with malformed markers
            if not self._has_valid_error_markers(marked_error_text):
                skipped_count += 1
                continue
            
            # Input: clean error text, Output: text with <> markers
            user_content = user_template.format(input_text=pair.error_text)
            
            message = self._create_message(
                system_prompt=system_prompt,
                user_content=user_content,
                assistant_content=marked_error_text,
                task_type="gec_error_detection",
                metadata={"source": "gec_corpus"}
            )
            messages.append(message)
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} problematic DETECT examples")
        
        return messages
    
    def _is_entire_sentence_marked(self, text: str) -> bool:
        """Check if the entire sentence is marked as error."""
        text = text.strip()
        # Check if text starts with < and ends with >。 or just >
        return (text.startswith('<') and 
                (text.endswith('>。') or text.endswith('>'))) and \
               text.count('<') == 1 and text.count('>') == 1
    
    def _has_valid_error_markers(self, text: str) -> bool:
        """Check if the text has valid error markers."""
        import re
        
        # Count opening and closing brackets
        open_count = text.count('<')
        close_count = text.count('>')
        
        # Must have equal number of opening and closing brackets
        if open_count != close_count:
            return False
        
        # Must have at least one error marker
        if open_count == 0:
            return False
        
        # Check for properly formed markers (no nested or empty markers)
        markers = re.findall(r'<([^<>]*)>', text)
        
        # All markers should have content
        for marker in markers:
            if not marker.strip():
                return False
        
        return True
    
    def generate_gec_error_correction_data(self, gec_pairs: List[GECPair]) -> List[TrainingMessage]:
        """Generate error correction training data from GEC corpus."""
        prompts = self._get_task_prompts("gec_error_correction")
        system_prompt = prompts["system_prompt"]
        user_template = prompts["user_template"]
        
        messages = []
        for pair in gec_pairs:
            # Input: text with <> markers, Output: corrected text
            marked_error_text = pair.original_line.split('\t')[0]
            user_content = user_template.format(input_text=marked_error_text)
            
            message = self._create_message(
                system_prompt=system_prompt,
                user_content=user_content,
                assistant_content=pair.correct_text,
                task_type="gec_error_correction",
                metadata={"source": "gec_corpus"}
            )
            messages.append(message)
        
        return messages
    
    def generate_gec_end_to_end_data(self, gec_pairs: List[GECPair]) -> List[TrainingMessage]:
        """Generate end-to-end correction training data from GEC corpus."""
        prompts = self._get_task_prompts("gec_end_to_end")
        system_prompt = prompts["system_prompt"]
        user_template = prompts["user_template"]
        
        messages = []
        for pair in gec_pairs:
            # Input: clean error text, Output: corrected text
            user_content = user_template.format(input_text=pair.error_text)
            
            message = self._create_message(
                system_prompt=system_prompt,
                user_content=user_content,
                assistant_content=pair.correct_text,
                task_type="gec_end_to_end",
                metadata={"source": "gec_corpus"}
            )
            messages.append(message)
        
        return messages
    
    def generate_autojqe_end_to_end_data(self, qe_pairs: List[QEPair], 
                                        min_quality: float = 3.5) -> List[TrainingMessage]:
        """Generate end-to-end correction training data from AutoJQE corpus."""
        prompts = self._get_task_prompts("gec_end_to_end")
        system_prompt = prompts["system_prompt"]
        user_template = prompts["user_template"]
        
        # Filter high-quality corrections that actually changed the text
        high_quality_pairs = [
            pair for pair in qe_pairs 
            if pair.average_score >= min_quality and pair.source_text != pair.corrected_text
        ]
        
        messages = []
        for pair in high_quality_pairs:
            # Input: source text, Output: corrected text
            user_content = user_template.format(input_text=pair.source_text)
            
            message = self._create_message(
                system_prompt=system_prompt,
                user_content=user_content,
                assistant_content=pair.corrected_text,
                task_type="gec_end_to_end",
                metadata={
                    "source": "autojqe_corpus",
                    "quality_score": pair.average_score,
                    "individual_scores": pair.individual_scores
                }
            )
            messages.append(message)
        
        return messages
    
    def generate_quality_assessment_data(self, qe_pairs: List[QEPair]) -> List[TrainingMessage]:
        """Generate quality assessment training data from AutoJQE corpus."""
        prompts = self._get_task_prompts("quality_assessment")
        system_prompt = prompts["system_prompt"]
        user_template = prompts["user_template"]
        
        messages = []
        for pair in qe_pairs:
            # Input: source + corrected text, Output: quality score with explanation
            user_content = user_template.format(
                source_text=pair.source_text,
                corrected_text=pair.corrected_text
            )
            
            # Generate response with score and brief explanation
            score = int(round(pair.average_score))
            quality_labels = {1: "poor", 2: "fair", 3: "good", 4: "excellent"}
            quality_label = quality_labels.get(score, "fair")
            
            if pair.source_text == pair.corrected_text:
                if score >= 3:
                    explanation = "The original text is already correct and needs no changes."
                else:
                    explanation = "The text remains unchanged but still contains errors."
            else:
                if score >= 3:
                    explanation = "The correction successfully improves the grammar and clarity."
                else:
                    explanation = "The correction has limited effectiveness or introduces new issues."
            
            assistant_content = f"{score} ({quality_label}): {explanation}"
            
            message = self._create_message(
                system_prompt=system_prompt,
                user_content=user_content,
                assistant_content=assistant_content,
                task_type="quality_assessment",
                metadata={
                    "source": "autojqe_corpus",
                    "quality_score": pair.average_score,
                    "individual_scores": pair.individual_scores
                }
            )
            messages.append(message)
        
        return messages
    
    def generate_all_datasets(self, gec_corpus_path: str, autojqe_datasets_dir: str) -> Dict[str, List[TrainingMessage]]:
        """Generate all training datasets."""
        print("Loading GEC corpus...")
        gec_pairs = parse_gec_corpus(gec_corpus_path)
        print(f"Loaded {len(gec_pairs)} GEC pairs")
        
        print("Loading AutoJQE datasets...")
        autojqe_datasets = parse_all_autojqe_datasets(autojqe_datasets_dir)
        all_qe_pairs = []
        for dataset_name, pairs in autojqe_datasets.items():
            all_qe_pairs.extend(pairs)
        print(f"Loaded {len(all_qe_pairs)} AutoJQE pairs")
        
        # Generate datasets for each task
        datasets = {}
        
        # GEC tasks
        print("Generating GEC error detection data...")
        datasets["gec_error_detection"] = self.generate_gec_error_detection_data(gec_pairs)
        
        print("Generating GEC error correction data...")
        datasets["gec_error_correction"] = self.generate_gec_error_correction_data(gec_pairs)
        
        print("Generating GEC end-to-end data...")
        gec_e2e_data = self.generate_gec_end_to_end_data(gec_pairs)
        autojqe_e2e_data = self.generate_autojqe_end_to_end_data(all_qe_pairs)
        datasets["gec_end_to_end"] = gec_e2e_data + autojqe_e2e_data
        
        # AutoJQE tasks
        print("Generating quality assessment data...")
        datasets["quality_assessment"] = self.generate_quality_assessment_data(all_qe_pairs)
        
        # Print statistics
        print("\n=== Dataset Statistics ===")
        total_examples = 0
        for task_name, messages in datasets.items():
            count = len(messages)
            total_examples += count
            print(f"{task_name}: {count} examples")
        print(f"Total: {total_examples} examples")
        
        return datasets
    
    def export_to_jsonl(self, datasets: Dict[str, List[TrainingMessage]], output_dir: str = "datasets"):
        """Export datasets to JSONL files in MLX format."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for task_name, messages in datasets.items():
            if not messages:
                continue
                
            # Create task-specific directory
            task_dir = os.path.join(output_dir, task_name)
            os.makedirs(task_dir, exist_ok=True)
            
            # Split data into train/valid/test (80/10/10 split)
            random.shuffle(messages)
            train_split = int(len(messages) * 0.8)
            valid_split = int(len(messages) * 0.9)
            train_messages = messages[:train_split]
            valid_messages = messages[train_split:valid_split]
            test_messages = messages[valid_split:]
            
            # Export train.jsonl
            train_path = os.path.join(task_dir, "train.jsonl")
            with open(train_path, 'w', encoding='utf-8') as f:
                for message in train_messages:
                    data = {"messages": message.messages}
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            # Export valid.jsonl
            valid_path = os.path.join(task_dir, "valid.jsonl")
            with open(valid_path, 'w', encoding='utf-8') as f:
                for message in valid_messages:
                    data = {"messages": message.messages}
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            # Export test.jsonl
            test_path = os.path.join(task_dir, "test.jsonl")
            with open(test_path, 'w', encoding='utf-8') as f:
                for message in test_messages:
                    data = {"messages": message.messages}
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
            
            print(f"Exported {task_name}: {len(train_messages)} train + {len(valid_messages)} valid + {len(test_messages)} test examples")
    
    def export_combined_dataset(self, datasets: Dict[str, List[TrainingMessage]], 
                               output_dir: str = "datasets/combined"):
        """Export all datasets combined into MLX format (train.jsonl, valid.jsonl)."""
        all_messages = []
        for task_name, messages in datasets.items():
            all_messages.extend(messages)
        
        random.shuffle(all_messages)
        
        # Split into train/valid/test (80/10/10)
        train_split = int(len(all_messages) * 0.8)
        valid_split = int(len(all_messages) * 0.9)
        train_messages = all_messages[:train_split]
        valid_messages = all_messages[train_split:valid_split]
        test_messages = all_messages[valid_split:]
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Export train.jsonl
        train_path = str(Path(output_dir) / "train.jsonl")
        with open(train_path, 'w', encoding='utf-8') as f:
            for message in train_messages:
                data = {"messages": message.messages}
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        # Export valid.jsonl
        valid_path = str(Path(output_dir) / "valid.jsonl")
        with open(valid_path, 'w', encoding='utf-8') as f:
            for message in valid_messages:
                data = {"messages": message.messages}
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        # Export test.jsonl
        test_path = str(Path(output_dir) / "test.jsonl")
        with open(test_path, 'w', encoding='utf-8') as f:
            for message in test_messages:
                data = {"messages": message.messages}
                f.write(json.dumps(data, ensure_ascii=False) + '\n')
        
        print(f"Exported combined dataset: {len(train_messages)} train + {len(valid_messages)} valid + {len(test_messages)} test examples")
        
        # Print task distribution
        task_counts = {}
        for message in all_messages:
            task_type = message.task_type
            task_counts[task_type] = task_counts.get(task_type, 0) + 1
        
        print("\nTask distribution in combined dataset:")
        for task_type, count in sorted(task_counts.items()):
            percentage = count / len(all_messages) * 100
            print(f"  {task_type}: {count} ({percentage:.1f}%)")
    
    def export_curriculum_datasets(self, datasets: Dict[str, List[TrainingMessage]], 
                                  output_dir: str = "datasets/curriculum"):
        """Export datasets organized for curriculum learning in MLX format."""
        import os
        
        # Define curriculum stages
        curriculum_stages = {
            "stage1_basic_correction": ["gec_end_to_end"],
            "stage2_error_detection": ["gec_error_detection"],
            "stage3_precise_correction": ["gec_error_correction"],
            "stage4_quality_assessment": ["quality_assessment"]
        }
        
        print("\n📚 Creating curriculum learning datasets...")
        
        for stage_name, task_names in curriculum_stages.items():
            stage_messages = []
            for task_name in task_names:
                if task_name in datasets:
                    stage_messages.extend(datasets[task_name])
            
            if stage_messages:
                random.shuffle(stage_messages)
                
                # Split into train/valid/test (80/10/10)
                train_split = int(len(stage_messages) * 0.8)
                valid_split = int(len(stage_messages) * 0.9)
                train_messages = stage_messages[:train_split]
                valid_messages = stage_messages[train_split:valid_split]
                test_messages = stage_messages[valid_split:]
                
                # Create stage directory
                stage_dir = Path(output_dir) / stage_name
                stage_dir.mkdir(parents=True, exist_ok=True)
                
                # Export train.jsonl
                train_path = str(stage_dir / "train.jsonl")
                with open(train_path, 'w', encoding='utf-8') as f:
                    for message in train_messages:
                        data = {"messages": message.messages}
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
                
                # Export valid.jsonl
                valid_path = str(stage_dir / "valid.jsonl")
                with open(valid_path, 'w', encoding='utf-8') as f:
                    for message in valid_messages:
                        data = {"messages": message.messages}
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
                
                # Export test.jsonl
                test_path = str(stage_dir / "test.jsonl")
                with open(test_path, 'w', encoding='utf-8') as f:
                    for message in test_messages:
                        data = {"messages": message.messages}
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
                
                print(f"  📄 {stage_name}: {len(train_messages)} train + {len(valid_messages)} valid + {len(test_messages)} test examples")
        
        # Create a curriculum training guide
        guide_path = str(Path(output_dir) / "curriculum_guide.md")
        with open(guide_path, 'w', encoding='utf-8') as f:
            f.write("# Curriculum Learning Guide\n\n")
            f.write("## Training Stages\n\n")
            f.write("Train the model in the following order:\n\n")
            for i, (stage_name, task_names) in enumerate(curriculum_stages.items(), 1):
                f.write(f"{i}. **{stage_name}**: {', '.join(task_names)}\n")
            f.write("\n## Usage\n\n")
            f.write("```bash\n")
            f.write("# Stage 1: Basic correction ability\n")
            f.write("mlx_lm.lora --data datasets/curriculum/stage1_basic_correction\n\n")
            f.write("# Stage 2: Error detection ability (continue from stage 1)\n")
            f.write("mlx_lm.lora --data datasets/curriculum/stage2_error_detection --resume\n\n")
            f.write("# Continue with stages 3 and 4...\n")
            f.write("```\n")
        
        print(f"  📖 curriculum_guide.md: Training guide created")


def main():
    """Main function to generate all datasets."""
    generator = DatasetGenerator()
    
    # Generate all datasets
    datasets = generator.generate_all_datasets(
        gec_corpus_path="exclude/japanese_gec_corpus/corpus_v0.txt",
        autojqe_datasets_dir="exclude/autoJQE/datasets"
    )
    
    # Export individual task datasets
    generator.export_to_jsonl(datasets)
    
    # Export combined dataset
    generator.export_combined_dataset(datasets)


if __name__ == "__main__":
    main()