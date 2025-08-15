#!/usr/bin/env python3
"""
Script to generate training datasets for Japanese GEC multi-task learning.
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from utils.dataset_generator import DatasetGenerator


def main():
    """Generate all training datasets."""
    print("=== Japanese GEC Multi-Task Dataset Generator ===\n")
    
    # Check if required files exist
    gec_corpus_path = "exclude/japanese_gec_corpus/corpus_v0.txt"
    autojqe_datasets_dir = "exclude/autoJQE/datasets"
    
    missing_files = []
    if not os.path.exists(gec_corpus_path):
        missing_files.append(gec_corpus_path)
    if not os.path.exists(autojqe_datasets_dir):
        missing_files.append(autojqe_datasets_dir)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return
    
    # Initialize generator (now uses unified prompts API)
    generator = DatasetGenerator()
    
    # Generate all datasets
    print("🚀 Starting dataset generation...\n")
    datasets = generator.generate_all_datasets(gec_corpus_path, autojqe_datasets_dir)
    
    # Export datasets
    print("\n📁 Exporting individual task datasets...")
    generator.export_to_jsonl(datasets, output_dir="datasets")
    
    print("\n📦 Creating combined dataset for multi-task training...")
    generator.export_combined_dataset(datasets, "datasets/combined")
    
    print("\n📚 Creating curriculum learning datasets...")
    generator.export_curriculum_datasets(datasets, "datasets/curriculum")
    
    print("\n✅ Dataset generation completed!")
    print("\nGenerated directories (MLX LoRA format):")
    print("  📂 datasets/")
    for task_name in datasets.keys():
        if datasets[task_name]:  # Only show non-empty datasets
            print(f"     📂 {task_name}/ (train.jsonl, valid.jsonl, test.jsonl)")
    print("     📂 combined/ (train.jsonl, valid.jsonl, test.jsonl)")
    print("  📂 datasets/curriculum/")
    print("     📂 stage1_basic_correction/ (train.jsonl, valid.jsonl, test.jsonl)")
    print("     📂 stage2_error_detection/ (train.jsonl, valid.jsonl, test.jsonl)") 
    print("     📂 stage3_precise_correction/ (train.jsonl, valid.jsonl, test.jsonl)")
    print("     📂 stage4_quality_assessment/ (train.jsonl, valid.jsonl, test.jsonl)")
    print("     📖 curriculum_guide.md")
    
    print("\n💡 Training Commands:")
    print("  🎯 Single-task: mlx_lm.lora --data datasets/gec_end_to_end")
    print("  🔄 Multi-task: mlx_lm.lora --data datasets/combined")
    print("  📚 Curriculum: mlx_lm.lora --data datasets/curriculum/stage1_basic_correction")
    print("  📖 See curriculum_guide.md for detailed curriculum training steps")


if __name__ == "__main__":
    main()