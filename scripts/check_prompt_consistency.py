#!/usr/bin/env python3
"""
Check consistency between prompts.py and prompt_config.yaml

This script ensures that the prompt templates in both files are synchronized.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from config.prompts import validate_prompt_consistency

def main():
    print("🔍 Checking prompt consistency...")
    is_consistent = validate_prompt_consistency()
    
    if is_consistent:
        print("\n🎉 All prompts are consistent!")
        return 0
    else:
        print("\n❌ Prompt inconsistencies found. Please fix them.")
        return 1

if __name__ == "__main__":
    exit(main())