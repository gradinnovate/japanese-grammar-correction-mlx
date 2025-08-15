"""
Global prompt templates for Japanese Grammar Correction system.

This module contains all system and user prompt templates used across
the training, evaluation, and inference scripts.

Note: This module is synchronized with config/prompt_config.yaml
"""

import yaml
from pathlib import Path

# Load prompts from YAML configuration
_yaml_config = None

def _load_yaml_config():
    """Load YAML configuration once and cache it."""
    global _yaml_config
    if _yaml_config is None:
        try:
            config_file = Path(__file__).parent / "prompt_config.yaml"
            with open(config_file, 'r', encoding='utf-8') as f:
                _yaml_config = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load prompt_config.yaml: {e}")
            _yaml_config = {}
    return _yaml_config

def _get_system_prompt(task_type: str, use_english: bool = True):
    """Get system prompt from YAML config."""
    config = _load_yaml_config()
    objectives = config.get('training_objectives', {})
    
    # Map task types to YAML keys
    task_mapping = {
        'FIX': 'gec_end_to_end',
        'DETECT': 'gec_error_detection', 
        'CORRECT': 'gec_error_correction',
        'ASSESS': 'quality_assessment'
    }
    
    yaml_key = task_mapping.get(task_type, 'gec_end_to_end')
    if yaml_key in objectives:
        return objectives[yaml_key].get('system_prompt', '')
    
    # Fallback
    return "You are a Japanese grammar correction specialist."

def _get_user_template(task_type: str, use_english: bool = True):
    """Get user template from YAML config."""
    config = _load_yaml_config()
    objectives = config.get('training_objectives', {})
    
    # Map task types to YAML keys
    task_mapping = {
        'FIX': 'gec_end_to_end',
        'DETECT': 'gec_error_detection',
        'CORRECT': 'gec_error_correction', 
        'ASSESS': 'quality_assessment'
    }
    
    yaml_key = task_mapping.get(task_type, 'gec_end_to_end')
    if yaml_key in objectives:
        template = objectives[yaml_key].get('user_template', '')
        # Add task prefix
        prefixes = config.get('multi_task_config', {}).get('task_prefixes', {})
        prefix = prefixes.get(yaml_key, f'[{task_type}]')
        if not template.startswith(prefix):
            template = f"{prefix} {template}"
        return template
    
    # Fallback
    return f"[{task_type}] Please process this Japanese sentence: {{input_text}}"

# Dynamic prompt access
class _PromptDict:
    """Dynamic dictionary that loads prompts from YAML."""
    def __init__(self, use_english=True):
        self.use_english = use_english
    
    def get(self, key, default=None):
        return _get_system_prompt(key, self.use_english) or default
    
    def __getitem__(self, key):
        return _get_system_prompt(key, self.use_english)

class _TemplateDict:
    """Dynamic dictionary that loads templates from YAML."""
    def __init__(self, use_english=True):
        self.use_english = use_english
    
    def get(self, key, default=None):
        return _get_user_template(key, self.use_english) or default
    
    def __getitem__(self, key):
        return _get_user_template(key, self.use_english)

# Dynamic prompt dictionaries
SYSTEM_PROMPTS_EN = _PromptDict(use_english=True)
SYSTEM_PROMPTS_JA = _PromptDict(use_english=False)  # Note: Currently using English prompts from YAML
USER_PROMPT_TEMPLATES_EN = _TemplateDict(use_english=True)
USER_PROMPT_TEMPLATES_JA = _TemplateDict(use_english=False)

# Legacy prompts for backward compatibility
def get_legacy_system_prompt(use_english=False):
    return _get_system_prompt("FIX", use_english)

def get_legacy_user_template(use_english=False):
    return _get_user_template("FIX", use_english)

# Legacy constants (computed at import time)
SYSTEM_PROMPT = get_legacy_system_prompt(use_english=False)
SYSTEM_PROMPT_EN = get_legacy_system_prompt(use_english=True)
USER_PROMPT_TEMPLATE = get_legacy_user_template(use_english=False)
USER_PROMPT_TEMPLATE_EN = get_legacy_user_template(use_english=True)

# Chat format templates
CHAT_TEMPLATE = "<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"

# Messages format for training data
def create_messages_format(input_text: str, output_text: str, use_english: bool = False, task_type: str = "FIX"):
    """
    Create messages format for training data.
    
    Args:
        input_text: Input text with grammatical errors
        output_text: Corrected text
        use_english: Whether to use English prompts
        task_type: Task type (FIX, DETECT, CORRECT, ASSESS)
        
    Returns:
        Dictionary with messages format
    """
    system_prompt = _get_system_prompt(task_type, use_english)
    user_template = _get_user_template(task_type, use_english)
    user_prompt = user_template.format(input_text=input_text)
    
    return {
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            },
            {
                "role": "assistant",
                "content": output_text
            }
        ]
    }

# Chat format for inference
def create_chat_prompt(input_text: str, use_english: bool = False, task_type: str = "FIX", **kwargs):
    """
    Create chat format prompt for inference.
    
    Args:
        input_text: Input text with grammatical errors
        use_english: Whether to use English prompts
        task_type: Task type (FIX, DETECT, CORRECT, ASSESS)
        **kwargs: Additional arguments for specific tasks (e.g., source_text, corrected_text for ASSESS)
        
    Returns:
        Formatted chat prompt string
    """
    system_message = _get_system_prompt(task_type, use_english)
    user_template = _get_user_template(task_type, use_english)
    
    # Format user message based on task type
    if task_type == "ASSESS":
        source_text = kwargs.get('source_text', input_text)
        corrected_text = kwargs.get('corrected_text', input_text)
        user_message = user_template.format(source_text=source_text, corrected_text=corrected_text)
    else:
        user_message = user_template.format(input_text=input_text)
    
    return CHAT_TEMPLATE.format(
        system_message=system_message,
        user_message=user_message
    )

# Prompt configuration
class PromptConfig:
    """Configuration class for prompt settings."""
    
    def __init__(self, use_english: bool = False):
        self.use_english = use_english
        
    @property
    def system_prompt(self):
        return SYSTEM_PROMPT_EN if self.use_english else SYSTEM_PROMPT
    
    @property
    def user_template(self):
        return USER_PROMPT_TEMPLATE_EN if self.use_english else USER_PROMPT_TEMPLATE
    
    def format_user_prompt(self, input_text: str):
        return self.user_template.format(input_text=input_text)
    
    def create_messages(self, input_text: str, output_text: str, task_type: str = "FIX"):
        return create_messages_format(input_text, output_text, self.use_english, task_type)
    
    def create_chat_prompt(self, input_text: str, task_type: str = "FIX", **kwargs):
        return create_chat_prompt(input_text, self.use_english, task_type, **kwargs)
    
    def get_system_prompt(self, task_type: str = "FIX"):
        """Get system prompt for specific task type."""
        return _get_system_prompt(task_type, self.use_english)
    
    def get_user_template(self, task_type: str = "FIX"):
        """Get user prompt template for specific task type."""
        return _get_user_template(task_type, self.use_english)


def load_yaml_prompts(config_path: str = "config/prompt_config.yaml"):
    """
    Load prompts from YAML configuration file.
    This ensures consistency between prompts.py and prompt_config.yaml
    """
    try:
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('training_objectives', {})
    except Exception as e:
        print(f"Warning: Could not load YAML prompts: {e}")
    return {}


def validate_prompt_consistency():
    """
    Validate that prompts.py loads correctly from prompt_config.yaml.
    """
    try:
        # Test loading all task types
        task_types = ['FIX', 'DETECT', 'CORRECT', 'ASSESS']
        
        for task_type in task_types:
            system_prompt = _get_system_prompt(task_type, use_english=True)
            user_template = _get_user_template(task_type, use_english=True)
            
            if not system_prompt:
                print(f"⚠️  Missing system prompt for {task_type}")
                return False
            
            if not user_template:
                print(f"⚠️  Missing user template for {task_type}")
                return False
        
        print("✅ All prompts loaded successfully from prompt_config.yaml")
        print(f"✅ Available task types: {', '.join(task_types)}")
        return True
        
    except Exception as e:
        print(f"❌ Error validating prompts: {e}")
        return False