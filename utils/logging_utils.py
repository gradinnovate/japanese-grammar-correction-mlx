"""
Logging utilities for Japanese Grammar Correction system.
Provides centralized logging configuration and utilities.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        log_format: Optional custom log format
    """
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        format=log_format,
        handlers=[]
    )
    
    # Add console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_formatter = logging.Formatter(log_format)
    console_handler.setFormatter(console_formatter)
    
    # Add file handler if specified
    handlers = [console_handler]
    if log_file:
        # Ensure log directory exists
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(numeric_level)
        file_formatter = logging.Formatter(log_format)
        file_handler.setFormatter(file_formatter)
        handlers.append(file_handler)
    
    # Configure root logger with handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    for handler in handlers:
        root_logger.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def log_function_call(func_name: str, **kwargs) -> None:
    """
    Log a function call with its parameters.
    
    Args:
        func_name: Name of the function being called
        **kwargs: Function parameters to log
    """
    logger = get_logger(__name__)
    params = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
    logger.debug(f"Calling {func_name}({params})")


def log_data_stats(data_name: str, count: int, **stats) -> None:
    """
    Log statistics about a dataset or data processing step.
    
    Args:
        data_name: Name of the dataset
        count: Number of items
        **stats: Additional statistics to log
    """
    logger = get_logger(__name__)
    stats_str = ", ".join([f"{k}={v}" for k, v in stats.items()])
    logger.info(f"{data_name}: {count} items" + (f" ({stats_str})" if stats_str else ""))


def log_error_with_context(error: Exception, context: str) -> None:
    """
    Log an error with additional context information.
    
    Args:
        error: Exception that occurred
        context: Additional context about when/where the error occurred
    """
    logger = get_logger(__name__)
    logger.error(f"Error in {context}: {type(error).__name__}: {error}")


def log_training_progress(epoch: int, step: int, loss: float, **metrics) -> None:
    """
    Log training progress information.
    
    Args:
        epoch: Current epoch
        step: Current step
        loss: Current loss value
        **metrics: Additional metrics to log
    """
    logger = get_logger(__name__)
    metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
    logger.info(f"Epoch {epoch}, Step {step}: loss={loss:.4f}" + 
                (f", {metrics_str}" if metrics_str else ""))