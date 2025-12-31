"""
Custom logging setup for the project.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logger(name: str, log_file: str = None, level=logging.INFO):
    """
    Setup a logger with both console and file handlers.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
    
    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Avoid duplicate handlers if setup_logger is called multiple times
    existing_handler_types = {type(h) for h in logger.handlers}
    if logging.StreamHandler not in existing_handler_types:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Only add a file handler if we don't already have one pointing to the same file
        existing_files = {getattr(h, 'baseFilename', None) for h in logger.handlers}
        if str(log_path) not in existing_files:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


def get_experiment_logger(experiment_name: str, output_dir: Path):
    """
    Create a logger for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        output_dir: Directory for log files
    
    Returns:
        Logger instance
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / "logs" / f"{experiment_name}_{timestamp}.log"
    return setup_logger(experiment_name, str(log_file))

