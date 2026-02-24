import logging
from config import config

def setup_logger(name: str = __name__) -> logging.Logger:
    """Setup logger for the application"""
    logger = logging.getLogger(name)
    logger.setLevel(config.framework.log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(config.framework.log_level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    
    # File handler if configured
    if config.framework.log_file:
        file_handler = logging.FileHandler(config.framework.log_file)
        file_handler.setLevel(config.framework.log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logger(__name__)
