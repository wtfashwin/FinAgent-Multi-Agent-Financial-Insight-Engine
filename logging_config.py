"""
Logging configuration for FinAgent
Provides structured logging with different levels for development and production
"""
import logging
import logging.handlers
import os
from pathlib import Path


def setup_logging(log_level: str = None, log_file: str = None):
    """
    Setup logging configuration for the application

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
    """
    # Get log level from environment or parameter
    level = log_level or os.getenv('LOG_LEVEL', 'INFO')
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Create logs directory if it doesn't exist
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)

    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    # Remove existing handlers
    root_logger.handlers = []

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    # File handler (if log_file specified)
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(file_handler)

    # Set specific loggers to appropriate levels
    # Reduce verbosity of some third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('chromadb').setLevel(logging.WARNING)

    logging.info(f"Logging configured with level: {level}")


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


# Default setup for development
if __name__ != "__main__":
    # Auto-configure logging when imported
    environment = os.getenv('ENVIRONMENT', 'development')

    if environment == 'production':
        # Production: Log to file with INFO level
        setup_logging(log_level='INFO', log_file='logs/finagent.log')
    else:
        # Development: Log to console with DEBUG level
        setup_logging(log_level='DEBUG')
