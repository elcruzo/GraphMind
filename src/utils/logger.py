"""
Logging utilities for GraphMind
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import structlog
from datetime import datetime


def setup_logger(
    name: str = "graphmind",
    level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = True
) -> logging.Logger:
    """
    Setup logger with structured logging support
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        structured: Use structured logging
        
    Returns:
        Configured logger
    """
    
    if structured:
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        logger = structlog.get_logger(name)
    else:
        # Standard logging
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger instance"""
    return structlog.get_logger(name)


class LoggerAdapter:
    """
    Adapter for distributed logging with rank information
    """
    
    def __init__(self, logger: logging.Logger, rank: int, world_size: int):
        self.logger = logger
        self.rank = rank
        self.world_size = world_size
        
    def _add_rank_info(self, msg: str) -> str:
        return f"[Rank {self.rank}/{self.world_size}] {msg}"
    
    def debug(self, msg: str, **kwargs):
        self.logger.debug(self._add_rank_info(msg), **kwargs)
        
    def info(self, msg: str, **kwargs):
        self.logger.info(self._add_rank_info(msg), **kwargs)
        
    def warning(self, msg: str, **kwargs):
        self.logger.warning(self._add_rank_info(msg), **kwargs)
        
    def error(self, msg: str, **kwargs):
        self.logger.error(self._add_rank_info(msg), **kwargs)
        
    def critical(self, msg: str, **kwargs):
        self.logger.critical(self._add_rank_info(msg), **kwargs)