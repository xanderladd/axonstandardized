# neuroncompare/src/logger.py
import logging
import os
import sys
import argparse
from typing import Optional

from neuroncompare.src.config_manager import get_config
from neuroncompare.src.path_manager import get_path_manager

class Logger:
    """
    Centralized logging system for the neuroncompare pipeline.
    Provides consistent logging across all components.
    """
    
    # Log levels
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL
    
    def __init__(self, name: str = 'neuroncompare', log_file: Optional[str] = None, 
                 log_level: int = logging.INFO, to_console: bool = True):
        """
        Initialize the Logger.
        
        Args:
            name: Logger name
            log_file: Path to log file. If None, logs to run_dir/logs/pipeline.log
            log_level: Logging level
            to_console: If True, also log to console
        """
        self.name = name
        self.config = get_config()
        self.path_manager = get_path_manager()
        
        # Create logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        self.logger.propagate = False
        
        # Clear any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Create log file path if None
        if log_file is None:
            log_dir = self.path_manager.get_subdirectory('logs')
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, 'pipeline.log')
        
        # Create file handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add file handler to logger
        self.logger.addHandler(file_handler)
        
        # Add console handler if requested
        if to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # Log initial message
        self.logger.info(f"Logger initialized: {name}")
    
    def debug(self, message: str, **kwargs) -> None:
        """
        Log a debug message.
        
        Args:
            message: Message to log
            **kwargs: Additional context to include in the log
        """
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs) -> None:
        """
        Log an info message.
        
        Args:
            message: Message to log
            **kwargs: Additional context to include in the log
        """
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs) -> None:
        """
        Log a warning message.
        
        Args:
            message: Message to log
            **kwargs: Additional context to include in the log
        """
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs) -> None:
        """
        Log an error message.
        
        Args:
            message: Message to log
            **kwargs: Additional context to include in the log
        """
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs) -> None:
        """
        Log a critical message.
        
        Args:
            message: Message to log
            **kwargs: Additional context to include in the log
        """
        self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs) -> None:
        """
        Internal method to log a message with context.
        
        Args:
            level: Logging level
            message: Message to log
            **kwargs: Additional context to include in the log
        """
        # Add context to message if kwargs provided
        if kwargs:
            context_str = ' '.join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} [{context_str}]"
        
        self.logger.log(level, message)
    
    def log_stage_start(self, stage_name: str) -> None:
        """
        Log the start of a pipeline stage.
        
        Args:
            stage_name: Name of the stage
        """
        self.info(f"Starting stage: {stage_name}")
    
    def log_stage_end(self, stage_name: str, success: bool = True) -> None:
        """
        Log the end of a pipeline stage.
        
        Args:
            stage_name: Name of the stage
            success: Whether the stage completed successfully
        """
        if success:
            self.info(f"Completed stage: {stage_name}")
        else:
            self.error(f"Failed stage: {stage_name}")
    
    def log_command(self, command: str) -> None:
        """
        Log a command being executed.
        
        Args:
            command: Command being executed
        """
        self.debug(f"Executing command: {command}")
    
    def create_stage_logger(self, stage_name: str) -> 'Logger':
        """
        Create a logger for a specific pipeline stage.
        
        Args:
            stage_name: Name of the stage
            
        Returns:
            Logger for the stage
        """
        log_dir = self.path_manager.get_subdirectory('logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{stage_name}.log")
        
        return Logger(f"{self.name}.{stage_name}", log_file)

# Global instance for singleton-like access
_logger_instance = None

def get_logger(name: str = 'neuroncompare', log_file: Optional[str] = None, 
               log_level: int = logging.INFO, to_console: bool = True) -> Logger:
    """
    Get a global instance of the Logger.
    
    Args:
        name: Logger name
        log_file: Path to log file
        log_level: Logging level
        to_console: If True, also log to console
        
    Returns:
        Logger instance
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = Logger(name, log_file, log_level, to_console)
    return _logger_instance

# Command Line Interface
def cli_main():
    """
    Command-line interface for logging messages from bash scripts.
    """
    parser = argparse.ArgumentParser(description='Log a message from shell scripts')
    parser.add_argument('--level', choices=['debug', 'info', 'warning', 'error', 'critical'],
                        default='info', help='Log level')
    parser.add_argument('--stage', help='Pipeline stage name')
    parser.add_argument('message', help='Message to log')
    args = parser.parse_args()
    
    logger = get_logger()
    
    if args.stage:
        logger = logger.create_stage_logger(args.stage)
    
    log_levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL
    }
    
    logger._log(log_levels[args.level], args.message)

if __name__ == "__main__":
    cli_main()