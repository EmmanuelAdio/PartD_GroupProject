"""
Base Agent Class
Provides common functionality for all AI agents.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import logging

logging.basicConfig(level=logging.INFO)


class BaseAgent(ABC):
    """
    Abstract base class for all AI agents.
    
    All agents must implement the process() method to define their specific behavior.
    """
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        """
        Initialize the base agent.
        
        Args:
            name: Name of the agent
            config: Configuration dictionary for the agent
        """
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.logger.info(f"Initialized {name} agent")
    
    @abstractmethod
    def process(self, input_data: Any) -> Dict[str, Any]:
        """
        Process the input data and return results.
        
        Args:
            input_data: Input data to process
            
        Returns:
            Dictionary containing processing results
        """
        pass
    
    def log_info(self, message: str):
        """Log an info message."""
        self.logger.info(f"[{self.name}] {message}")
    
    def log_error(self, message: str):
        """Log an error message."""
        self.logger.error(f"[{self.name}] {message}")
