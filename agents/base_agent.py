from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from models import AnalysisResult, TestResultInfo, FocalMethodInfo
from llm import LLMClient
from utils import logger, get_sample_logger

class BaseAgent(ABC):
    """Base class for all agents"""
    
    def __init__(self, name: str, agent_type: str):
        self.name = name
        self.agent_type = agent_type  # 'analyzer', 'error_handler', 'updater', 'coordinator'
        self.llm_client = LLMClient(agent_name=name)  # Pass agent name for logging
        self.logger = logger
        self.sample_logger = get_sample_logger()
        
    @abstractmethod
    def execute(self, *args, **kwargs) -> Any:
        """Execute agent logic"""
        pass
    
    def log_info(self, message: str):
        """Log info message"""
        self.logger.info(f"[{self.name}] {message}")
        # Also log to sample logger if initialized
        if self.sample_logger.log_file:
            self.sample_logger.log_info(f"[{self.name}] {message}")
    
    def log_error(self, message: str):
        """Log error message"""
        self.logger.error(f"[{self.name}] {message}")
        if self.sample_logger.log_file:
            self.sample_logger.log_error(f"[{self.name}] {message}")
    
    def log_warning(self, message: str):
        """Log warning message"""
        self.logger.warning(f"[{self.name}] {message}")
        if self.sample_logger.log_file:
            self.sample_logger.log_warning(f"[{self.name}] {message}")
    
    def log_debug(self, message: str):
        """Log debug message"""
        self.logger.debug(f"[{self.name}] {message}")
        if self.sample_logger.log_file:
            self.sample_logger.log_debug(f"[{self.name}] {message}")
    
    def log_analysis_result(self, analysis: Any):
        """Log analysis result to sample logger"""
        if self.sample_logger.log_file:
            self.sample_logger.log_analysis_result(self.name, analysis)
    
    def log_custom(self, category: str, data: Any, description: str = ""):
        """Log custom data to sample logger"""
        if self.sample_logger.log_file:
            self.sample_logger.log_custom(category, data, description)
