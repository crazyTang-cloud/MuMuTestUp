"""
Data structures for the test fix pipeline.

Defines the input/output models and status enums for the multi-phase
retrieval pipeline.
"""
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Optional, Dict, Any, List
from datetime import datetime
import json


class PipelineStatus(str, Enum):
    """Overall pipeline completion status."""
    SKIPPED_AT_PHASE_1 = "SKIPPED_AT_PHASE_1"
    STOPPED_AT_PHASE_2 = "STOPPED_AT_PHASE_2"
    FAILED_AT_PHASE_3 = "FAILED_AT_PHASE_3"
    SUCCESS_PHASE_3 = "SUCCESS_PHASE_3"
    ERROR = "ERROR"


class PhaseStatus(str, Enum):
    """Individual phase execution status."""
    SUCCESS = "success"
    SKIP = "skip"
    PROCEED = "proceed"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class PipelineInput:
    """
    Input configuration for the pipeline.
    
    Wraps all necessary configuration from config.Config with additional
    pipeline-specific parameters.
    """
    # Repository information
    repo_path: str
    error_commit_id: str
    
    # Error information
    error_test_code_log: str
    error_message: str
    error_log: str
    
    # LLM configuration
    agent_model: str
    openai_api_key: str
    openai_base_url: str
    temperature: float = 0.1  # LLM temperature
    timeout: int = 3000  # Request timeout in seconds
    max_retries: int = 3  # Maximum retry attempts
    
    # Pipeline options
    skip_phase0: bool = False  # Skip git reset (for testing)
    skip_rebuild: bool = False  # Skip index rebuild (for iteration)
    max_sql_rounds: int = 3
    max_rag_rounds: int = 3
    rag_top_k: int = 8
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.
        
        Sensitive fields like API keys and base URLs are removed to avoid
        leaking secrets in logs or result files.
        """
        data = asdict(self)
        # Remove sensitive credentials
        data.pop("openai_api_key", None)
        data.pop("openai_base_url", None)
        return data
    
    @classmethod
    def from_config(cls, config, **kwargs):
        """
        Create PipelineInput from a config.Config instance.
        
        This method now reads API settings and temperature from main config.py
        for consistency across the entire BEAM framework.
        
        Args:
            config: Instance of config.Config (from index_retrieve_module_v3)
            **kwargs: Additional override parameters
        """
        import os
        import sys
        from pathlib import Path
        
        # Try to import main BEAM config for API settings and temperature
        try:
            # Add parent directory to import main config
            parent_dir = Path(__file__).parent.parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            
            from config import config as main_config
            
            # Use main config for API settings
            api_key = main_config.llm.api_key or os.getenv("OPENAI_API_KEY", "")
            
            # Get base URL from main config
            api_url = main_config.llm.api_url
            if api_url:
                base_url = api_url.rstrip('/')
                if not base_url.endswith('/v1'):
                    base_url = base_url + '/v1'
            else:
                base_url = main_config.llm.get_base_url()
                if not base_url.endswith('/v1'):
                    base_url = base_url + '/v1'
            
            # Use temperature from main config
            temperature = main_config.llm.temperature
            
            # Use timeout from main config
            timeout = main_config.llm.timeout
            
            # Use max_retries from main config
            max_retries = main_config.llm.max_retries
            
        except ImportError:
            # Fallback to environment variables
            api_key = os.getenv("OPENAI_API_KEY", "")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            temperature = 0.1  # Default fallback
            timeout = 3000  # Default fallback
            max_retries = 3  # Default fallback
        
        return cls(
            repo_path=config.repo_path,
            error_commit_id=config.error_commit_id,
            error_test_code_log=config.error_test_code_log,
            error_message=config.error_message,
            error_log=config.error_log,
            agent_model=config.agent_model,
            openai_api_key=api_key,
            openai_base_url=base_url,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )


@dataclass
class PhaseResult:
    """
    Result from a single phase execution.
    
    Contains the decision, reasoning, and any retrieved context.
    """
    phase_name: str
    status: PhaseStatus
    reason: str
    context: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    rounds: int = 1
    duration_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'phase_name': self.phase_name,
            'status': self.status.value,
            'reason': self.reason,
            'context': self.context,
            'metadata': self.metadata,
            'rounds': self.rounds,
            'duration_seconds': round(self.duration_seconds, 2)
        }


@dataclass
class PipelineOutput:
    """
    Complete output from the pipeline execution.
    
    Contains results from all executed phases, final status, and metadata.
    """
    # Overall status
    final_status: PipelineStatus
    final_reason: str
    final_context: Optional[str] = None
    
    # Phase results
    phase0_result: Optional[PhaseResult] = None
    phase1_result: Optional[PhaseResult] = None
    phase2_result: Optional[PhaseResult] = None
    phase3_result: Optional[PhaseResult] = None
    
    # Execution metadata
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    total_duration_seconds: float = 0.0
    error_message: Optional[str] = None
    
    # Configuration snapshot
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'timestamp': self.timestamp,
            'final_status': self.final_status.value,
            'final_reason': self.final_reason,
            'final_context': self.final_context,
            'phase0': self.phase0_result.to_dict() if self.phase0_result else None,
            'phase1': self.phase1_result.to_dict() if self.phase1_result else None,
            'phase2': self.phase2_result.to_dict() if self.phase2_result else None,
            'phase3': self.phase3_result.to_dict() if self.phase3_result else None,
            'total_duration_seconds': round(self.total_duration_seconds, 2),
            'error_message': self.error_message,
            'config': self.config_snapshot
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    def save_to_file(self, filepath: str):
        """Save output to JSON file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.to_json())


@dataclass
class SQLQueryResult:
    """Result from a SQL query execution."""
    query: str
    success: bool
    results: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    row_count: int = 0


@dataclass
class RAGSearchResult:
    """Result from a RAG semantic search."""
    query: str
    target: str  # "methods", "fields", or "both"
    results: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = True
    error: Optional[str] = None

