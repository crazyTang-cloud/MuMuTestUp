"""
Configuration management for the indexer.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class IndexerConfig:
    """Configuration for the indexer."""
    
    # Project paths
    project_root: Path
    output_dir: Path
    
    # OpenAI configuration
    openai_api_key: str
    openai_base_url: str
    
    # Indexing options
    sqlite_db_name: str = "assets.db"
    chroma_collection_name: str = "java_methods"
    
    # Directories to ignore during traversal
    ignored_dirs: tuple = ("target", ".git", ".idea", ".vscode", "node_modules", "build", "out")
    
    # File patterns
    java_file_pattern: str = "*.java"
    
    @classmethod
    def from_main_config(cls, project_root: str, output_dir: Optional[str] = None) -> "IndexerConfig":
        """
        Create configuration from main config.py (RECOMMENDED).
        
        Args:
            project_root: Path to the Java project
            output_dir: Output directory for index data (optional)
        
        Returns:
            IndexerConfig instance
        """
        # Import main config from parent directory
        try:
            # Add parent directory to path to import main config
            parent_dir = Path(__file__).parent.parent.parent
            if str(parent_dir) not in sys.path:
                sys.path.insert(0, str(parent_dir))
            
            from config import config as main_config
            
            # Use main config settings
            api_key = main_config.llm.api_key
            if not api_key:
                raise ValueError("API key not configured in main config.py")
            
            # Get base URL from main config using get_base_url() method
            # This properly extracts the base URL from api_url if it contains full path
            base_url = main_config.llm.get_base_url()
            if base_url:
                base_url = base_url.rstrip('/')
                if not base_url.endswith('/v1'):
                    base_url = base_url + '/v1'
            else:
                base_url = "https://api.openai.com/v1"
            
        except ImportError:
            # Fallback to environment variables if main config not available
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("API key not found in config.py or OPENAI_API_KEY environment variable")
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        # Get project root
        project_path = Path(project_root).resolve()
        if not project_path.exists():
            raise ValueError(f"Project root does not exist: {project_path}")
        
        # Output directory
        if output_dir is None:
            output_dir = Path.cwd() / "index_data"
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        return cls(
            project_root=project_path,
            output_dir=output_dir,
            openai_api_key=api_key,
            openai_base_url=base_url
        )
    
    @classmethod
    def from_env(cls, project_root: Optional[str] = None) -> "IndexerConfig":
        """
        Create configuration from environment variables (LEGACY).
        
        DEPRECATED: Use from_main_config() instead to ensure consistency with main config.py
        
        Args:
            project_root: Path to the Java project
            
        Returns:
            IndexerConfig instance
        """
        # Get API configuration from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        # Get project root
        if project_root is None:
            project_root = os.getenv("JAVA_PROJECT_ROOT")
            if not project_root:
                raise ValueError("Project root must be provided via --project-root or JAVA_PROJECT_ROOT env var")
        
        project_path = Path(project_root).resolve()
        if not project_path.exists():
            raise ValueError(f"Project root does not exist: {project_path}")
        
        # Output directory (in current working directory)
        output_dir = Path.cwd() / "index_data"
        output_dir.mkdir(exist_ok=True)
        
        return cls(
            project_root=project_path,
            output_dir=output_dir,
            openai_api_key=api_key,
            openai_base_url=base_url
        )
    
    @property
    def sqlite_path(self) -> Path:
        """Get full path to SQLite database."""
        return self.output_dir / self.sqlite_db_name
    
    @property
    def chroma_path(self) -> Path:
        """Get full path to ChromaDB directory."""
        return self.output_dir / "chroma_db"

