# Configuration for BEAM (Test Update Framework)
from dataclasses import dataclass
from typing import Optional, Literal
import os

@dataclass
class LLMConfig:
    """LLM configuration - supports multiple providers"""
    # Provider type: "ollama" for local Ollama, "openai" for OpenAI-compatible APIs
    provider: Literal["ollama", "openai"] = "openai"
    
    # Ollama specific settings
    ollama_host: str = "http://localhost"
    ollama_port: int = 11434
    
    
    api_key: Optional[str] = ""  
    api_url: Optional[str] = ""  
    
    # Common settings
    model: str = "deepseek-v3.2"  # Model name
    timeout: int = 3000  # Request timeout in seconds
    temperature: float = 0.0  # Generation temperature
    
    # Retry settings (for unstable APIs like third-party proxies)
    max_retries: int = 4  # Maximum number of retries on failure
    retry_delay: float = 1.0  # Initial retry delay in seconds (uses exponential backoff)
    
    def get_base_url(self) -> str:
        """Get the base URL based on provider"""
        if self.provider == "ollama":
            return f"{self.ollama_host}:{self.ollama_port}"
        elif self.provider == "openai":
            # For OpenAI-compatible APIs, return base URL (without /chat/completions)
            if self.api_url:
                # Remove trailing endpoint if present
                base = self.api_url.rstrip('/')
                if base.endswith('/chat/completions'):
                    base = base[:-len('/chat/completions')]
                if base.endswith('/v1'):
                    base = base[:-len('/v1')]
                return base
            return "https://api.openai.com"
        return ""

# Backward compatibility
@dataclass
class OllamaConfig:
    """Ollama LLM configuration (Deprecated - use LLMConfig instead)"""
    host: str = "http://localhost"
    port: int = 11434
    model: str = "qwen2.5-coder:7b"  # Default model, can be changed
    timeout: int = 3000

@dataclass
class JavaConfig:
    """Java execution configuration"""
    java_homes: dict = None  # {version: path}
    maven_home: str = "/data/david/maven/apache-maven-3.9.6"  # Default Maven path
    maven_repo_dir: Optional[str] = None  # Local Maven repository path (None = auto-generate per project)
    maven_repo_base: str = "/data/david/maven_repo"  # Base directory for Maven repositories
    repos_dir: str = "/data/david/project/beam/repos"  # Where to clone repos
    logs_base_dir: str = "/data/david/project/beam/logs"  # Base directory for logs (suffix will be added based on ablation)
    reports_base_dir: str = "/data/david/project/beam/reports"  # Base directory for reports (suffix will be added based on ablation)
    github_tokens: list = None  # List of GitHub tokens
    test_timeout: int = 6000  # Test execution timeout (seconds) - increased for mutation testing
    
    def __post_init__(self):
        if self.java_homes is None:
            # Default Java paths - PLEASE UPDATE THESE TO YOUR ACTUAL PATHS
            self.java_homes = {
                "8": "/data/david/java/jdk1.8.0_391",
                "11": "/data/david/java/jdk-11.0.22",
                "17": "/data/david/java/jdk-17.0.10",
                "21": "/data/david/java/jdk-21.0.1"
            }
        if self.github_tokens is None:
            # Add your GitHub tokens here for better API rate limits
            # Format: ["ghp_xxxxxxxxxxxxx", "ghp_yyyyyyyyyyyyy"]
            self.github_tokens = []
    
    def get_maven_repo_path(self, project_name: str = None) -> str:
        """
        Get the Maven repository path.
        
        Args:
            project_name: Optional project name to generate project-specific repo path
                         Format: "org/project" -> extracts "project" only
            
        Returns:
            Path to Maven repository
        """
        # If custom repo path is configured, use it
        if self.maven_repo_dir:
            return self.maven_repo_dir
        
        # If project name provided, generate project-specific repo path
        if project_name:
            from pathlib import Path
            # Extract project name only (e.g., "dromara/hutool" -> "hutool")
            project_only = project_name.split('/')[-1]
            # Generate path: /data/david/maven_repo/{project}_repo
            project_repo = Path(self.maven_repo_base) / f"{project_only}_repo"
            return str(project_repo)
        
        # Otherwise use default Maven repo (~/.m2/repository)
        return None
    
    def get_logs_dir(self, framework_config: 'FrameworkConfig' = None) -> str:
        """
        Get the logs directory with ablation suffix.
        
        Args:
            framework_config: FrameworkConfig instance to get ablation suffix
            
        Returns:
            Path to logs directory (e.g., /data/david/project/beam/logs_wo_mut)
        """
        suffix = framework_config.get_ablation_suffix() if framework_config else ""
        return f"{self.logs_base_dir}{suffix}"
    
    def get_reports_dir(self, framework_config: 'FrameworkConfig' = None) -> str:
        """
        Get the reports directory with ablation suffix.
        
        Args:
            framework_config: FrameworkConfig instance to get ablation suffix
            
        Returns:
            Path to reports directory (e.g., /data/david/project/beam/reports_wo_mut)
        """
        suffix = framework_config.get_ablation_suffix() if framework_config else ""
        return f"{self.reports_base_dir}{suffix}"
    
    def get_sample_reports_dir(self, project_name: str, sample_id: str, framework_config: 'FrameworkConfig' = None) -> str:
        """
        Get the reports directory for a specific sample.
        
        Args:
            project_name: Project name (e.g., "dromara/hutool")
            sample_id: Sample ID (e.g., "13", "dromara/hutool:13", or any unique identifier)
            framework_config: FrameworkConfig instance to get ablation suffix
            
        Returns:
            Path to sample reports directory (e.g., reports_wo_mut/dromara/hutool/13/)
        """
        from pathlib import Path
        
        # Extract numeric ID if sample_id contains project name (e.g., "dromara/hutool:13" -> "13")
        if ':' in sample_id:
            sample_id = sample_id.split(':')[-1]
        
        # Replace any remaining slashes with underscores for filesystem safety
        sample_id = sample_id.replace('/', '_')
        
        # Get reports directory with ablation suffix
        reports_dir = self.get_reports_dir(framework_config)
        
        # Create path: reports_dir/project_name/sample_id/
        sample_reports = Path(reports_dir) / project_name / sample_id
        return str(sample_reports)

@dataclass
class FrameworkConfig:
    """Framework configuration"""
    max_iterations: int = 4
    max_retries: int = 3
    log_level: str = "INFO"
    log_file: Optional[str] = "beam.log"
    
    # Coverage and mutation thresholds (unified for all code)
    coverage_threshold: float = 1.0    # 70% line coverage for all code
    branch_coverage_threshold: float = 1.0    # 70% branch coverage (only checked if branches exist)
    mutation_threshold: float = 1.0    # 70% mutation kill rate for all code
    
    # Ablation experiment: use prioritized_changes instead of hunk filtering
    # When True, uses prioritized_changes from dataset directly without root_cause_analysis_agent filtering
    # When False, uses test_method, focal_method, focal_file, high_frequency hunks with root_cause_analysis_agent filtering
    # Note: class_fields and non_test_methods are always filtered by root_cause_analysis_agent
    use_target_prioritized_changes: bool = True
    
    # Ablation experiments: disable specific agents
    # When an agent is disabled, it is skipped and replaced with simplified instructions
    # Note: Test execution (coverage & mutation) still runs to determine iteration completion
    ablation_disable_mutation: bool = False      # Skip mutation_analysis_agent, use "杀死的变异体不足"
    ablation_disable_coverage: bool = False      # Skip coverage_analysis_agent, use "覆盖率不足"
    ablation_coverage_lines_only: bool = False   # Skip coverage_analysis_agent, return only uncovered lines
    ablation_disable_retrieval: bool = False     # Skip retrieval_agent (affects error_analyze_agent prompt)
    ablation_disable_error: bool = False         # Skip error_analyze_agent, use raw error log
    
    # More thorough ablation experiments (completely disable components)
    # These are more radical than the above ablation settings
    all_ablation_disable_mutation: bool = False  # Completely disable mutation: no computation, no agent, no scoring, no threshold check
    all_ablation_disable_coverage: bool = False  # Completely disable coverage: no computation, no agents, no scoring, no threshold check
    all_ablation_disable_error: bool = False     # Completely disable error feedback: no iteration, one-shot generation only
    
    # Retrieval ablation: limit retrieval to single-round SQLite only
    ablation_sqlite_single_round: bool = False   # Only use SQLite with 1 round (no iteration, no ChromaDB)
    
    # Retrieval ablation: different retrieval strategies
    ablation_sql_only_3rounds: bool = False       # Only use SQL retrieval with 3 rounds (no RAG)
    ablation_rag_only_3rounds: bool = True      # Only use RAG retrieval with 3 rounds (no SQL)
    ablation_sql_1round_rag_3rounds: bool = False # Use 1 round SQL retrieval, then 3 rounds RAG retrieval
    
    # Retrieval ablation: use LSP instead of SQLite/ChromaDB
    ablation_use_lsp: bool = False               # Use LSP (Language Server Protocol) for retrieval instead of SQLite/ChromaDB
    
    # Ablation study: baseline rerun setting
    # When enabled, if any ablation config performs better than baseline, rerun baseline once
    # Scoring: compile fail=-1000, run fail=-100, success=1000+(line_cov+branch_cov+mutation)*10
    ablation_rerun_baseline: bool = False         # Enable baseline rerun if ablation outperforms
    
    def get_ablation_suffix(self) -> str:
        """
        Generate suffix based on active ablation settings.
        
        Returns:
            Suffix string (e.g., "_wo_mut_wo_cov" or "" if no ablation)
        """
        suffixes = []
        
        # Thorough ablation settings (higher priority in naming)
        if self.all_ablation_disable_mutation:
            suffixes.append("_all_wo_mut")
        if self.all_ablation_disable_coverage:
            suffixes.append("_all_wo_cov")
        if self.all_ablation_disable_error:
            suffixes.append("_all_wo_error")
        
        # Regular ablation settings
        if self.ablation_disable_mutation:
            suffixes.append("_wo_mut")
        if self.ablation_disable_coverage:
            suffixes.append("_wo_cov")
        if self.ablation_coverage_lines_only:
            suffixes.append("_cov_lines_only")
        if self.ablation_disable_retrieval:
            suffixes.append("_wo_retrieval")
        if self.ablation_disable_error:
            suffixes.append("_wo_error")
        if self.ablation_sqlite_single_round:
            suffixes.append("_sqlite_1round")
        if self.ablation_sql_only_3rounds:
            suffixes.append("_sql_only_3rounds")
        if self.ablation_rag_only_3rounds:
            suffixes.append("_rag_only_3rounds")
        if self.ablation_sql_1round_rag_3rounds:
            suffixes.append("_sql_1round_rag_3rounds")
        if self.ablation_use_lsp:
            suffixes.append("_use_lsp")
        if self.use_target_prioritized_changes:
            suffixes.append("_w_prior")
        
        return "".join(suffixes)

@dataclass
class Config:
    """Main configuration"""
    llm: LLMConfig = None  # New unified LLM config
    ollama: OllamaConfig = None  # Deprecated, kept for backward compatibility
    framework: FrameworkConfig = None
    java: JavaConfig = None
    
    def __post_init__(self):
        # Initialize LLM config first - with environment variable support
        if self.llm is None:
            # Check for environment variables
            provider = os.getenv("LLM_PROVIDER", "ollama")
            api_url = os.getenv("LLM_API_URL")
            api_key = os.getenv("LLM_API_KEY")
            model = os.getenv("LLM_MODEL", "qwen2.5-coder:7b")
            temperature = float(os.getenv("LLM_TEMPERATURE", "0.7"))
            timeout = int(os.getenv("LLM_TIMEOUT", "3000"))
            
            # Ollama-specific env vars
            ollama_host = os.getenv("OLLAMA_HOST", "http://localhost")
            ollama_port = int(os.getenv("OLLAMA_PORT", "11434"))
            
            # Retry settings
            max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
            retry_delay = float(os.getenv("LLM_RETRY_DELAY", "1.0"))
            
            self.llm = LLMConfig(
                provider=provider,
                ollama_host=ollama_host,
                ollama_port=ollama_port,
                api_url=api_url,
                api_key=api_key,
                model=model,
                timeout=timeout,
                temperature=temperature,
                max_retries=max_retries,
                retry_delay=retry_delay
            )
        
        # For backward compatibility, sync with ollama config if it exists
        if self.ollama is None:
            self.ollama = OllamaConfig()
        
        if self.framework is None:
            self.framework = FrameworkConfig()
        if self.java is None:
            self.java = JavaConfig()

# Global config instance
config = Config(llm=LLMConfig())
