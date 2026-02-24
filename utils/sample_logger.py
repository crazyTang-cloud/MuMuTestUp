"""
Sample-level Detailed Logger for BEAM Framework

Creates detailed logs for each sample being processed, including:
- All prompts sent to LLM
- All LLM responses
- Intermediate analysis results
- Coverage and mutation details
- Test code at each iteration
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional
from dataclasses import asdict, is_dataclass


class SampleLogger:
    """
    Detailed logger for individual samples.
    
    Creates a separate log file for each sample in the appropriate directory structure.
    For example, if processing dataset at:
        D:/project/python/beam/dataset/dromara/hutool/data.json
    
    Logs will be saved to:
        D:/project/python/beam/dataset/logs/dromara/hutool/<sample_name>_<timestamp>.log
    """
    
    _instance: Optional['SampleLogger'] = None
    _initialized: bool = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if SampleLogger._initialized:
            return
        
        self.log_dir: Optional[Path] = None
        self.sample_name: str = ""
        self.sample_id: str = ""
        self.log_file: Optional[Path] = None
        self.file_handler: Optional[logging.FileHandler] = None
        self.logger: Optional[logging.Logger] = None
        self.iteration: int = 0
        self.step: str = ""
        self._log_buffer: List[Dict] = []
        self._json_log_path: Optional[Path] = None
        
        SampleLogger._initialized = True
    
    @classmethod
    def get_instance(cls) -> 'SampleLogger':
        """Get singleton instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def initialize(self, dataset_path: str, sample_name: str, sample_index: int = 0,
                   sample_id: str = None):
        """
        Initialize logger for a specific sample.
        
        Args:
            dataset_path: Path to the dataset JSON file
            sample_name: Name of the test case/sample
            sample_index: Index of the sample in the dataset
            sample_id: Unique sample ID (e.g., "dromara/hutool:13")
        """
        # Use config to get logs directory with ablation suffix
        from config import config
        logs_base_dir = config.java.get_logs_dir(config.framework)
        
        # Parse dataset path to determine log directory
        dataset_path = Path(dataset_path)
        dataset_dir = dataset_path.parent  # e.g., dataset/dromara/hutool
        
        # Get the relative path from 'dataset' directory
        # e.g., dromara/hutool
        try:
            dataset_base = None
            for parent in dataset_path.parents:
                if parent.name == 'dataset':
                    dataset_base = parent
                    break
            
            if dataset_base:
                relative_path = dataset_dir.relative_to(dataset_base)
            else:
                relative_path = Path(dataset_dir.name)
        except ValueError:
            relative_path = Path(dataset_dir.name)
        
        # Create log directory with ablation suffix: logs_wo_mut/dromara/hutool/
        logs_base = Path(logs_base_dir)
        self.log_dir = logs_base / relative_path
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Use sample_id for filename if provided
        # sample_id format: "dromara/hutool:13" -> "dromara_hutool_13"
        if sample_id:
            clean_id = self._clean_filename(sample_id.replace('/', '_').replace(':', '_'))
            log_name = clean_id
        else:
            clean_name = self._clean_filename(sample_name)
            log_name = f"sample_{sample_index:04d}_{clean_name}"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create log files
        # Text log for human-readable output
        self.log_file = self.log_dir / f"{log_name}_{timestamp}.log"
        
        # JSON log for structured data
        self._json_log_path = self.log_dir / f"{log_name}_{timestamp}.json"
        
        self.sample_name = sample_name
        self.sample_id = sample_id or f"sample_{sample_index}"
        self.iteration = 0
        self.step = ""
        self._log_buffer = []
        
        # Setup file logger
        self._setup_file_logger()
        
        # Log initialization
        self.log_header("BEAM Framework - Sample Processing Log")
        self.log_info(f"Sample ID: {self.sample_id}")
        self.log_info(f"Sample Name: {sample_name}")
        self.log_info(f"Sample Index: {sample_index}")
        self.log_info(f"Dataset: {dataset_path}")
        self.log_info(f"Log File: {self.log_file}")
        self.log_info(f"JSON Log: {self._json_log_path}")
        self.log_info(f"Started: {datetime.now().isoformat()}")
        self.log_separator()
    
    def _clean_filename(self, name: str) -> str:
        """Clean a string to be used as filename"""
        # Replace invalid characters
        invalid_chars = '<>:"/\\|?*()'
        clean = name
        for char in invalid_chars:
            clean = clean.replace(char, '_')
        # Limit length
        if len(clean) > 50:
            clean = clean[:50]
        return clean
    
    def _setup_file_logger(self):
        """Setup file-based logger"""
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)
        
        self.logger = logging.getLogger(f"sample_logger_{id(self)}")
        self.logger.setLevel(logging.DEBUG)
        self.logger.handlers = []  # Clear existing handlers
        
        self.file_handler = logging.FileHandler(self.log_file, encoding='utf-8')
        self.file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S')
        self.file_handler.setFormatter(formatter)
        
        self.logger.addHandler(self.file_handler)
    
    def set_iteration(self, iteration: int):
        """Set current iteration number"""
        self.iteration = iteration
        self.log_header(f"ITERATION {iteration}")
    
    def set_step(self, step: str):
        """Set current step name"""
        self.step = step
        self.log_subheader(step)
    
    def log_header(self, title: str):
        """Log a major section header"""
        if self.logger:
            self.logger.info("=" * 80)
            self.logger.info(f"  {title}")
            self.logger.info("=" * 80)
    
    def log_subheader(self, title: str):
        """Log a subsection header"""
        if self.logger:
            self.logger.info("")
            self.logger.info("-" * 60)
            self.logger.info(f"  {title}")
            self.logger.info("-" * 60)
    
    def log_separator(self):
        """Log a separator line"""
        if self.logger:
            self.logger.info("-" * 80)
    
    def log_info(self, message: str):
        """Log info message"""
        if self.logger:
            prefix = f"[Iter {self.iteration}]" if self.iteration > 0 else ""
            self.logger.info(f"{prefix} {message}")
    
    def log_debug(self, message: str):
        """Log debug message"""
        if self.logger:
            prefix = f"[Iter {self.iteration}]" if self.iteration > 0 else ""
            self.logger.debug(f"{prefix} {message}")
    
    def log_error(self, message: str):
        """Log error message"""
        if self.logger:
            prefix = f"[Iter {self.iteration}]" if self.iteration > 0 else ""
            self.logger.error(f"{prefix} ERROR: {message}")
    
    def log_warning(self, message: str):
        """Log warning message"""
        if self.logger:
            prefix = f"[Iter {self.iteration}]" if self.iteration > 0 else ""
            self.logger.warning(f"{prefix} WARNING: {message}")
    
    def log_prompt(self, agent_name: str, prompt_type: str, 
                   system_prompt: str, user_prompt: str):
        """
        Log a prompt sent to LLM
        
        Args:
            agent_name: Name of the agent sending the prompt
            prompt_type: Type of prompt (e.g., 'analysis', 'update', 'root_cause')
            system_prompt: The system prompt
            user_prompt: The user prompt
        """
        self.log_subheader(f"LLM PROMPT - {agent_name} ({prompt_type})")
        
        self.log_info("=== SYSTEM PROMPT ===")
        if self.logger:
            for line in system_prompt.split('\n'):
                self.logger.info(f"  {line}")
        
        self.log_info("")
        self.log_info("=== USER PROMPT ===")
        if self.logger:
            for line in user_prompt.split('\n'):
                self.logger.info(f"  {line}")
        
        self.log_separator()
        
        # Add to structured log
        self._add_to_json_log({
            "type": "prompt",
            "iteration": self.iteration,
            "step": self.step,
            "agent": agent_name,
            "prompt_type": prompt_type,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_response(self, agent_name: str, response: str, 
                     parsed_result: Any = None):
        """
        Log LLM response
        
        Args:
            agent_name: Name of the agent
            response: Raw LLM response
            parsed_result: Parsed/structured result (optional)
        """
        self.log_subheader(f"LLM RESPONSE - {agent_name}")
        
        self.log_info("=== RAW RESPONSE ===")
        if self.logger:
            for line in response.split('\n'):
                self.logger.info(f"  {line}")
        
        if parsed_result:
            self.log_info("")
            self.log_info("=== PARSED RESULT ===")
            self._log_object(parsed_result)
        
        self.log_separator()
        
        # Add to structured log
        self._add_to_json_log({
            "type": "response",
            "iteration": self.iteration,
            "step": self.step,
            "agent": agent_name,
            "raw_response": response,
            "parsed_result": self._serialize_object(parsed_result),
            "timestamp": datetime.now().isoformat()
        })
    
    def log_test_result(self, test_result: Any):
        """Log test execution result"""
        self.log_subheader("TEST EXECUTION RESULT")
        self._log_object(test_result)
        
        # Log full error output if available (for compile/run errors)
        if hasattr(test_result, 'raw_error_output') and test_result.raw_error_output:
            self.log_info("")
            self.log_info("=== FULL ERROR OUTPUT (Maven) ===")
            if self.logger:
                for line in test_result.raw_error_output.split('\n'):
                    self.logger.info(f"  {line}")
        
        # Add to structured log
        self._add_to_json_log({
            "type": "test_result",
            "iteration": self.iteration,
            "step": self.step,
            "result": self._serialize_object(test_result),
            "timestamp": datetime.now().isoformat()
        })
    
    def log_coverage_details(self, coverage_info: Any, annotated_code: str = None):
        """Log detailed coverage information"""
        self.log_subheader("COVERAGE DETAILS")
        self._log_object(coverage_info)
        
        if annotated_code:
            self.log_info("")
            self.log_info("=== ANNOTATED FOCAL METHOD CODE ===")
            if self.logger:
                for line in annotated_code.split('\n'):
                    self.logger.info(f"  {line}")
        
        # Add to structured log
        self._add_to_json_log({
            "type": "coverage_details",
            "iteration": self.iteration,
            "step": self.step,
            "coverage_info": self._serialize_object(coverage_info),
            "annotated_code": annotated_code,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_mutation_details(self, mutation_info: Any, annotated_code: str = None):
        """Log detailed mutation information"""
        self.log_subheader("MUTATION DETAILS")
        self._log_object(mutation_info)
        
        if annotated_code:
            self.log_info("")
            self.log_info("=== ANNOTATED FOCAL METHOD CODE ===")
            if self.logger:
                for line in annotated_code.split('\n'):
                    self.logger.info(f"  {line}")
        
        # Add to structured log
        self._add_to_json_log({
            "type": "mutation_details",
            "iteration": self.iteration,
            "step": self.step,
            "mutation_info": self._serialize_object(mutation_info),
            "annotated_code": annotated_code,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_test_code(self, code: str, label: str = "Test Code"):
        """Log test code"""
        self.log_subheader(label)
        if self.logger:
            for i, line in enumerate(code.split('\n'), 1):
                self.logger.info(f"  {i:4d} | {line}")
        
        # Add to structured log
        self._add_to_json_log({
            "type": "test_code",
            "iteration": self.iteration,
            "step": self.step,
            "label": label,
            "code": code,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_analysis_result(self, agent_name: str, analysis: Any):
        """Log analysis result from an agent"""
        self.log_subheader(f"ANALYSIS RESULT - {agent_name}")
        self._log_object(analysis)
        
        # Add to structured log
        self._add_to_json_log({
            "type": "analysis_result",
            "iteration": self.iteration,
            "step": self.step,
            "agent": agent_name,
            "analysis": self._serialize_object(analysis),
            "timestamp": datetime.now().isoformat()
        })
    
    def log_instructions(self, instructions: List[Any]):
        """Log update instructions"""
        self.log_subheader("UPDATE INSTRUCTIONS")
        for i, instr in enumerate(instructions, 1):
            self.log_info(f"Instruction {i}:")
            self._log_object(instr, indent=2)
        
        # Add to structured log
        self._add_to_json_log({
            "type": "instructions",
            "iteration": self.iteration,
            "step": self.step,
            "instructions": [self._serialize_object(i) for i in instructions],
            "timestamp": datetime.now().isoformat()
        })
    
    def log_iteration_summary(self, iteration: int, status: str, 
                              line_coverage: float = None, branch_coverage: float = None, 
                              coverage: float = None, mutation: float = 0.0, score: float = 0.0):
        """
        Log iteration summary
        
        Args:
            iteration: Iteration number
            status: Test result status
            line_coverage: Line coverage percentage (new parameter)
            branch_coverage: Branch coverage percentage or None if no branches (new parameter)
            coverage: Overall coverage percentage (deprecated, kept for backward compatibility)
            mutation: Mutation kill percentage
            score: Overall score
        """
        self.log_subheader(f"ITERATION {iteration} SUMMARY")
        self.log_info(f"Status: {status}")
        
        # 使用新的分离的覆盖率参数，如果没有提供则使用旧的 coverage 参数
        if line_coverage is not None:
            self.log_info(f"Line Coverage: {line_coverage:.2f}%")
            if branch_coverage is not None:
                self.log_info(f"Branch Coverage: {branch_coverage:.2f}%")
            else:
                self.log_info(f"Branch Coverage: None")
        elif coverage is not None:
            # 向后兼容：如果只提供了 coverage 参数
            self.log_info(f"Coverage: {coverage:.2f}%")
        
        self.log_info(f"Mutation Kill Rate: {mutation:.2f}%")
        self.log_info(f"Score: {score:.2f}")
        
        # Add to structured log
        self._add_to_json_log({
            "type": "iteration_summary",
            "iteration": iteration,
            "status": status,
            "line_coverage": line_coverage,
            "branch_coverage": branch_coverage,
            "coverage": coverage if coverage is not None else line_coverage,  # 向后兼容
            "mutation_kill_rate": mutation,
            "score": score,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_final_result(self, result: Any, best_iteration: int, 
                         final_code: str):
        """Log final framework result"""
        self.log_header("FINAL RESULT")
        self.log_info(f"Best Iteration: {best_iteration}")
        self._log_object(result)
        
        self.log_subheader("FINAL TEST CODE")
        if self.logger:
            for i, line in enumerate(final_code.split('\n'), 1):
                self.logger.info(f"  {i:4d} | {line}")
        
        self.log_info(f"Completed: {datetime.now().isoformat()}")
        
        # Add to structured log
        self._add_to_json_log({
            "type": "final_result",
            "best_iteration": best_iteration,
            "result": self._serialize_object(result),
            "final_code": final_code,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save JSON log
        self._save_json_log()
    
    def log_custom(self, category: str, data: Any, description: str = ""):
        """Log custom data"""
        self.log_subheader(f"CUSTOM: {category}")
        if description:
            self.log_info(description)
        self._log_object(data)
        
        # Add to structured log
        self._add_to_json_log({
            "type": "custom",
            "iteration": self.iteration,
            "step": self.step,
            "category": category,
            "description": description,
            "data": self._serialize_object(data),
            "timestamp": datetime.now().isoformat()
        })
    
    def _log_object(self, obj: Any, indent: int = 0):
        """Log an object with proper formatting"""
        if obj is None:
            self.log_info("  " * indent + "None")
            return
        
        prefix = "  " * indent
        
        if is_dataclass(obj) and not isinstance(obj, type):
            try:
                obj_dict = asdict(obj)
                self._log_dict(obj_dict, indent)
            except Exception:
                self.log_info(f"{prefix}{str(obj)}")
        elif isinstance(obj, dict):
            self._log_dict(obj, indent)
        elif isinstance(obj, (list, tuple)):
            for i, item in enumerate(obj):
                self.log_info(f"{prefix}[{i}]:")
                self._log_object(item, indent + 1)
        else:
            self.log_info(f"{prefix}{str(obj)}")
    
    def _log_dict(self, d: dict, indent: int = 0):
        """Log a dictionary"""
        prefix = "  " * indent
        
        # 定义需要隐藏的字段（为了避免冗余信息）
        # coverage_percentage 是冗余的，因为我们已经有 line_coverage_percentage 和 branch_coverage_percentage
        # coverage 也是冗余的（最终结果字典中），我们使用 line_coverage 和 branch_coverage
        hidden_fields = {'coverage_percentage', 'coverage'}
        
        for key, value in d.items():
            # 跳过隐藏字段
            if key in hidden_fields:
                continue
                
            if isinstance(value, (dict, list)) or (is_dataclass(value) and not isinstance(value, type)):
                self.log_info(f"{prefix}{key}:")
                self._log_object(value, indent + 1)
            elif isinstance(value, str) and '\n' in value:
                self.log_info(f"{prefix}{key}:")
                for line in value.split('\n'):
                    self.log_info(f"{prefix}  {line}")
            else:
                # Truncate very long values
                value_str = str(value)
                if len(value_str) > 200:
                    value_str = value_str[:200] + "..."
                self.log_info(f"{prefix}{key}: {value_str}")
    
    def _serialize_object(self, obj: Any) -> Any:
        """Serialize an object to JSON-compatible format"""
        if obj is None:
            return None
        
        if is_dataclass(obj) and not isinstance(obj, type):
            try:
                return asdict(obj)
            except Exception:
                return str(obj)
        elif isinstance(obj, dict):
            return {k: self._serialize_object(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._serialize_object(item) for item in obj]
        elif isinstance(obj, (int, float, str, bool)):
            return obj
        elif hasattr(obj, '__dict__'):
            return {k: self._serialize_object(v) for k, v in obj.__dict__.items()}
        else:
            return str(obj)
    
    def _add_to_json_log(self, entry: Dict):
        """Add entry to JSON log buffer"""
        self._log_buffer.append(entry)
    
    def _save_json_log(self):
        """Save JSON log to file"""
        if self._json_log_path and self._log_buffer:
            try:
                with open(self._json_log_path, 'w', encoding='utf-8') as f:
                    json.dump({
                        "sample_id": self.sample_id,
                        "sample_name": self.sample_name,
                        "log_file": str(self.log_file),
                        "entries": self._log_buffer
                    }, f, indent=2, ensure_ascii=False)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Failed to save JSON log: {e}")
    
    def close(self):
        """Close and save the log"""
        self._save_json_log()
        
        if self.file_handler:
            self.file_handler.close()
            self.logger.removeHandler(self.file_handler)
        
        self.log_file = None
        self.file_handler = None


# Global function to get logger instance
def get_sample_logger() -> SampleLogger:
    """Get the global sample logger instance"""
    return SampleLogger.get_instance()

