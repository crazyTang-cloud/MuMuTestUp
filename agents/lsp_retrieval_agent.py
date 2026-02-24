"""
LSP Retrieval Agent for using Language Server Protocol to retrieve code information.

This agent uses LSP (Language Server Protocol) to retrieve method and class definitions
by leveraging the language server's "Go to Definition" capability.
"""
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
import sys

from agents.base_agent import BaseAgent
from models import (
    RetrievalResult, RetrievedMethod, RetrievedField,
    DiffHunk, FocalMethodInfo
)


class LSPRetrievalAgent(BaseAgent):
    """Agent for retrieving code using LSP (Language Server Protocol)"""
    
    def __init__(self, repo_path: str, project_name: str, commit_id: str):
        """
        Initialize LSP retrieval agent.
        
        Args:
            repo_path: Path to the repository
            project_name: Name of the project (e.g., "apache/druid")
            commit_id: Commit ID to work with (aCommit)
        """
        super().__init__("LSPRetrievalAgent", "lsp_retrieval")
        
        self.repo_path = Path(repo_path).resolve()
        self.project_name = project_name
        self.commit_id = commit_id
        self.lsp = None
        self.lsp_initialized = False
        
        self.log_info(f"LSPRetrievalAgent initialized for {project_name}@{commit_id}")
        self.log_info(f"Repository path: {self.repo_path}")
    
    def execute(self, *args, **kwargs):
        """
        Execute method required by BaseAgent abstract class.
        LSPRetrievalAgent provides specific retrieval methods instead of a generic execute.
        """
        raise NotImplementedError(
            "LSPRetrievalAgent does not use generic execute(). "
            "Use specific methods like retrieve_symbols() or retrieve_for_error_analysis()"
        )
    
    def _ensure_lsp_initialized(self) -> bool:
        """
        Ensure LSP server is initialized and running.
        
        Returns:
            True if LSP is ready, False otherwise
        """
        if self.lsp_initialized and self.lsp:
            return True
        
        try:
            # Import multilspy from local utils directory
            from utils.multilspy import SyncLanguageServer
            from utils.multilspy.multilspy_config import MultilspyConfig
            from utils.multilspy.multilspy_logger import MultilspyLogger
            
            # Initialize LSP config for Java
            lsp_config = MultilspyConfig.from_dict({
                "code_language": "java",
                "trace_lsp_communication": False  # Set to True for debugging
            })
            lsp_logger = MultilspyLogger()
            
            # Create LSP server
            self.lsp = SyncLanguageServer.create(lsp_config, lsp_logger, str(self.repo_path))
            
            self.log_info("LSP server initialized successfully")
            self.lsp_initialized = True
            return True
            
        except Exception as e:
            self.log_error(f"Failed to initialize LSP server: {e}")
            import traceback
            self.log_error(traceback.format_exc())
            return False
    
    def start_lsp_server(self):
        """
        Start the LSP server context.
        Should be used with 'with' statement.
        
        Returns:
            LSP server context manager
        """
        if not self._ensure_lsp_initialized():
            raise RuntimeError("Failed to initialize LSP server")
        
        return self.lsp.start_server()
    
    def retrieve_symbols(
        self,
        symbols: List[str],
        context_file: str,
        context_code: str,
        test_file: Optional[str] = None
    ) -> RetrievalResult:
        """
        Retrieve definitions for a list of symbols using LSP.
        
        Args:
            symbols: List of symbol names to retrieve (e.g., ["CustomValidator", "save"])
            context_file: Relative path to the file where symbols are used (usually focal method file)
            context_code: Code snippet containing the symbols
            test_file: Optional test file path to search if symbol not found in context_file
            
        Returns:
            RetrievalResult with retrieved information
        """
        if not self._ensure_lsp_initialized():
            return RetrievalResult(
                retrieval_successful=False,
                retrieval_reasoning="Failed to initialize LSP server"
            )
        
        self.log_info(f"Retrieving {len(symbols)} symbol(s) using LSP: {', '.join(symbols)}")
        
        retrieved_methods = []
        retrieved_fields = []
        failed_symbols = []
        
        try:
            # Read the context file
            context_file_path = self.repo_path / context_file
            if not context_file_path.exists():
                self.log_error(f"Context file not found: {context_file}")
                return RetrievalResult(
                    retrieval_successful=False,
                    retrieval_reasoning=f"Context file not found: {context_file}",
                    failed_symbols=symbols
                )
            
            file_content = context_file_path.read_text(encoding='utf-8')
            
            # For each symbol, find its position and request definition
            for symbol in symbols:
                try:
                    symbol_name = symbol.rstrip('()')  # Remove () if it's a method
                    is_method = '(' in symbol
                    
                    # Try to find and retrieve symbol from context_file
                    definition_loc = self._find_symbol_and_request_definition(
                        symbol_name, context_file, file_content, context_code
                    )
                    
                    # If not found and test_file is provided, try test_file
                    if (not definition_loc or len(definition_loc) == 0) and test_file:
                        self.log_info(f"Symbol '{symbol_name}' not found in {context_file}, trying {test_file}")
                        test_file_path = self.repo_path / test_file
                        if test_file_path.exists():
                            test_file_content = test_file_path.read_text(encoding='utf-8')
                            definition_loc = self._find_symbol_and_request_definition(
                                symbol_name, test_file, test_file_content, context_code
                            )
                    
                    if not definition_loc or len(definition_loc) == 0:
                        self.log_warning(f"No definition found for symbol '{symbol_name}'")
                        failed_symbols.append(symbol)
                        continue
                    
                    # Extract definition information
                    for loc in definition_loc:
                        rel_path = loc.get("relativePath", "")
                        start_line = loc.get("range", {}).get("start", {}).get("line", 0)
                        
                        # Read the definition file
                        def_file_path = self.repo_path / rel_path
                        if not def_file_path.exists():
                            continue
                        
                        def_content = def_file_path.read_text(encoding='utf-8')
                        
                        # Extract method or class definition
                        if is_method:
                            method_info = self._extract_method_from_line(def_content, start_line)
                            if method_info:
                                # Parse method info
                                class_name = self._extract_class_name(def_content, start_line)
                                signature = self._extract_method_signature(method_info)
                                
                                retrieved_methods.append(RetrievedMethod(
                                    class_name=class_name or "Unknown",
                                    method_name=symbol_name,
                                    signature=signature,
                                    body=method_info,
                                    javadoc=self._extract_javadoc(def_content, start_line),
                                    file_path=rel_path,
                                    relevance_score=1.0
                                ))
                                self.log_info(f"Retrieved method: {class_name}.{symbol_name}")
                        else:
                            # Try to get class definition
                            class_info = self._extract_class_from_line(def_content, start_line)
                            if class_info:
                                class_name = self._extract_class_name_from_definition(class_info)
                                
                                retrieved_methods.append(RetrievedMethod(
                                    class_name=class_name or symbol_name,
                                    method_name="<class_definition>",
                                    signature="",
                                    body=class_info,
                                    javadoc=self._extract_javadoc(def_content, start_line),
                                    file_path=rel_path,
                                    relevance_score=1.0
                                ))
                                self.log_info(f"Retrieved class: {class_name}")
                            else:
                                # Might be a field
                                field_info = self._extract_field_from_line(def_content, start_line)
                                if field_info:
                                    class_name = self._extract_class_name(def_content, start_line)
                                    field_type = self._extract_field_type(field_info)
                                    
                                    retrieved_fields.append(RetrievedField(
                                        class_name=class_name or "Unknown",
                                        field_name=symbol_name,
                                        field_type=field_type,
                                        value=None,
                                        javadoc=self._extract_javadoc(def_content, start_line),
                                        file_path=rel_path,
                                        relevance_score=1.0
                                    ))
                                    self.log_info(f"Retrieved field: {class_name}.{symbol_name}")
                
                except Exception as e:
                    self.log_error(f"Error retrieving symbol '{symbol}': {e}")
                    failed_symbols.append(symbol)
            
            # Build result
            success = len(retrieved_methods) > 0 or len(retrieved_fields) > 0
            
            return RetrievalResult(
                retrieved_methods=retrieved_methods,
                retrieved_fields=retrieved_fields,
                retrieval_successful=success,
                retrieval_reasoning=f"LSP retrieved {len(retrieved_methods)} methods and {len(retrieved_fields)} fields",
                failed_symbols=failed_symbols
            )
        
        except Exception as e:
            self.log_error(f"LSP retrieval failed: {e}")
            import traceback
            self.log_error(traceback.format_exc())
            return RetrievalResult(
                retrieval_successful=False,
                retrieval_reasoning=f"LSP retrieval error: {str(e)}",
                failed_symbols=symbols
            )
    
    def _find_symbol_and_request_definition(
        self,
        symbol_name: str,
        file_path: str,
        file_content: str,
        context_code: str
    ) -> list:
        """
        Find a symbol in a file and request its definition from LSP.
        
        Args:
            symbol_name: Name of the symbol to find
            file_path: Relative path to the file
            file_content: Content of the file
            context_code: Code snippet that might contain the symbol
            
        Returns:
            List of definition locations from LSP, or empty list if not found
        """
        pattern = rf'\b{re.escape(symbol_name)}\b'
        
        # First, try to find symbol in context_code (more precise)
        context_match = re.search(pattern, context_code)
        
        if context_match:
            # Find the position in the full file content
            # Search for context_code in file_content to get offset
            context_in_file = file_content.find(context_code[:100])  # Use first 100 chars as anchor
            if context_in_file != -1:
                position = context_in_file + context_match.start()
            else:
                # Fallback: search in entire file
                match = re.search(pattern, file_content)
                if not match:
                    self.log_warning(f"Symbol '{symbol_name}' not found in {file_path}")
                    return []
                position = match.start()
        else:
            # Fallback: search in entire file (but skip import statements)
            # Find all matches and skip those in import lines
            all_matches = list(re.finditer(pattern, file_content))
            valid_match = None
            for m in all_matches:
                # Get the line containing this match
                line_start = file_content.rfind('\n', 0, m.start()) + 1
                line_end = file_content.find('\n', m.start())
                if line_end == -1:
                    line_end = len(file_content)
                line_content = file_content[line_start:line_end].strip()
                
                # Skip if it's an import statement
                if not line_content.startswith('import '):
                    valid_match = m
                    break
            
            if not valid_match:
                self.log_warning(f"Symbol '{symbol_name}' not found in {file_path} (excluding imports)")
                return []
            
            position = valid_match.start()
        
        # Convert position to line and column
        line_num, col_num = self._get_line_col_from_index(file_content, position)
        
        self.log_info(f"Found symbol '{symbol_name}' at {file_path}:{line_num}:{col_num}")
        
        # Request definition from LSP
        try:
            definition_loc = self.lsp.request_definition(file_path, line_num, col_num)
            return definition_loc if definition_loc else []
        except Exception as e:
            self.log_error(f"LSP request_definition failed for '{symbol_name}' at {file_path}:{line_num}:{col_num}: {e}")
            return []
    
    def _get_line_col_from_index(self, content: str, index: int) -> tuple:
        """Convert string index to line and column numbers (1-indexed)"""
        lines = content[:index].split('\n')
        line_num = len(lines)
        col_num = len(lines[-1]) if lines else 0
        return line_num, col_num
    
    def _extract_method_from_line(self, content: str, line_num: int) -> Optional[str]:
        """Extract complete method definition starting from a line"""
        lines = content.split('\n')
        if line_num >= len(lines):
            return None
        
        # Find method start (might have annotations before)
        start_idx = line_num
        while start_idx > 0 and (lines[start_idx - 1].strip().startswith('@') or 
                                  not lines[start_idx - 1].strip()):
            start_idx -= 1
        
        # Find method end (matching braces)
        brace_count = 0
        end_idx = line_num
        in_method = False
        
        for i in range(line_num, len(lines)):
            line = lines[i]
            for char in line:
                if char == '{':
                    brace_count += 1
                    in_method = True
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and in_method:
                        end_idx = i
                        return '\n'.join(lines[start_idx:end_idx + 1])
        
        return None
    
    def _extract_class_from_line(self, content: str, line_num: int) -> Optional[str]:
        """Extract complete class definition starting from a line"""
        lines = content.split('\n')
        if line_num >= len(lines):
            return None
        
        # Find class declaration
        start_idx = line_num
        while start_idx > 0 and not re.search(r'\b(class|interface|enum)\b', lines[start_idx]):
            start_idx -= 1
        
        # Include annotations and modifiers before class
        while start_idx > 0 and (lines[start_idx - 1].strip().startswith('@') or
                                  re.search(r'\b(public|private|protected|static|final|abstract)\b', 
                                           lines[start_idx - 1])):
            start_idx -= 1
        
        # Find class end (matching braces)
        brace_count = 0
        end_idx = start_idx
        in_class = False
        
        for i in range(start_idx, len(lines)):
            line = lines[i]
            for char in line:
                if char == '{':
                    brace_count += 1
                    in_class = True
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0 and in_class:
                        end_idx = i
                        # Limit class size (return first 100 lines max)
                        max_lines = min(end_idx + 1, start_idx + 100)
                        return '\n'.join(lines[start_idx:max_lines])
        
        return None
    
    def _extract_field_from_line(self, content: str, line_num: int) -> Optional[str]:
        """Extract field definition from a line"""
        lines = content.split('\n')
        if line_num >= len(lines):
            return None
        
        # Get the field line and possibly previous lines (for annotations)
        start_idx = line_num
        while start_idx > 0 and lines[start_idx - 1].strip().startswith('@'):
            start_idx -= 1
        
        # Field definition is usually one line (or until semicolon)
        field_lines = []
        for i in range(start_idx, min(line_num + 5, len(lines))):
            field_lines.append(lines[i])
            if ';' in lines[i]:
                break
        
        return '\n'.join(field_lines)
    
    def _extract_class_name(self, content: str, line_num: int) -> Optional[str]:
        """Extract the class name that contains the given line"""
        lines = content.split('\n')
        
        # Search backwards for class declaration
        for i in range(line_num, -1, -1):
            match = re.search(r'\b(?:class|interface|enum)\s+(\w+)', lines[i])
            if match:
                return match.group(1)
        
        return None
    
    def _extract_class_name_from_definition(self, class_def: str) -> Optional[str]:
        """Extract class name from class definition"""
        match = re.search(r'\b(?:class|interface|enum)\s+(\w+)', class_def)
        return match.group(1) if match else None
    
    def _extract_method_signature(self, method_def: str) -> str:
        """Extract method signature from method definition"""
        # Find the line with method declaration (has parentheses)
        for line in method_def.split('\n'):
            if '(' in line and ')' in line and not line.strip().startswith('//'):
                # Extract from method name to closing paren
                match = re.search(r'(\w+\s*\([^)]*\))', line)
                if match:
                    return match.group(1)
        return ""
    
    def _extract_field_type(self, field_def: str) -> str:
        """Extract field type from field definition"""
        # Pattern: [modifiers] Type fieldName [= value];
        match = re.search(r'\b(?:public|private|protected|static|final)\s+(.+?)\s+\w+\s*[;=]', field_def)
        if match:
            return match.group(1).strip()
        return "Unknown"
    
    def _extract_javadoc(self, content: str, line_num: int) -> Optional[str]:
        """Extract Javadoc comment before a line"""
        lines = content.split('\n')
        
        # Search backwards for Javadoc
        javadoc_lines = []
        i = line_num - 1
        
        # Skip empty lines and annotations
        while i >= 0 and (not lines[i].strip() or lines[i].strip().startswith('@')):
            i -= 1
        
        # Check if there's a Javadoc comment
        if i >= 0 and lines[i].strip().endswith('*/'):
            # Found end of Javadoc, collect backwards
            javadoc_lines.insert(0, lines[i])
            i -= 1
            while i >= 0:
                javadoc_lines.insert(0, lines[i])
                if lines[i].strip().startswith('/**'):
                    return '\n'.join(javadoc_lines)
                i -= 1
        
        return None
    
    def close(self):
        """Close LSP server"""
        if self.lsp:
            try:
                # LSP server cleanup is handled by context manager
                self.log_info("LSP server closed")
            except Exception as e:
                self.log_error(f"Error closing LSP server: {e}")

