"""
Retrieval Agent for integrating the index_retrieve_module_v3 system.

This agent encapsulates the retrieval system and provides a unified interface
for other agents to retrieve methods and fields from the codebase.
"""
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import os
import sys
import re

# Add index_retrieve_module_v3 to path
sys.path.insert(0, str(Path(__file__).parent.parent / "index_retrieve_module_v3"))

from agents.base_agent import BaseAgent
from models import (
    RetrievalResult, RetrievedMethod, RetrievedField,
    DiffHunk, FocalMethodInfo, TestCase
)

# Import retrieval system components
from indexer.config import IndexerConfig
from indexer.storage import SQLiteStorage
from indexer.vector_store import VectorStore
from pipeline.index_builder import PipelineIndexBuilder
from pipeline.agent import LLMAgent


class RetrievalAgent(BaseAgent):
    """Agent for retrieving methods and fields from the codebase"""
    
    def __init__(self, repo_path: str, project_name: str, commit_id: str):
        """
        Initialize retrieval agent.
        
        Args:
            repo_path: Path to the repository
            project_name: Name of the project (e.g., "apache/druid")
            commit_id: Commit ID to index (aCommit)
        """
        super().__init__("RetrievalAgent", "retrieval")
        
        self.repo_path = Path(repo_path).resolve()
        self.project_name = project_name
        self.commit_id = commit_id
        
        # Check if we should use LSP
        from config import config
        self.use_lsp = config.framework.ablation_use_lsp
        self.lsp_server_context = None  # Track LSP server context
        
        if self.use_lsp:
            # Initialize LSP retrieval agent
            from agents.lsp_retrieval_agent import LSPRetrievalAgent
            self.lsp_agent = LSPRetrievalAgent(
                repo_path=str(self.repo_path),
                project_name=project_name,
                commit_id=commit_id
            )
            self.log_info(f"RetrievalAgent initialized with LSP for {project_name}@{commit_id}")
        else:
            # Setup index directory: index_data/{project_name}/{commit_id}/
            self.index_dir = self._get_index_dir()
            
            # Initialize index builder
            self.index_builder = PipelineIndexBuilder(
                repo_path=str(self.repo_path),
                output_dir=self.index_dir
            )
            
            # Initialize LLM agent for query generation
            # Use config.llm settings instead of environment variables
            api_key = config.llm.api_key or os.getenv("OPENAI_API_KEY", "")
            # Use get_base_url() to properly handle API URL configuration
            base_url = config.llm.get_base_url()
            # Ensure base_url ends with /v1 for OpenAI-compatible APIs
            if not base_url.endswith('/v1'):
                base_url = base_url.rstrip('/') + '/v1'
            
            self.llm_agent = LLMAgent(
                model=config.llm.model,
                api_key=api_key,
                base_url=base_url,
                temperature=config.llm.temperature,
                max_retries=config.llm.max_retries,
                timeout=config.llm.timeout
            )
            
            self.log_info(f"RetrievalAgent initialized for {project_name}@{commit_id}")
            self.log_info(f"Index directory: {self.index_dir}")
    
    def execute(self, *args, **kwargs):
        """
        Execute method required by BaseAgent abstract class.
        RetrievalAgent provides specific retrieval methods instead of a generic execute.
        """
        raise NotImplementedError(
            "RetrievalAgent does not use generic execute(). "
            "Use specific methods like retrieve_for_error_analysis() or retrieve_for_root_cause_analysis()"
        )
    
    def start_lsp_server(self):
        """
        Start LSP server if using LSP ablation.
        Should be called at the beginning of the update process.
        """
        if self.use_lsp and self.lsp_agent and not self.lsp_server_context:
            self.log_info("Starting LSP server...")
            try:
                self.lsp_server_context = self.lsp_agent.start_lsp_server()
                self.lsp_server_context.__enter__()
                self.log_info("LSP server started successfully")
            except Exception as e:
                self.log_error(f"Failed to start LSP server: {e}")
                import traceback
                self.log_error(traceback.format_exc())
                self.lsp_server_context = None
    
    def stop_lsp_server(self):
        """
        Stop LSP server if it was started.
        Should be called at the end of the update process.
        """
        if self.lsp_server_context:
            self.log_info("Stopping LSP server...")
            try:
                self.lsp_server_context.__exit__(None, None, None)
                self.log_info("LSP server stopped successfully")
            except Exception as e:
                self.log_error(f"Error stopping LSP server: {e}")
            finally:
                self.lsp_server_context = None
    
    def _get_index_dir(self) -> Path:
        """Get index directory path for this project and commit"""
        # Sanitize project name for filesystem
        safe_project_name = self.project_name.replace("/", "_").replace("\\", "_")
        
        # Create path: index_data/{project_name}/{commit_id}/
        base_dir = Path(__file__).parent.parent / "index_data"
        index_dir = base_dir / safe_project_name / self.commit_id
        
        return index_dir
    
    def ensure_index_built(self, force_rebuild: bool = False) -> bool:
        """
        Ensure SQLite index is built for this project and commit.
        
        Args:
            force_rebuild: Force rebuild even if index exists
            
        Returns:
            True if successful, False otherwise
        """
        self.log_info("Ensuring SQLite index is built...")
        
        # Check if index already exists
        if not force_rebuild and self.index_builder.check_index_exists(require_vectors=False):
            self.log_info("SQLite index already exists, skipping rebuild")
            return True
        
        # Build lightweight index (SQLite only, no vectors)
        self.log_info("Building SQLite index...")
        success = self.index_builder.build_lightweight_index(
            skip_rebuild=False
        )
        
        if success:
            self.log_info("SQLite index built successfully")
        else:
            self.log_error("Failed to build SQLite index")
        
        return success
    
    def retrieve_for_root_cause_analysis(
        self,
        test_code: str,
        focal_method_info: FocalMethodInfo,
        filtered_hunks: List[DiffHunk],
        focal_method_changed: bool
    ) -> RetrievalResult:
        """
        Retrieve information for root cause analysis.
        
        Args:
            test_code: The test code
            focal_method_info: Focal method information
            filtered_hunks: Filtered relevant hunks
            focal_method_changed: Whether focal method changed
            
        Returns:
            RetrievalResult with retrieved information
        """
        self.log_info("Retrieving for root cause analysis...")
        
        # Check if using LSP
        from config import config
        
        if config.framework.ablation_use_lsp:
            # Use LSP for retrieval
            self.log_info("Ablation: Using LSP for retrieval")
            return self._try_lsp_retrieval_root_cause(
                test_code, focal_method_info, filtered_hunks, focal_method_changed
            )
        
        # Build context for LLM
        context = self._build_root_cause_context(
            test_code, focal_method_info, filtered_hunks, focal_method_changed
        )
        
        # Determine max_rounds based on ablation settings
        if config.framework.ablation_sqlite_single_round:
            # Single-round SQLite only
            self.log_info("Ablation: SQLite single-round only (no iteration, no ChromaDB)")
            retrieval_result = self._try_sql_retrieval(context, max_rounds=1)
        else:
            # Normal behavior: 3 rounds SQLite, then ChromaDB if needed
            retrieval_result = self._try_sql_retrieval(context, max_rounds=3)
            
            # If SQL retrieval insufficient, try RAG retrieval (Phase 3)
            if not retrieval_result.retrieval_successful:
                self.log_info("SQL retrieval insufficient, trying RAG retrieval...")
                retrieval_result = self._try_rag_retrieval(context, max_rounds=3, top_k=8)
        
        return retrieval_result
    
    def retrieve_for_error_analysis(
        self,
        test_code: str,
        focal_method_info: FocalMethodInfo,
        filtered_hunks: List[DiffHunk],
        error_message: str,
        raw_error_output: Optional[str] = None
    ) -> RetrievalResult:
        """
        Retrieve information for error analysis.
        
        Args:
            test_code: The updated test code
            focal_method_info: Focal method information
            filtered_hunks: Filtered relevant hunks
            error_message: Error message
            raw_error_output: Raw error output
            
        Returns:
            RetrievalResult with retrieved information
        """
        self.log_info("Retrieving for error analysis...")
        
        # Check if using LSP
        from config import config
        
        if config.framework.ablation_use_lsp:
            # Use LSP for retrieval
            self.log_info("Ablation: Using LSP for retrieval")
            return self._try_lsp_retrieval_error(
                test_code, focal_method_info, filtered_hunks, error_message, raw_error_output
            )
        
        # Build context for LLM
        context = self._build_error_analysis_context(
            test_code, focal_method_info, filtered_hunks, error_message, raw_error_output
        )
        
        # Determine max_rounds based on ablation settings
        if config.framework.ablation_sqlite_single_round:
            # Single-round SQLite only
            self.log_info("Ablation: SQLite single-round only (no iteration, no ChromaDB)")
            retrieval_result = self._try_sql_retrieval(context, max_rounds=1)
        elif config.framework.ablation_sql_only_3rounds:
            # Only SQL retrieval with 3 rounds
            self.log_info("Ablation: SQL-only retrieval with 3 rounds (no RAG)")
            retrieval_result = self._try_sql_retrieval(context, max_rounds=3)
        elif config.framework.ablation_rag_only_3rounds:
            # Only RAG retrieval with 3 rounds
            self.log_info("Ablation: RAG-only retrieval with 3 rounds (no SQL)")
            retrieval_result = self._try_rag_retrieval(context, max_rounds=3, top_k=3)
        elif config.framework.ablation_sql_1round_rag_3rounds:
            # 1 round SQL, then 3 rounds RAG (always execute both)
            self.log_info("Ablation: 1 round SQL retrieval, then 3 rounds RAG retrieval")
            retrieval_result = self._try_sql_retrieval(context, max_rounds=1)
            # Always try RAG retrieval after SQL (regardless of SQL success)
            self.log_info("Proceeding to RAG retrieval after SQL retrieval...")
            retrieval_result = self._try_rag_retrieval(context, max_rounds=3, top_k=3)
        else:
            # Normal behavior: 3 rounds SQLite, then ChromaDB if needed
            retrieval_result = self._try_sql_retrieval(context, max_rounds=3)
            
            # If SQL retrieval insufficient, try RAG retrieval
            if not retrieval_result.retrieval_successful:
                self.log_info("SQL retrieval insufficient, trying RAG retrieval...")
                retrieval_result = self._try_rag_retrieval(context, max_rounds=3, top_k=3)
        
        return retrieval_result
    
    def retrieve_for_unknown_symbols(
        self,
        symbols: List[str],
        error_message: str,
        test_code: str,
        focal_method_info: FocalMethodInfo,
        filtered_hunks: List[DiffHunk],
        error_type: str
    ) -> RetrievalResult:
        """
        Retrieve information for unknown symbols with multi-round exploration.
        
        Args:
            symbols: List of unknown symbols to retrieve
            error_message: Error message
            test_code: Test code
            focal_method_info: Focal method info
            filtered_hunks: Filtered hunks
            error_type: Error type (for context)
            
        Returns:
            RetrievalResult with retrieved_items grouped by symbol
        """
        self.log_info(f"Retrieving for {len(symbols)} unknown symbol(s): {', '.join(symbols)}")
        
        # Check if using LSP
        from config import config
        
        if config.framework.ablation_use_lsp:
            # Use LSP for retrieval
            self.log_info("Ablation: Using LSP for unknown symbols retrieval")
            return self._try_lsp_retrieval_unknown_symbols(
                symbols, error_message, test_code, focal_method_info, filtered_hunks
            )
        
        # Build context
        context = self._build_symbol_retrieval_context(
            symbols, error_message, test_code, focal_method_info, filtered_hunks, error_type
        )
        
        # Round 1: Initial precise queries
        round_1_response = self._llm_generate_initial_sql(context)
        
        if not round_1_response:
            return RetrievalResult(
                retrieval_successful=False,
                retrieval_reasoning='Failed to generate initial SQL query',
                retrieved_items={},
                failed_symbols=list(symbols)
            )
        
        if not round_1_response.get('needs_retrieval'):
            return RetrievalResult(
                retrieval_successful=False,
                retrieval_reasoning=round_1_response.get('reasoning', 'LLM determined retrieval not needed'),
                retrieved_items={},
                failed_symbols=[]
            )
        
        # Execute Round 1 queries
        retrieved_items = {}
        pending_symbols = set(symbols)
        all_failed_queries = []
        
        for query_info in round_1_response.get('sql_queries', []):
            symbol = query_info.get('symbol')
            sql = query_info.get('sql')
            
            if not symbol or not sql:
                continue
            
            self.log_info(f"[Round 1] Querying {symbol}: {sql[:80]}...")
            results = self._execute_sql_query(sql)
            
            if results:
                retrieved_items[symbol] = results
                pending_symbols.discard(symbol)
                self.log_info(f"[Round 1] Found {len(results)} result(s) for {symbol}")
            else:
                all_failed_queries.append({
                    'symbol': symbol,
                    'sql': sql,
                    'round': 1,
                    'result': 'no matches'
                })
                self.log_info(f"[Round 1] No results for {symbol}")
        
        # Multi-round exploration for pending symbols
        max_rounds = 3
        for round_num in range(2, max_rounds + 1):
            if not pending_symbols:
                break  # All symbols found
            
            self.log_info(f"[Round {round_num}] Exploring {len(pending_symbols)} pending symbol(s)...")
            
            # Generate exploratory queries
            exploratory_response = self._llm_generate_exploratory_sql(
                context, pending_symbols, all_failed_queries, round_num
            )
            
            if not exploratory_response or not exploratory_response.get('exploratory_queries'):
                self.log_info(f"[Round {round_num}] No more queries generated")
                break
            
            # Execute exploratory queries
            for query_info in exploratory_response.get('exploratory_queries', []):
                symbol = query_info.get('original_symbol')
                sql = query_info.get('sql')
                reasoning = query_info.get('reasoning', '')
                
                if not symbol or not sql:
                    continue
                
                self.log_info(f"[Round {round_num}] Exploring {symbol}: {reasoning}")
                results = self._execute_sql_query(sql)
                
                if results:
                    if symbol not in retrieved_items:
                        retrieved_items[symbol] = []
                    retrieved_items[symbol].extend(results)
                    pending_symbols.discard(symbol)
                    self.log_info(f"[Round {round_num}] Found {len(results)} result(s) for {symbol}")
                else:
                    all_failed_queries.append({
                        'symbol': symbol,
                        'sql': sql,
                        'round': round_num,
                        'reasoning': reasoning,
                        'result': 'no matches'
                    })
        
        # Build result
        failed_symbols = list(pending_symbols)
        
        # If SQL retrieval didn't find all symbols, try RAG retrieval as fallback
        # (unless ablation_sqlite_single_round is enabled)
        from config import config
        if failed_symbols and not config.framework.ablation_sqlite_single_round:
            self.log_info(f"SQL retrieval incomplete for {len(failed_symbols)} symbol(s), trying RAG retrieval...")
            rag_result = self._try_rag_retrieval(context, max_rounds=3, top_k=8)
            
            if rag_result.retrieval_successful:
                # Merge RAG results with SQL results
                # Note: RAG returns methods/fields, need to group them appropriately
                if rag_result.retrieved_methods:
                    for method in rag_result.retrieved_methods:
                        # Try to match methods to failed symbols
                        for symbol in failed_symbols[:]:  # Copy to allow removal during iteration
                            if symbol in method.class_name or symbol in method.method_name:
                                if symbol not in retrieved_items:
                                    retrieved_items[symbol] = []
                                retrieved_items[symbol].append({
                                    'class_name': method.class_name,
                                    'method_name': method.method_name,
                                    'signature': method.signature,
                                    'body': method.body,
                                    'javadoc': method.javadoc,
                                    'file_path': method.file_path
                                })
                                if symbol in failed_symbols:
                                    failed_symbols.remove(symbol)
                
                if rag_result.retrieved_fields:
                    for field in rag_result.retrieved_fields:
                        # Try to match fields to failed symbols
                        for symbol in failed_symbols[:]:
                            if symbol in field.class_name or symbol in field.field_name:
                                if symbol not in retrieved_items:
                                    retrieved_items[symbol] = []
                                retrieved_items[symbol].append({
                                    'class_name': field.class_name,
                                    'field_name': field.field_name,
                                    'field_type': field.field_type,
                                    'javadoc': field.javadoc,
                                    'file_path': field.file_path
                                })
                                if symbol in failed_symbols:
                                    failed_symbols.remove(symbol)
                
                self.log_info(f"After RAG retrieval: {len(retrieved_items)}/{len(symbols)} symbols found")
        
        return RetrievalResult(
            retrieval_successful=len(retrieved_items) > 0,
            retrieval_reasoning=f"Retrieved {len(retrieved_items)}/{len(symbols)} symbols after {round_num} rounds (SQL + RAG)",
            retrieved_items=retrieved_items,
            failed_symbols=failed_symbols
        )
    
    def _build_symbol_retrieval_context(
        self,
        symbols: List[str],
        error_message: str,
        test_code: str,
        focal_method_info: FocalMethodInfo,
        filtered_hunks: List[DiffHunk],
        error_type: str
    ) -> str:
        """Build context for symbol retrieval"""
        
        hunks_text = self._format_hunks_for_context(filtered_hunks)
        
        return f"""
Symbols (preliminary analysis suggests these may be project-specific):
{', '.join(symbols)}

Note: These symbols have been preliminarily identified as potentially not from 
standard Java library or common test frameworks. They appear to be project-specific 
classes, methods, or fields. However, you should make the final judgment.

Error Type: {error_type}
Error Message: {error_message}

Test Code:
{test_code}

Focal Method:
{focal_method_info.current_code}

Code Changes:
{hunks_text}

Database schema:
- methods table columns: id, module_name, file_path, class_name, method_name, signature, javadoc, body, start_line, end_line, is_test, imports
  ⚠️ methods table does NOT have: field_name, field_type, modifiers, initializer
- fields table columns: id, module_name, file_path, class_name, field_name, field_type, modifiers, initializer, javadoc, start_line, end_line, is_test, imports
  ⚠️ fields table does NOT have: method_name, signature, body
"""
    
    def _llm_generate_initial_sql(self, context: str) -> Optional[Dict[str, Any]]:
        """Generate initial SQL queries using LLM"""
        
        prompt = f"""
{context}

YOUR TASK:
1. Determine if retrieval is needed for these symbols
2. If YES, generate SQL queries (SELECT only needed fields, not SELECT *)

⚠️ IMPORTANT:
- Be specific in WHERE clauses
- Use LIKE with wildcards for fuzzy matching
- Generate separate queries for each symbol

Response format (JSON):
{{
  "needs_retrieval": boolean,
  "reasoning": "why retrieval is/isn't needed",
  "sql_queries": [
    {{"symbol": "CustomValidator", "sql": "SELECT class_name, method_name, signature FROM methods WHERE class_name LIKE '%CustomValidator%'"}},
    {{"symbol": "Helper", "sql": "SELECT class_name, field_name, field_type FROM fields WHERE class_name LIKE '%Helper%'"}}
  ]
}}
"""
        
        try:
            response_text = self.llm_agent.ask(prompt)
            return self._parse_json_response(response_text)
        except Exception as e:
            self.log_error(f"Failed to generate initial SQL: {e}")
            return None
    
    def _llm_generate_exploratory_sql(
        self,
        context: str,
        pending_symbols: set,
        failed_queries: List[Dict],
        round_num: int
    ) -> Optional[Dict[str, Any]]:
        """Generate exploratory SQL queries for failed symbols"""
        
        failed_queries_text = '\n'.join([
            f"- Symbol: {q['symbol']}, SQL: {q['sql']}, Result: {q['result']}"
            for q in failed_queries
        ])
        
        prompt = f"""
{context}

Previous Retrieval Attempts (ALL FAILED QUERIES):
{failed_queries_text}

PENDING SYMBOLS (still need to find):
{', '.join(pending_symbols)}

⚠️ ANALYSIS: These symbols were not found. This might be because:
1. The symbol name is hallucinated (LLM-generated but doesn't exist)
2. The actual symbol has a similar but different name
3. The functionality exists but with different naming convention

Examples of common mismatches:
- Hallucinated: A.save() → Actual: store(A) or persist(A)
- Hallucinated: saveResult() → Actual: save() or storeResult()
- Hallucinated: CustomValidator → Actual: Validator or CustomValidation
- Hallucinated: getUserName() → Actual: getName() or getUser().getName()

YOUR TASK (Round {round_num}):
Generate exploratory SQL queries to find similar functionality.
⚠️ You can generate MULTIPLE queries per symbol to explore different possibilities.

Response format (JSON):
{{
  "exploratory_queries": [
    {{
      "original_symbol": "CustomValidator",
      "reasoning": "Trying class name variations",
      "sql": "SELECT class_name, method_name FROM methods WHERE class_name LIKE '%Validat%'"
    }},
    {{
      "original_symbol": "CustomValidator",
      "reasoning": "Looking in related packages",
      "sql": "SELECT class_name FROM methods WHERE class_name LIKE '%validation%' GROUP BY class_name"
    }}
  ]
}}
"""
        
        try:
            response_text = self.llm_agent.ask(prompt)
            return self._parse_json_response(response_text)
        except Exception as e:
            self.log_error(f"Failed to generate exploratory SQL: {e}")
            return None
    
    def _execute_sql_query(self, sql: str) -> List[Dict[str, Any]]:
        """Execute a single SQL query and return results"""
        try:
            sqlite_path = self.index_dir / "assets.db"
            if not sqlite_path.exists():
                self.log_error(f"SQLite database not found: {sqlite_path}")
                return []
            
            with SQLiteStorage(sqlite_path) as storage:
                cursor = storage.conn.cursor()
                cursor.execute(sql)
                
                columns = [desc[0] for desc in cursor.description]
                results = []
                
                for row in cursor.fetchall():
                    results.append(dict(zip(columns, row)))
                
                return results
        except Exception as e:
            self.log_error(f"SQL execution failed: {e}")
            return []
    
    def _parse_json_response(self, response_text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON from LLM response"""
        import json
        import re
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError as e:
                self.log_error(f"JSON parse error: {e}")
        
        return None
    
    def _build_root_cause_context(
        self,
        test_code: str,
        focal_method_info: FocalMethodInfo,
        filtered_hunks: List[DiffHunk],
        focal_method_changed: bool
    ) -> str:
        """Build context string for root cause analysis retrieval"""
        
        # Format focal method section
        if focal_method_changed:
            focal_section = f"""
**Focal Method (CHANGED):**

Before:
```java
{focal_method_info.original_code if focal_method_info.original_code else "N/A"}
```

After:
```java
{focal_method_info.current_code}
```
"""
        else:
            focal_section = f"""
**Focal Method (UNCHANGED):**
```java
{focal_method_info.current_code if focal_method_info.current_code else focal_method_info.original_code}
```
"""
        
        # Format hunks by type
        hunks_by_type = {}
        for hunk in filtered_hunks:
            hunk_type = hunk.hunk_type or "unknown"
            if hunk_type not in hunks_by_type:
                hunks_by_type[hunk_type] = []
            hunks_by_type[hunk_type].append(hunk)
        
        hunks_text = ""
        for hunk_type, hunks in hunks_by_type.items():
            hunks_text += f"\n**{hunk_type.upper()} Hunks:**\n"
            for i, hunk in enumerate(hunks, 1):
                hunks_text += f"\nHunk {i} - {hunk.file_path}:\n{hunk.context}\n"
        
        context = f"""**Test Code:**
```java
{test_code}
```

{focal_section}

**Relevant Code Changes:**
{hunks_text}

**Task:** Analyze the test and code changes to determine what additional information is needed to update the test correctly.
"""
        return context
    
    def _build_error_analysis_context(
        self,
        test_code: str,
        focal_method_info: FocalMethodInfo,
        filtered_hunks: List[DiffHunk],
        error_message: str,
        raw_error_output: Optional[str]
    ) -> str:
        """Build context string for error analysis retrieval"""
        
        # Format hunks
        hunks_text = ""
        for i, hunk in enumerate(filtered_hunks, 1):
            hunk_type = hunk.hunk_type or "unknown"
            hunks_text += f"\n[{hunk_type}] Hunk {i} - {hunk.file_path}:\n{hunk.context}\n"
        
        error_details = error_message
        if raw_error_output:
            error_details += f"\n\nDetailed Error:\n{raw_error_output[:1000]}"
        
        context = f"""**Test Code (with error):**
```java
{test_code}
```

**Focal Method:**
```java
{focal_method_info.current_code}
```

**Error Information:**
{error_details}

**Relevant Code Changes:**
{hunks_text}

**Task:** Analyze the error to determine what methods, fields, or classes are missing or need to be imported.
"""
        return context
    
    def _try_sql_retrieval(self, context: str, max_rounds: int = 3) -> RetrievalResult:
        """
        Try SQL-based retrieval (Phase 2 style).
        
        Args:
            context: Context string for LLM
            max_rounds: Maximum query rounds
            
        Returns:
            RetrievalResult
        """
        self.log_info("Attempting SQL retrieval...")
        
        sqlite_path = self.index_dir / "assets.db"
        if not sqlite_path.exists():
            self.log_warning("SQLite database not found")
            return RetrievalResult(
                retrieval_successful=False,
                retrieval_reasoning="SQLite database not found"
            )
        
        retrieved_methods = []
        retrieved_fields = []
        sql_queries = []
        
        previous_queries = []
        previous_results = []
        
        try:
            for round_num in range(1, max_rounds + 1):
                self.log_info(f"SQL retrieval round {round_num}/{max_rounds}")
                
                # Generate SQL query
                success, sql, reasoning = self.llm_agent.generate_sql_query(
                    error_context=context,
                    previous_queries=previous_queries,
                    previous_results=previous_results,
                    round_num=round_num
                )
                
                if not success:
                    self.log_warning(f"Failed to generate SQL in round {round_num}")
                    continue
                
                sql_queries.append(sql)
                
                # Validate SQL
                is_valid, validation_msg = self.llm_agent.validate_sql(sql)
                if not is_valid:
                    self.log_warning(f"SQL validation failed: {validation_msg}")
                    previous_queries.append(sql)
                    previous_results.append(f"VALIDATION ERROR: {validation_msg}")
                    continue
                
                # Execute SQL
                results = self._execute_sql(sqlite_path, sql)
                
                if not results:
                    previous_queries.append(sql)
                    previous_results.append("No results found")
                    continue
                
                self.log_info(f"Retrieved {len(results)} rows")
                
                # Parse results into methods/fields
                methods, fields = self._parse_sql_results(results)
                retrieved_methods.extend(methods)
                retrieved_fields.extend(fields)
                
                # Evaluate if sufficient
                decision, eval_reasoning, _ = self.llm_agent.evaluate_sql_results(
                    error_context=context,
                    sql_query=sql,
                    results=results,
                    round_num=round_num,
                    max_rounds=max_rounds
                )
                
                if decision == "SUFFICIENT":
                    self.log_info("SQL retrieval successful")
                    return RetrievalResult(
                        retrieved_methods=retrieved_methods,
                        retrieved_fields=retrieved_fields,
                        retrieval_successful=True,
                        retrieval_reasoning=eval_reasoning,
                        sql_queries_used=sql_queries
                    )
                
                previous_queries.append(sql)
                previous_results.append(f"Found {len(results)} rows. {eval_reasoning}")
            
            # Completed all rounds without sufficient results
            if retrieved_methods or retrieved_fields:
                return RetrievalResult(
                    retrieved_methods=retrieved_methods,
                    retrieved_fields=retrieved_fields,
                    retrieval_successful=False,
                    retrieval_reasoning="SQL retrieval incomplete, need RAG search",
                    sql_queries_used=sql_queries
                )
            else:
                return RetrievalResult(
                    retrieval_successful=False,
                    retrieval_reasoning="SQL retrieval found no results",
                    sql_queries_used=sql_queries
                )
        
        except Exception as e:
            self.log_error(f"SQL retrieval failed: {e}")
            return RetrievalResult(
                retrieval_successful=False,
                retrieval_reasoning=f"SQL retrieval error: {str(e)}",
                sql_queries_used=sql_queries
            )
    
    def _try_rag_retrieval(
        self,
        context: str,
        max_rounds: int = 3,
        top_k: int = 8
    ) -> RetrievalResult:
        """
        Try RAG-based retrieval (Phase 3 style).
        
        Args:
            context: Context string for LLM
            max_rounds: Maximum query rounds
            top_k: Number of results per query
            
        Returns:
            RetrievalResult
        """
        self.log_info("Attempting RAG retrieval...")
        
        # Ensure ChromaDB is initialized
        chroma_path = self.index_dir / "chroma_db"
        sqlite_path = self.index_dir / "assets.db"
        
        if not sqlite_path.exists():
            self.log_error("SQLite database not found, cannot do RAG retrieval")
            return RetrievalResult(
                retrieval_successful=False,
                retrieval_reasoning="SQLite database not found"
            )
        
        retrieved_methods = []
        retrieved_fields = []
        rag_queries = []
        
        try:
            # Get available modules
            available_modules = self.index_builder.get_available_modules()
            if not available_modules:
                self.log_warning("No modules found")
                return RetrievalResult(
                    retrieval_successful=False,
                    retrieval_reasoning="No modules found in repository"
                )
            
            embedded_modules = []
            previous_results_summary = None
            
            for round_num in range(1, max_rounds + 1):
                self.log_info(f"RAG retrieval round {round_num}/{max_rounds}")
                
                # Agent decides: embed module or refine query
                action, module_to_embed, action_reasoning = self.llm_agent.decide_rag_action(
                    error_context=context,
                    available_modules=available_modules,
                    embedded_modules=embedded_modules,
                    round_num=round_num,
                    previous_results=previous_results_summary
                )
                
                # Embed module if needed
                if action == "EMBED_MODULE" and module_to_embed:
                    self.log_info(f"Embedding module: {module_to_embed}")
                    success = self.index_builder.build_incremental_module_embedding(
                        module_name=module_to_embed
                    )
                    if success:
                        embedded_modules.append(module_to_embed)
                
                # Skip search if no modules embedded yet
                if not embedded_modules:
                    continue
                
                # Generate search query
                success, query, query_reasoning = self.llm_agent.generate_rag_query(
                    error_context=context,
                    previous_attempts=[],
                    round_num=round_num
                )
                
                if not success:
                    continue
                
                rag_queries.append(query)
                
                # Decide search target
                target = self.llm_agent.decide_search_target(context, query)
                
                # Execute search
                all_results = []
                
                if target in ["methods", "both"]:
                    method_results = self._search_collection(
                        chroma_path, "java_methods", query, top_k
                    )
                    all_results.extend(method_results)
                
                if target in ["fields", "both"]:
                    field_results = self._search_collection(
                        chroma_path, "java_fields", query, top_k
                    )
                    all_results.extend(field_results)
                
                # Enrich with full bodies from SQLite
                enriched_results = self._enrich_with_bodies(sqlite_path, all_results)
                
                # Parse into methods/fields
                methods, fields = self._parse_rag_results(enriched_results)
                retrieved_methods.extend(methods)
                retrieved_fields.extend(fields)
                
                # Evaluate usefulness
                decision, eval_reasoning, extracted_context = self.llm_agent.evaluate_rag_results(
                    error_context=context,
                    query=query,
                    results=enriched_results,
                    round_num=round_num,
                    max_rounds=max_rounds
                )
                
                if decision == "USEFUL":
                    self.log_info("RAG retrieval successful")
                    return RetrievalResult(
                        retrieved_methods=retrieved_methods,
                        retrieved_fields=retrieved_fields,
                        retrieval_successful=True,
                        retrieval_reasoning=eval_reasoning,
                        rag_queries_used=rag_queries
                    )
                
                previous_results_summary = f"Round {round_num}: {eval_reasoning}"
            
            # Completed all rounds
            if retrieved_methods or retrieved_fields:
                return RetrievalResult(
                    retrieved_methods=retrieved_methods,
                    retrieved_fields=retrieved_fields,
                    retrieval_successful=True,
                    retrieval_reasoning="RAG retrieval completed with partial results",
                    rag_queries_used=rag_queries
                )
            else:
                return RetrievalResult(
                    retrieval_successful=False,
                    retrieval_reasoning="RAG retrieval found no useful results",
                    rag_queries_used=rag_queries
                )
        
        except Exception as e:
            self.log_error(f"RAG retrieval failed: {e}")
            return RetrievalResult(
                retrieval_successful=False,
                retrieval_reasoning=f"RAG retrieval error: {str(e)}",
                rag_queries_used=rag_queries
            )
    
    def _execute_sql(self, sqlite_path: Path, sql: str) -> List[Dict[str, Any]]:
        """Execute SQL query and return results"""
        try:
            with SQLiteStorage(sqlite_path) as storage:
                cursor = storage.conn.cursor()
                cursor.execute(sql)
                
                # Get column names
                columns = [desc[0] for desc in cursor.description]
                
                # Fetch results
                rows = cursor.fetchall()
                
                # Convert to list of dicts
                results = []
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    results.append(row_dict)
                
                return results
        
        except Exception as e:
            self.log_error(f"SQL execution error: {e}")
            return []
    
    def _search_collection(
        self,
        chroma_path: Path,
        collection_name: str,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Search a ChromaDB collection"""
        try:
            from config import config
            # Use config.llm settings instead of environment variables
            api_key = config.llm.api_key or os.getenv("OPENAI_API_KEY", "")
            # Use get_base_url() to properly handle API URL configuration
            base_url = config.llm.get_base_url()
            # Ensure base_url ends with /v1 for OpenAI-compatible APIs
            if not base_url.endswith('/v1'):
                base_url = base_url.rstrip('/') + '/v1'
            
            vector_store = VectorStore(
                chroma_path=chroma_path,
                collection_name=collection_name,
                openai_api_key=api_key,
                openai_base_url=base_url
            )
            
            results = vector_store.search(
                query=query,
                top_k=top_k,
                include_tests=False
            )
            
            return results
        
        except Exception as e:
            self.log_error(f"ChromaDB search error: {e}")
            return []
    
    def _enrich_with_bodies(
        self,
        sqlite_path: Path,
        results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enrich results with full method/field bodies from SQLite"""
        if not results:
            return []
        
        try:
            enriched = []
            
            with SQLiteStorage(sqlite_path) as storage:
                for result in results:
                    result_id = result.get('id')
                    if not result_id:
                        continue
                    
                    # Query SQLite for full details
                    cursor = storage.conn.cursor()
                    cursor.execute("""
                        SELECT class_name, method_name, signature, javadoc, body, file_path
                        FROM methods
                        WHERE id = ?
                    """, (result_id,))
                    
                    row = cursor.fetchone()
                    if row:
                        enriched_result = dict(result)
                        enriched_result.update({
                            'class_name': row[0],
                            'method_name': row[1],
                            'signature': row[2],
                            'javadoc': row[3],
                            'body': row[4],
                            'file_path': row[5]
                        })
                        enriched.append(enriched_result)
            
            return enriched
        
        except Exception as e:
            self.log_error(f"Error enriching results: {e}")
            return results
    
    def _parse_sql_results(
        self,
        results: List[Dict[str, Any]]
    ) -> Tuple[List[RetrievedMethod], List[RetrievedField]]:
        """Parse SQL results into methods and fields"""
        methods = []
        fields = []
        
        for result in results:
            # Check if it's a method (has method_name) or field (has field_name)
            if 'method_name' in result and result.get('method_name'):
                method = RetrievedMethod(
                    class_name=result.get('class_name', ''),
                    method_name=result.get('method_name', ''),
                    signature=result.get('signature', ''),
                    body=result.get('body'),
                    javadoc=result.get('javadoc'),
                    file_path=result.get('file_path'),
                    relevance_score=1.0
                )
                methods.append(method)
            
            elif 'field_name' in result and result.get('field_name'):
                field = RetrievedField(
                    class_name=result.get('class_name', ''),
                    field_name=result.get('field_name', ''),
                    field_type=result.get('field_type', ''),
                    value=result.get('initializer'),  # Database uses 'initializer' column
                    javadoc=result.get('javadoc'),
                    file_path=result.get('file_path'),
                    relevance_score=1.0
                )
                fields.append(field)
        
        return methods, fields
    
    def _parse_rag_results(
        self,
        results: List[Dict[str, Any]]
    ) -> Tuple[List[RetrievedMethod], List[RetrievedField]]:
        """Parse RAG results into methods and fields"""
        methods = []
        fields = []
        
        for result in results:
            # RAG results include distance/score
            relevance_score = 1.0 - result.get('distance', 0.0)  # Convert distance to similarity
            
            if 'method_name' in result and result.get('method_name'):
                method = RetrievedMethod(
                    class_name=result.get('class_name', ''),
                    method_name=result.get('method_name', ''),
                    signature=result.get('signature', ''),
                    body=result.get('body'),
                    javadoc=result.get('javadoc'),
                    file_path=result.get('file_path'),
                    relevance_score=relevance_score
                )
                methods.append(method)
            
            elif 'field_name' in result and result.get('field_name'):
                field = RetrievedField(
                    class_name=result.get('class_name', ''),
                    field_name=result.get('field_name', ''),
                    field_type=result.get('field_type', ''),
                    value=result.get('initializer'),  # Database uses 'initializer' column
                    javadoc=result.get('javadoc'),
                    file_path=result.get('file_path'),
                    relevance_score=relevance_score
                )
                fields.append(field)
        
        return methods, fields
    
    def _format_hunks_for_context(self, filtered_hunks: List[DiffHunk]) -> str:
        """Format diff hunks for LLM context"""
        if not filtered_hunks:
            return "No relevant code changes"
        
        formatted = []
        for i, hunk in enumerate(filtered_hunks[:5], 1):  # Show first 5
            hunk_type = hunk.hunk_type or "unknown"
            formatted.append(f"\nHunk {i} [{hunk_type}] - {hunk.file_path}:")
            formatted.append(hunk.context)
        
        if len(filtered_hunks) > 5:
            formatted.append(f"\n... and {len(filtered_hunks) - 5} more hunks")
        
        return '\n'.join(formatted)
    
    def _try_lsp_retrieval_root_cause(
        self,
        test_code: str,
        focal_method_info: FocalMethodInfo,
        filtered_hunks: List[DiffHunk],
        focal_method_changed: bool
    ) -> RetrievalResult:
        """
        Try LSP-based retrieval for root cause analysis.
        
        Args:
            test_code: The test code
            focal_method_info: Focal method information
            filtered_hunks: Filtered relevant hunks
            focal_method_changed: Whether focal method changed
            
        Returns:
            RetrievalResult
        """
        self.log_info("Attempting LSP retrieval for root cause analysis...")
        
        if not self.use_lsp or not hasattr(self, 'lsp_agent'):
            return RetrievalResult(
                retrieval_successful=False,
                retrieval_reasoning="LSP agent not initialized"
            )
        
        # Extract symbols from test code and focal method
        symbols = self._extract_symbols_from_code(test_code)
        symbols.extend(self._extract_symbols_from_code(focal_method_info.current_code))
        
        # Remove duplicates
        symbols = list(set(symbols))
        
        if not symbols:
            self.log_warning("No symbols extracted for LSP retrieval")
            return RetrievalResult(
                retrieval_successful=False,
                retrieval_reasoning="No symbols found to retrieve"
            )
        
        self.log_info(f"Extracted {len(symbols)} symbols for LSP retrieval")
        
        # Use LSP to retrieve definitions
        # Note: We need a context file - use focal method's file
        context_file = getattr(focal_method_info, 'source_file_path', '') or getattr(focal_method_info, 'file_path', '')
        
        if not context_file:
            # Try to extract from hunks
            if filtered_hunks:
                context_file = filtered_hunks[0].file_path
        
        if not context_file:
            return RetrievalResult(
                retrieval_successful=False,
                retrieval_reasoning="No context file available for LSP"
            )
        
        # Try to find test file path from filtered_hunks
        test_file = None
        if filtered_hunks:
            for hunk in filtered_hunks:
                if '/test/' in hunk.file_path or hunk.file_path.endswith('Test.java'):
                    test_file = hunk.file_path
                    break
        
        # If not found in hunks, try to infer from context_file
        if not test_file and context_file:
            if '/src/main/java/' in context_file:
                test_file = context_file.replace('/src/main/java/', '/src/test/java/')
                if test_file.endswith('.java') and not test_file.endswith('Test.java'):
                    test_file = test_file[:-5] + 'Test.java'
        
        try:
            result = self.lsp_agent.retrieve_symbols(
                symbols=symbols[:10],  # Limit to 10 symbols
                context_file=context_file,
                context_code=test_code + "\n" + focal_method_info.current_code,
                test_file=test_file
            )
            return result
        except Exception as e:
            self.log_error(f"LSP retrieval failed: {e}")
            return RetrievalResult(
                retrieval_successful=False,
                retrieval_reasoning=f"LSP retrieval error: {str(e)}"
            )
    
    def _try_lsp_retrieval_error(
        self,
        test_code: str,
        focal_method_info: FocalMethodInfo,
        filtered_hunks: List[DiffHunk],
        error_message: str,
        raw_error_output: Optional[str]
    ) -> RetrievalResult:
        """
        Try LSP-based retrieval for error analysis.
        
        Args:
            test_code: The updated test code
            focal_method_info: Focal method information
            filtered_hunks: Filtered relevant hunks
            error_message: Error message
            raw_error_output: Raw error output
            
        Returns:
            RetrievalResult
        """
        self.log_info("Attempting LSP retrieval for error analysis...")
        
        if not self.use_lsp or not hasattr(self, 'lsp_agent'):
            return RetrievalResult(
                retrieval_successful=False,
                retrieval_reasoning="LSP agent not initialized"
            )
        
        # Extract symbols from error message
        symbols = self._extract_symbols_from_error(error_message, raw_error_output)
        
        if not symbols:
            self.log_warning("No symbols extracted from error for LSP retrieval")
            # Fallback: extract from test code
            symbols = self._extract_symbols_from_code(test_code)
        
        symbols = list(set(symbols))
        
        if not symbols:
            return RetrievalResult(
                retrieval_successful=False,
                retrieval_reasoning="No symbols found in error to retrieve"
            )
        
        self.log_info(f"Extracted {len(symbols)} symbols from error for LSP retrieval")
        
        # Use focal method's file as context
        context_file = getattr(focal_method_info, 'source_file_path', '') or getattr(focal_method_info, 'file_path', '')
        
        if not context_file and filtered_hunks:
            context_file = filtered_hunks[0].file_path
        
        if not context_file:
            return RetrievalResult(
                retrieval_successful=False,
                retrieval_reasoning="No context file available for LSP"
            )
        
        # Try to find test file path from filtered_hunks
        test_file = None
        if filtered_hunks:
            for hunk in filtered_hunks:
                if '/test/' in hunk.file_path or hunk.file_path.endswith('Test.java'):
                    test_file = hunk.file_path
                    break
        
        # If not found in hunks, try to infer from context_file
        if not test_file and context_file:
            if '/src/main/java/' in context_file:
                test_file = context_file.replace('/src/main/java/', '/src/test/java/')
                if test_file.endswith('.java') and not test_file.endswith('Test.java'):
                    test_file = test_file[:-5] + 'Test.java'
        
        try:
            result = self.lsp_agent.retrieve_symbols(
                symbols=symbols[:10],  # Limit to 10 symbols
                context_file=context_file,
                context_code=test_code,
                test_file=test_file
            )
            return result
        except Exception as e:
            self.log_error(f"LSP retrieval failed: {e}")
            return RetrievalResult(
                retrieval_successful=False,
                retrieval_reasoning=f"LSP retrieval error: {str(e)}"
            )
    
    def _extract_symbols_from_code(self, code: str) -> List[str]:
        """Extract potential symbols (method calls, class names) from code"""
        symbols = []
        
        # Extract method calls: word followed by (
        method_pattern = r'\b([A-Z][a-zA-Z0-9]*|[a-z][a-zA-Z0-9]*)\s*\('
        methods = re.findall(method_pattern, code)
        symbols.extend([m + '()' for m in methods if m not in ['if', 'while', 'for', 'switch']])
        
        # Extract class names: capitalized words (likely classes)
        class_pattern = r'\b([A-Z][a-zA-Z0-9]+)\b'
        classes = re.findall(class_pattern, code)
        symbols.extend([c for c in classes if len(c) > 1])
        
        return symbols
    
    def _extract_symbols_from_error(self, error_message: str, raw_error: Optional[str]) -> List[str]:
        """Extract symbols from error messages"""
        symbols = []
        
        error_text = error_message
        if raw_error:
            error_text += "\n" + raw_error
        
        # Extract "cannot find symbol" errors
        symbol_pattern = r'cannot find symbol[:\s]+(?:symbol|variable|class|method):\s+(\w+)'
        found_symbols = re.findall(symbol_pattern, error_text, re.IGNORECASE)
        symbols.extend(found_symbols)
        
        # Extract undefined references
        undefined_pattern = r'undefined reference to [`\'](\w+)'
        undefined = re.findall(undefined_pattern, error_text)
        symbols.extend(undefined)
        
        # Extract from "package X does not exist"
        package_pattern = r'package ([\w.]+) does not exist'
        packages = re.findall(package_pattern, error_text)
        for pkg in packages:
            # Extract last component as potential class name
            symbols.append(pkg.split('.')[-1])
        
        return symbols
    
    def _try_lsp_retrieval_unknown_symbols(
        self,
        symbols: List[str],
        error_message: str,
        test_code: str,
        focal_method_info: FocalMethodInfo,
        filtered_hunks: List[DiffHunk]
    ) -> RetrievalResult:
        """
        Try LSP-based retrieval for unknown symbols.
        
        Args:
            symbols: List of unknown symbols to retrieve
            error_message: Error message
            test_code: Test code
            focal_method_info: Focal method information
            filtered_hunks: Filtered relevant hunks
            
        Returns:
            RetrievalResult with retrieved_items grouped by symbol
        """
        self.log_info("Attempting LSP retrieval for unknown symbols...")
        
        if not self.use_lsp or not hasattr(self, 'lsp_agent'):
            return RetrievalResult(
                retrieval_successful=False,
                retrieval_reasoning="LSP agent not initialized",
                retrieved_items={},
                failed_symbols=list(symbols)
            )
        
        # Use focal method's file as context
        context_file = getattr(focal_method_info, 'source_file_path', '') or getattr(focal_method_info, 'file_path', '')
        
        if not context_file and filtered_hunks:
            context_file = filtered_hunks[0].file_path
        
        if not context_file:
            return RetrievalResult(
                retrieval_successful=False,
                retrieval_reasoning="No context file available for LSP",
                retrieved_items={},
                failed_symbols=list(symbols)
            )
        
        # Try to find test file path from filtered_hunks
        test_file = None
        if filtered_hunks:
            for hunk in filtered_hunks:
                # Test files usually contain '/test/' in their path
                if '/test/' in hunk.file_path or hunk.file_path.endswith('Test.java'):
                    test_file = hunk.file_path
                    break
        
        # If not found in hunks, try to infer from context_file
        if not test_file and context_file:
            # Convert src/main/java to src/test/java
            if '/src/main/java/' in context_file:
                test_file = context_file.replace('/src/main/java/', '/src/test/java/')
                # Also replace class name: Foo.java -> FooTest.java
                if test_file.endswith('.java') and not test_file.endswith('Test.java'):
                    test_file = test_file[:-5] + 'Test.java'
        
        try:
            result = self.lsp_agent.retrieve_symbols(
                symbols=symbols,
                context_file=context_file,
                context_code=test_code,
                test_file=test_file
            )
            
            # Convert result to retrieved_items format (grouped by symbol)
            retrieved_items = {}
            
            for method in result.retrieved_methods:
                # Try to match method to symbols
                for symbol in symbols:
                    symbol_name = symbol.rstrip('()')
                    if symbol_name in method.method_name or symbol_name in method.class_name:
                        if symbol not in retrieved_items:
                            retrieved_items[symbol] = []
                        retrieved_items[symbol].append({
                            'class_name': method.class_name,
                            'method_name': method.method_name,
                            'signature': method.signature,
                            'body': method.body,
                            'javadoc': method.javadoc,
                            'file_path': method.file_path
                        })
            
            for field in result.retrieved_fields:
                # Try to match field to symbols
                for symbol in symbols:
                    if symbol in field.field_name or symbol in field.class_name:
                        if symbol not in retrieved_items:
                            retrieved_items[symbol] = []
                        retrieved_items[symbol].append({
                            'class_name': field.class_name,
                            'field_name': field.field_name,
                            'field_type': field.field_type,
                            'javadoc': field.javadoc,
                            'file_path': field.file_path
                        })
            
            failed_symbols = [s for s in symbols if s not in retrieved_items]
            
            return RetrievalResult(
                retrieval_successful=len(retrieved_items) > 0,
                retrieval_reasoning=f"LSP retrieved {len(retrieved_items)}/{len(symbols)} symbols",
                retrieved_items=retrieved_items,
                failed_symbols=failed_symbols
            )
        
        except Exception as e:
            self.log_error(f"LSP retrieval for unknown symbols failed: {e}")
            import traceback
            self.log_error(traceback.format_exc())
            return RetrievalResult(
                retrieval_successful=False,
                retrieval_reasoning=f"LSP retrieval error: {str(e)}",
                retrieved_items={},
                failed_symbols=list(symbols)
            )

