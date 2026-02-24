"""
Phase 2: SQL Metadata Search

Iterative SQL-based metadata retrieval with up to 3 rounds of refinement.
Implements the second circuit breaker if metadata is sufficient.
"""
import logging
import time
import json
from typing import Tuple, List, Dict, Any
from pathlib import Path

from ..models import PhaseResult, PhaseStatus, SQLQueryResult
from ..agent import LLMAgent
from ..index_builder import PipelineIndexBuilder
from indexer.storage import SQLiteStorage

logger = logging.getLogger(__name__)


class Phase2SQLSearch:
    """
    Phase 2: Iterative SQL metadata search.
    
    Attempts to find sufficient information through lightweight SQL queries
    before resorting to expensive vector embeddings.
    """
    
    def __init__(
        self,
        agent: LLMAgent,
        index_builder: PipelineIndexBuilder,
        max_rounds: int = 3
    ):
        """
        Initialize Phase 2.
        
        Args:
            agent: LLM agent for SQL generation and evaluation
            index_builder: Index builder for creating SQLite index
            max_rounds: Maximum query refinement rounds (default: 3)
        """
        self.agent = agent
        self.index_builder = index_builder
        self.max_rounds = max_rounds
    
    def execute(
        self,
        error_context: str,
        skip_rebuild: bool = False
    ) -> Tuple[bool, PhaseResult]:
        """
        Execute Phase 2 SQL search with iterative refinement.
        
        Args:
            error_context: Combined error information for context
            skip_rebuild: Skip index rebuild (for testing)
        
        Returns:
            Tuple of (is_sufficient, phase_result)
            - is_sufficient: True if metadata is sufficient (circuit breaker)
            - phase_result: PhaseResult with findings
        """
        logger.info("=" * 70)
        logger.info("PHASE 2: SQL METADATA SEARCH")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        try:
            # Step 1: Build lightweight index (SQL only, no vectors)
            logger.info("Building lightweight index (SQL only)...")
            
            success = self.index_builder.build_lightweight_index(
                skip_rebuild=skip_rebuild
            )
            
            if not success:
                duration = time.time() - start_time
                logger.error("Failed to build lightweight index")
                
                result = PhaseResult(
                    phase_name="Phase 2: SQL Search",
                    status=PhaseStatus.ERROR,
                    reason="Failed to build SQLite index",
                    context=None,
                    metadata={"error": "index_build_failed"},
                    duration_seconds=duration
                )
                
                return False, result
            
            # Step 2: Iterative SQL querying
            query_history = []
            previous_queries = []
            previous_results = []
            
            sqlite_path = self.index_builder.output_dir / "assets.db"
            
            for round_num in range(1, self.max_rounds + 1):
                logger.info(f"\n--- Round {round_num}/{self.max_rounds} ---")
                
                # Generate SQL query
                success, sql, reasoning = self.agent.generate_sql_query(
                    error_context=error_context,
                    previous_queries=previous_queries,
                    previous_results=previous_results,
                    round_num=round_num
                )
                
                if not success:
                    logger.warning(f"Failed to generate SQL query in round {round_num}")
                    continue
                
                logger.info(f"Generated SQL: {sql}")
                
                # Validate SQL safety
                is_valid, validation_msg = self.agent.validate_sql(sql)
                
                if not is_valid:
                    logger.warning(f"SQL validation failed: {validation_msg}")
                    previous_queries.append(sql)
                    previous_results.append(f"VALIDATION ERROR: {validation_msg}")
                    continue
                
                # Execute SQL query
                query_result = self._execute_sql(sqlite_path, sql)
                
                query_history.append({
                    "round": round_num,
                    "sql": sql,
                    "reasoning": reasoning,
                    "success": query_result.success,
                    "row_count": query_result.row_count,
                    "error": query_result.error
                })
                
                if not query_result.success:
                    logger.warning(f"SQL execution failed: {query_result.error}")
                    previous_queries.append(sql)
                    previous_results.append(f"EXECUTION ERROR: {query_result.error}")
                    continue
                
                logger.info(f"Retrieved {query_result.row_count} rows")
                
                # Evaluate results
                decision, eval_reasoning, extracted_context = self.agent.evaluate_sql_results(
                    error_context=error_context,
                    sql_query=sql,
                    results=query_result.results,
                    round_num=round_num,
                    max_rounds=self.max_rounds
                )
                
                logger.info(f"Evaluation: {decision}")
                
                if decision == "SUFFICIENT":
                    # Circuit breaker triggered - we have enough info
                    duration = time.time() - start_time
                    
                    logger.info("✓ Circuit breaker triggered: Metadata is sufficient")
                    
                    result = PhaseResult(
                        phase_name="Phase 2: SQL Search",
                        status=PhaseStatus.SUCCESS,
                        reason=eval_reasoning,
                        context=extracted_context,
                        metadata={
                            "circuit_breaker": "triggered",
                            "rounds_used": round_num,
                            "query_history": query_history,
                            "final_row_count": query_result.row_count
                        },
                        rounds=round_num,
                        duration_seconds=duration
                    )
                    
                    return True, result
                
                elif decision == "PROCEED_TO_RAG":
                    # Need semantic search
                    break
                
                # decision == "RETRY" - continue to next round
                previous_queries.append(sql)
                results_summary = f"Found {query_result.row_count} rows. {eval_reasoning}"
                previous_results.append(results_summary)
            
            # Completed all rounds without finding sufficient info
            duration = time.time() - start_time
            
            logger.info("→ Proceeding to Phase 3: Metadata insufficient, need semantic search")
            
            result = PhaseResult(
                phase_name="Phase 2: SQL Search",
                status=PhaseStatus.PROCEED,
                reason="SQL metadata search completed but insufficient for fixing error",
                context=None,
                metadata={
                    "circuit_breaker": "not_triggered",
                    "rounds_used": len(query_history),
                    "query_history": query_history
                },
                rounds=len(query_history),
                duration_seconds=duration
            )
            
            return False, result
        
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Phase 2 failed with error: {e}", exc_info=True)
            
            result = PhaseResult(
                phase_name="Phase 2: SQL Search",
                status=PhaseStatus.ERROR,
                reason=f"Phase 2 failed: {str(e)}",
                context=None,
                metadata={"error": str(e)},
                duration_seconds=duration
            )
            
            return False, result
    
    def _execute_sql(self, db_path: Path, sql: str) -> SQLQueryResult:
        """
        Execute SQL query against SQLite database.
        
        Args:
            db_path: Path to SQLite database
            sql: SQL query to execute
        
        Returns:
            SQLQueryResult with results or error
        """
        try:
            with SQLiteStorage(db_path) as storage:
                cursor = storage.conn.cursor()
                cursor.execute(sql)
                rows = cursor.fetchall()
                
                # Convert rows to list of dicts
                results = []
                for row in rows:
                    results.append(dict(row))
                
                return SQLQueryResult(
                    query=sql,
                    success=True,
                    results=results,
                    row_count=len(results)
                )
        
        except Exception as e:
            logger.error(f"SQL execution error: {e}")
            return SQLQueryResult(
                query=sql,
                success=False,
                error=str(e),
                row_count=0
            )

