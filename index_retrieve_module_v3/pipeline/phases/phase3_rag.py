"""
Phase 3: RAG Semantic Search

Semantic search using vector embeddings with up to 3 rounds of query refinement.
Final phase that determines if useful context was retrieved.
"""
import logging
import time
from typing import Tuple, List, Dict, Any
from pathlib import Path

from ..models import PhaseResult, PhaseStatus, RAGSearchResult
from ..agent import LLMAgent
from ..index_builder import PipelineIndexBuilder
from indexer.vector_store import VectorStore
from indexer.storage import SQLiteStorage

logger = logging.getLogger(__name__)


class Phase3RAGSearch:
    """
    Phase 3: Semantic RAG search with iterative refinement.
    
    Final phase that uses vector embeddings for semantic similarity search.
    """
    
    def __init__(
        self,
        agent: LLMAgent,
        index_builder: PipelineIndexBuilder,
        openai_api_key: str,
        openai_base_url: str,
        max_rounds: int = 3,
        top_k: int = 8
    ):
        """
        Initialize Phase 3.
        
        Args:
            agent: LLM agent for query generation and evaluation
            index_builder: Index builder for creating full index
            openai_api_key: OpenAI API key for embeddings
            openai_base_url: OpenAI API base URL
            max_rounds: Maximum query refinement rounds (default: 3)
            top_k: Number of results to retrieve per query (default: 8)
        """
        self.agent = agent
        self.index_builder = index_builder
        self.openai_api_key = openai_api_key
        self.openai_base_url = openai_base_url
        self.max_rounds = max_rounds
        self.top_k = top_k
    
    def execute(
        self,
        error_context: str,
        skip_rebuild: bool = False
    ) -> Tuple[bool, PhaseResult]:
        """
        Execute Phase 3 RAG search with module-by-module incremental embedding.
        
        Strategy:
        - Each round, agent decides: embed new module OR refine query
        - Cost-aware: Only embed modules when necessary
        - Iterative: Up to max_rounds attempts
        
        Args:
            error_context: Combined error information for context
            skip_rebuild: Skip initial SQLite rebuild (assumes Phase 2 built it)
        
        Returns:
            Tuple of (is_useful, phase_result)
            - is_useful: True if useful context was found
            - phase_result: PhaseResult with findings
        """
        logger.info("=" * 70)
        logger.info("PHASE 3: INCREMENTAL RAG SEMANTIC SEARCH")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        try:
            # Step 1: Ensure SQLite index exists (from Phase 2)
            sqlite_path = self.index_builder.output_dir / "assets.db"
            if not sqlite_path.exists():
                logger.info("SQLite not found, building lightweight index...")
                success = self.index_builder.build_lightweight_index(skip_rebuild=False)
                if not success:
                    duration = time.time() - start_time
                    return False, PhaseResult(
                        phase_name="Phase 3: RAG Search",
                        status=PhaseStatus.FAILED,
                        reason="Failed to build SQLite index",
                        context=None,
                        metadata={"error": "sqlite_build_failed"},
                        duration_seconds=duration
                    )
            
            # Step 2: Get available modules
            available_modules = self.index_builder.get_available_modules()
            if not available_modules:
                duration = time.time() - start_time
                return False, PhaseResult(
                    phase_name="Phase 3: RAG Search",
                    status=PhaseStatus.FAILED,
                    reason="No modules found in repository",
                    context=None,
                    metadata={"error": "no_modules"},
                    duration_seconds=duration
                )
            
            logger.info(f"Available modules: {available_modules}")
            
            # Step 3: Iterative RAG with module-by-module embedding
            chroma_path = self.index_builder.output_dir / "chroma_db"
            embedded_modules = []
            search_history = []
            previous_results_summary = None
            
            for round_num in range(1, self.max_rounds + 1):
                logger.info(f"\n--- Round {round_num}/{self.max_rounds} ---")
                logger.info(f"Embedded modules so far: {embedded_modules if embedded_modules else 'None'}")
                
                # Agent decides: embed new module or refine query?
                action, module_to_embed, action_reasoning = self.agent.decide_rag_action(
                    error_context=error_context,
                    available_modules=available_modules,
                    embedded_modules=embedded_modules,
                    round_num=round_num,
                    previous_results=previous_results_summary
                )
                
                logger.info(f"Agent decision: {action}")
                logger.info(f"Reasoning: {action_reasoning}")
                
                # Execute the action
                if action == "EMBED_MODULE" and module_to_embed:
                    # Embed the new module
                    logger.info(f"Embedding module: {module_to_embed}")
                    success = self.index_builder.build_incremental_module_embedding(
                        module_name=module_to_embed
                    )
                    
                    if not success:
                        logger.warning(f"Failed to embed module {module_to_embed}, continuing anyway")
                    else:
                        embedded_modules.append(module_to_embed)
                        logger.info(f"✓ Module {module_to_embed} embedded successfully")
                
                # If no modules embedded yet, can't search
                if not embedded_modules:
                    logger.warning("No modules embedded yet, cannot search")
                    continue
                
                # Generate search query
                previous_attempts = [h for h in search_history if 'query' in h]
                success, query, query_reasoning = self.agent.generate_rag_query(
                    error_context=error_context,
                    previous_attempts=previous_attempts,
                    round_num=round_num
                )
                
                if not success:
                    logger.warning(f"Failed to generate query in round {round_num}")
                    continue
                
                logger.info(f"Search query: {query[:100]}...")
                
                # Decide search target
                target = self.agent.decide_search_target(error_context, query)
                logger.info(f"Search target: {target}")
                
                # Execute search
                all_results = []
                
                if target in ["methods", "both"]:
                    method_results = self._search_collection(
                        chroma_path,
                        "java_methods",
                        query,
                        self.top_k
                    )
                    all_results.extend(method_results)
                
                if target in ["fields", "both"]:
                    field_results = self._search_collection(
                        chroma_path,
                        "java_fields",
                        query,
                        self.top_k
                    )
                    all_results.extend(field_results)
                
                # Enrich with full bodies
                enriched_results = self._enrich_with_bodies(sqlite_path, all_results)
                
                logger.info(f"Retrieved {len(enriched_results)} results")
                
                # Record this round
                search_history.append({
                    "round": round_num,
                    "action": action,
                    "module_embedded": module_to_embed,
                    "embedded_modules": list(embedded_modules),
                    "query": query,
                    "target": target,
                    "result_count": len(enriched_results)
                })
                
                # Evaluate results
                decision, eval_reasoning, extracted_context = self.agent.evaluate_rag_results(
                    error_context=error_context,
                    query=query,
                    results=enriched_results,
                    round_num=round_num,
                    max_rounds=self.max_rounds
                )
                
                logger.info(f"Evaluation: {decision}")
                
                # Update previous results summary for next round
                previous_results_summary = f"Round {round_num}: Found {len(enriched_results)} results. {eval_reasoning}"
                
                if decision == "USEFUL":
                    # Success!
                    duration = time.time() - start_time
                    logger.info("✓ Success: Found useful code snippets")
                    
                    result = PhaseResult(
                        phase_name="Phase 3: RAG Search",
                        status=PhaseStatus.SUCCESS,
                        reason=eval_reasoning,
                        context=extracted_context,
                        metadata={
                            "rounds_used": round_num,
                            "embedded_modules": embedded_modules,
                            "search_history": search_history,
                            "final_result_count": len(enriched_results)
                        },
                        rounds=round_num,
                        duration_seconds=duration
                    )
                    
                    return True, result
                
                elif decision == "NOT_USEFUL":
                    # Only give up if this is the last round
                    if round_num >= self.max_rounds:
                        logger.info(f"Round {round_num}/{self.max_rounds}: Results not useful, no more rounds")
                        break
                    else:
                        # Not last round - continue trying with different strategy
                        logger.info(f"Round {round_num}/{self.max_rounds}: Results not useful, will try different approach")
                        # Continue to next round (agent will decide new strategy)
                
                # decision == "RETRY" or "NOT_USEFUL" (not last round) - continue to next round
            
            # Completed all rounds without success
            duration = time.time() - start_time
            logger.info("✗ Phase 3 complete: No useful context found")
            
            result = PhaseResult(
                phase_name="Phase 3: RAG Search",
                status=PhaseStatus.FAILED,
                reason="Completed all search rounds but found no useful code context",
                context=None,
                metadata={
                    "rounds_used": len(search_history),
                    "embedded_modules": embedded_modules,
                    "search_history": search_history
                },
                rounds=len(search_history),
                duration_seconds=duration
            )
            
            return False, result
        
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Phase 3 failed with error: {e}", exc_info=True)
            
            result = PhaseResult(
                phase_name="Phase 3: RAG Search",
                status=PhaseStatus.FAILED,
                reason=f"Phase 3 failed: {str(e)}",
                context=None,
                metadata={"error": str(e)},
                duration_seconds=duration
            )
            
            return False, result
    
    def _search_collection(
        self,
        chroma_path: Path,
        collection_name: str,
        query: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Search a specific ChromaDB collection.
        
        Args:
            chroma_path: Path to ChromaDB directory
            collection_name: Collection to search
            query: Search query
            top_k: Number of results
        
        Returns:
            List of search results
        """
        try:
            vector_store = VectorStore(
                chroma_path=chroma_path,
                collection_name=collection_name,
                openai_api_key=self.openai_api_key,
                openai_base_url=self.openai_base_url
            )
            
            # CRITICAL: include_tests=False to exclude test code
            results = vector_store.search(
                query=query,
                top_k=top_k,
                include_tests=False  # Never search test code
            )
            
            return results
        
        except Exception as e:
            logger.error(f"Search failed for collection {collection_name}: {e}")
            return []
    
    def _enrich_with_bodies(
        self,
        sqlite_path: Path,
        rag_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Enrich RAG results with full method/field bodies from SQLite.
        
        Args:
            sqlite_path: Path to SQLite database
            rag_results: Results from vector search
        
        Returns:
            Enriched results with full bodies
        """
        try:
            with SQLiteStorage(sqlite_path) as storage:
                enriched = []
                
                for result in rag_results:
                    sqlite_id = result.get('sqlite_id')
                    
                    if not sqlite_id:
                        enriched.append(result)
                        continue
                    
                    # Try to get full method body
                    method = storage.get_method_by_id(sqlite_id)
                    
                    if method:
                        result['full_body'] = method.get('body', '')
                        result['signature'] = method.get('signature', '')
                        result['javadoc'] = method.get('javadoc', '')
                    else:
                        # Try field
                        field = storage.get_field_by_id(sqlite_id)
                        if field:
                            result['field_type'] = field.get('field_type', '')
                            result['modifiers'] = field.get('modifiers', '')
                            result['initializer'] = field.get('initializer', '')
                            result['javadoc'] = field.get('javadoc', '')
                    
                    enriched.append(result)
                
                return enriched
        
        except Exception as e:
            logger.error(f"Failed to enrich results: {e}")
            return rag_results

