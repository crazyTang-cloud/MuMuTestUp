"""
Main pipeline orchestrator.

Coordinates execution of all phases with circuit breakers and saves results.
"""
import logging
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from .models import (
    PipelineInput,
    PipelineOutput,
    PipelineStatus,
    PhaseResult,
    PhaseStatus
)
from .agent import LLMAgent
from .git_handler import GitHandler
from .index_builder import PipelineIndexBuilder
from .phases import Phase1Triage, Phase2SQLSearch, Phase3RAGSearch

logger = logging.getLogger(__name__)


class TestFixPipeline:
    """
    Main orchestrator for the test fix auxiliary information retrieval pipeline.
    
    Executes phases sequentially with circuit breakers to avoid unnecessary
    computation and cost.
    """
    
    def __init__(self, pipeline_input: PipelineInput):
        """
        Initialize pipeline.
        
        Args:
            pipeline_input: Pipeline input configuration
        """
        self.input = pipeline_input
        
        # Initialize components
        self.agent = LLMAgent(
            model=pipeline_input.agent_model,
            api_key=pipeline_input.openai_api_key,
            base_url=pipeline_input.openai_base_url,
            temperature=getattr(pipeline_input, 'temperature', 0.1),
            max_retries=getattr(pipeline_input, 'max_retries', 3),
            timeout=getattr(pipeline_input, 'timeout', 3000)
        )
        
        self.git_handler = GitHandler(pipeline_input.repo_path)
        
        self.index_builder = PipelineIndexBuilder(
            repo_path=pipeline_input.repo_path
        )
        
        # Initialize phases
        self.phase1 = Phase1Triage(self.agent)
        
        self.phase2 = Phase2SQLSearch(
            agent=self.agent,
            index_builder=self.index_builder,
            max_rounds=pipeline_input.max_sql_rounds
        )
        
        self.phase3 = Phase3RAGSearch(
            agent=self.agent,
            index_builder=self.index_builder,
            openai_api_key=pipeline_input.openai_api_key,
            openai_base_url=pipeline_input.openai_base_url,
            max_rounds=pipeline_input.max_rag_rounds,
            top_k=pipeline_input.rag_top_k
        )
    
    def run_pipeline(self) -> PipelineOutput:
        """
        Execute the complete pipeline.
        
        Returns:
            PipelineOutput with complete results
        """
        logger.info("=" * 70)
        logger.info("TEST FIX PIPELINE STARTED")
        logger.info("=" * 70)
        logger.info(f"Timestamp: {datetime.now().isoformat()}")
        logger.info(f"Repository: {self.input.repo_path}")
        logger.info(f"Commit: {self.input.error_commit_id}")
        logger.info("=" * 70)
        
        pipeline_start_time = time.time()
        
        output = PipelineOutput(
            final_status=PipelineStatus.ERROR,
            final_reason="Pipeline not completed",
            config_snapshot=self.input.to_dict()
        )
        
        try:
            # Phase 0: Git Reset
            phase0_result = self._execute_phase0()
            output.phase0_result = phase0_result
            
            if phase0_result.status == PhaseStatus.ERROR:
                output.final_status = PipelineStatus.ERROR
                output.final_reason = "Phase 0 failed: " + phase0_result.reason
                output.error_message = phase0_result.reason
                output.total_duration_seconds = time.time() - pipeline_start_time
                return output
            
            # Build error context for subsequent phases
            error_context = self._build_error_context()
            
            # Phase 1: Triage
            should_skip, phase1_result = self.phase1.execute(
                error_test_code=self.input.error_test_code_log,
                error_message=self.input.error_message,
                error_log=self.input.error_log
            )
            output.phase1_result = phase1_result
            
            if should_skip:
                # Circuit breaker triggered
                output.final_status = PipelineStatus.SKIPPED_AT_PHASE_1
                output.final_reason = phase1_result.reason
                output.final_context = None
                output.total_duration_seconds = time.time() - pipeline_start_time
                
                logger.info("=" * 70)
                logger.info("PIPELINE COMPLETED: SKIPPED AT PHASE 1")
                logger.info("=" * 70)
                
                return output
            
            # Phase 2: SQL Metadata Search
            is_sufficient, phase2_result = self.phase2.execute(
                error_context=error_context,
                skip_rebuild=self.input.skip_rebuild
            )
            output.phase2_result = phase2_result
            
            if is_sufficient:
                # Circuit breaker triggered
                output.final_status = PipelineStatus.STOPPED_AT_PHASE_2
                output.final_reason = phase2_result.reason
                output.final_context = phase2_result.context
                output.total_duration_seconds = time.time() - pipeline_start_time
                
                logger.info("=" * 70)
                logger.info("PIPELINE COMPLETED: STOPPED AT PHASE 2")
                logger.info("=" * 70)
                
                return output
            
            # Phase 3: RAG Semantic Search
            is_useful, phase3_result = self.phase3.execute(
                error_context=error_context,
                skip_rebuild=self.input.skip_rebuild
            )
            output.phase3_result = phase3_result
            
            if is_useful:
                # Success
                output.final_status = PipelineStatus.SUCCESS_PHASE_3
                output.final_reason = phase3_result.reason
                output.final_context = phase3_result.context
                output.total_duration_seconds = time.time() - pipeline_start_time
                
                logger.info("=" * 70)
                logger.info("PIPELINE COMPLETED: SUCCESS AT PHASE 3")
                logger.info("=" * 70)
                
                return output
            else:
                # Failed to find useful context
                output.final_status = PipelineStatus.FAILED_AT_PHASE_3
                output.final_reason = phase3_result.reason
                output.final_context = None
                output.total_duration_seconds = time.time() - pipeline_start_time
                
                logger.info("=" * 70)
                logger.info("PIPELINE COMPLETED: FAILED AT PHASE 3")
                logger.info("=" * 70)
                
                return output
        
        except Exception as e:
            logger.error(f"Pipeline failed with exception: {e}", exc_info=True)
            
            output.final_status = PipelineStatus.ERROR
            output.final_reason = f"Pipeline exception: {str(e)}"
            output.error_message = str(e)
            output.total_duration_seconds = time.time() - pipeline_start_time
            
            return output
    
    def _execute_phase0(self) -> PhaseResult:
        """
        Execute Phase 0: Git reset.
        
        Returns:
            PhaseResult for Phase 0
        """
        logger.info("=" * 70)
        logger.info("PHASE 0: GIT ENVIRONMENT RESET")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        if self.input.skip_phase0:
            logger.info("Skipping Phase 0 (skip_phase0=True)")
            
            return PhaseResult(
                phase_name="Phase 0: Git Reset",
                status=PhaseStatus.SUCCESS,
                reason="Skipped by configuration",
                context=None,
                metadata={"skipped": True},
                duration_seconds=time.time() - start_time
            )
        
        try:
            # Check current state
            is_clean, status = self.git_handler.check_clean_working_tree()
            logger.info(f"Working tree status: {status}")
            
            # Reset to target commit
            success, message = self.git_handler.reset_to_commit(
                self.input.error_commit_id
            )
            
            duration = time.time() - start_time
            
            if success:
                return PhaseResult(
                    phase_name="Phase 0: Git Reset",
                    status=PhaseStatus.SUCCESS,
                    reason=message,
                    context=None,
                    metadata={
                        "commit_id": self.input.error_commit_id,
                        "was_clean": is_clean
                    },
                    duration_seconds=duration
                )
            else:
                return PhaseResult(
                    phase_name="Phase 0: Git Reset",
                    status=PhaseStatus.ERROR,
                    reason=message,
                    context=None,
                    metadata={
                        "commit_id": self.input.error_commit_id,
                        "error": message
                    },
                    duration_seconds=duration
                )
        
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Phase 0 failed: {e}", exc_info=True)
            
            return PhaseResult(
                phase_name="Phase 0: Git Reset",
                status=PhaseStatus.ERROR,
                reason=f"Git reset failed: {str(e)}",
                context=None,
                metadata={"error": str(e)},
                duration_seconds=duration
            )
    
    def _build_error_context(self) -> str:
        """
        Build combined error context string for phases.
        
        Returns:
            Combined error context
        """
        context = f"""**Error Information:**

**Test Code:**
```java
{self.input.error_test_code_log}
```

**Error Message:**
{self.input.error_message}

**Error Log (excerpt):**
{self.input.error_log[:2000]}

**Repository:**
- Path: {self.input.repo_path}
- Commit: {self.input.error_commit_id}
"""
        return context
    
    def save_results(
        self,
        output: PipelineOutput,
        output_dir: Optional[Path] = None
    ) -> Path:
        """
        Save pipeline results to JSON file.
        
        Args:
            output: Pipeline output to save
            output_dir: Optional directory (default: ./pipeline_results)
        
        Returns:
            Path to saved file
        """
        if output_dir is None:
            output_dir = Path.cwd() / "pipeline_results"
        
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create filename with timestamp and commit
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        commit_short = self.input.error_commit_id[:8]
        filename = f"pipeline_{timestamp}_{commit_short}.json"
        
        filepath = output_dir / filename
        
        # Save to file
        output.save_to_file(str(filepath))
        
        logger.info(f"Results saved to: {filepath}")
        
        return filepath

