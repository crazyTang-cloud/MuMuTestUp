"""
Phase 1: Error Triage

Analyzes error to determine if it's simple enough to fix without
repository context, implementing the first circuit breaker.
"""
import logging
import time
from typing import Tuple

from ..models import PhaseResult, PhaseStatus
from ..agent import LLMAgent

logger = logging.getLogger(__name__)


class Phase1Triage:
    """
    Phase 1: Initial error triage using LLM.
    
    Determines if the error is simple enough to skip further retrieval,
    implementing the first circuit breaker in the pipeline.
    """
    
    def __init__(self, agent: LLMAgent):
        """
        Initialize Phase 1.
        
        Args:
            agent: LLM agent for decision making
        """
        self.agent = agent
    
    def execute(
        self,
        error_test_code: str,
        error_message: str,
        error_log: str
    ) -> Tuple[bool, PhaseResult]:
        """
        Execute Phase 1 triage.
        
        Args:
            error_test_code: The failing test code
            error_message: Short error message
            error_log: Full error log
        
        Returns:
            Tuple of (should_skip_retrieval, phase_result)
            - should_skip_retrieval: True if error is simple enough to skip
            - phase_result: PhaseResult with decision and reasoning
        """
        logger.info("=" * 70)
        logger.info("PHASE 1: ERROR TRIAGE")
        logger.info("=" * 70)
        
        start_time = time.time()
        
        try:
            # Call LLM agent for triage
            can_skip, reasoning, decision = self.agent.triage_error(
                error_test_code=error_test_code,
                error_message=error_message,
                error_log=error_log
            )
            
            duration = time.time() - start_time
            
            if can_skip:
                # Circuit breaker triggered - error is simple
                logger.info("✓ Circuit breaker triggered: Error is simple enough to skip retrieval")
                
                result = PhaseResult(
                    phase_name="Phase 1: Triage",
                    status=PhaseStatus.SKIP,
                    reason=reasoning,
                    context=None,
                    metadata={
                        "decision": decision,
                        "circuit_breaker": "triggered"
                    },
                    duration_seconds=duration
                )
                
                return True, result
            
            else:
                # Need to proceed to Phase 2
                logger.info("→ Proceeding to Phase 2: Error requires repository context")
                
                result = PhaseResult(
                    phase_name="Phase 1: Triage",
                    status=PhaseStatus.PROCEED,
                    reason=reasoning,
                    context=None,
                    metadata={
                        "decision": decision,
                        "circuit_breaker": "not_triggered"
                    },
                    duration_seconds=duration
                )
                
                return False, result
        
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Phase 1 failed with error: {e}", exc_info=True)
            
            # On error, proceed to be safe
            result = PhaseResult(
                phase_name="Phase 1: Triage",
                status=PhaseStatus.ERROR,
                reason=f"Phase 1 failed: {str(e)}. Proceeding to be safe.",
                context=None,
                metadata={"error": str(e)},
                duration_seconds=duration
            )
            
            return False, result

