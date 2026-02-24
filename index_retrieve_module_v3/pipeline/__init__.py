"""
Pipeline package for automated test case fixing auxiliary information retrieval.

This package implements a cost-aware, multi-phase pipeline that progressively
retrieves contextual information to help fix failed test cases.
"""

from .models import (
    PipelineInput,
    PipelineOutput,
    PhaseResult,
    PipelineStatus,
    PhaseStatus
)
from .pipeline import TestFixPipeline

__all__ = [
    'PipelineInput',
    'PipelineOutput',
    'PhaseResult',
    'PipelineStatus',
    'PhaseStatus',
    'TestFixPipeline'
]

