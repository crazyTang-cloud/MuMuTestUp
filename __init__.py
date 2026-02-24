"""
BEAM Framework - Multi-agent Test Update Framework

Build, Evaluate, Adapt, Monitor

A framework for automatically improving test cases using LLM-powered agents.
"""

from orchestrator import Orchestrator
from models import (
    TestCase, FocalMethodInfo, TestResultInfo, TestResultStatus,
    CoverageInfo, MutationInfo, AnalysisResult, UpdateInstruction,
    IterationResult, FrameworkState
)
from agents import (
    ErrorAnalyzeAgent, CoverageAnalysisAgent, MutationAnalysisAgent,
    TestUpdateAgent, CoordinatorAgent
)
from llm import LLMClient
from utils import logger
from config import config

__version__ = "0.1.0"
__author__ = "BEAM Team"

__all__ = [
    # Orchestrator
    'Orchestrator',
    
    # Models
    'TestCase', 'FocalMethodInfo', 'TestResultInfo', 'TestResultStatus',
    'CoverageInfo', 'MutationInfo', 'AnalysisResult', 'UpdateInstruction',
    'IterationResult', 'FrameworkState',
    
    # Agents
    'ErrorAnalyzeAgent', 'CoverageAnalysisAgent', 'MutationAnalysisAgent',
    'TestUpdateAgent', 'CoordinatorAgent',
    
    # Utilities
    'LLMClient', 'logger', 'config'
]
