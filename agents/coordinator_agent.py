from typing import List, Optional, Dict, Any
from models import (
    AnalysisResult, TestResultInfo, TestResultStatus, 
    UpdateInstruction, IterationResult, FrameworkState, DiffHunk
)
from agents.base_agent import BaseAgent


class CoordinatorAgent(BaseAgent):
    """Orchestrator agent that coordinates all other agents and manages framework flow"""
    
    def __init__(self, retrieval_agent=None, repo_path: str = None, project_name: str = None):
        super().__init__("CoordinatorAgent", "coordinator")
        self.retrieval_agent = retrieval_agent
        self.repo_path = repo_path  # For passing to ErrorAnalyzeAgent
        self.project_name = project_name  # For passing to ErrorAnalyzeAgent
        self.filtered_hunks = []  # Store filtered hunks from root cause analysis
        self.filtered_class_fields = None  # Store filtered class fields from root cause analysis
        self.filtered_non_test_methods = None  # Store filtered non-test methods from root cause analysis
    
    def execute(self, *args, **kwargs):
        """Required implementation of abstract method (not used directly)"""
        pass
    
    def route_test_result(self, test_result: TestResultInfo) -> List[AnalysisResult]:
        """
        Route test result to appropriate analysis agents
        
        Args:
            test_result: The test result to route
            
        Returns:
            List of analysis results from relevant agents
        """
        self.log_info(f"Routing test result: {test_result.status}")
        
        from config import config
        
        analysis_results = []
        
        if test_result.status == TestResultStatus.COMPILE_ERROR:
            analysis_results.append(self._route_to_error_analyzer(test_result))
        
        elif test_result.status == TestResultStatus.RUN_FAIL:
            analysis_results.append(self._route_to_error_analyzer(test_result))
        
        elif test_result.status == TestResultStatus.COVERAGE_LOSS:
            analysis_results.append(self._route_to_coverage_analyzer(test_result))
        
        elif test_result.status == TestResultStatus.MUTATION_LOSS:
            analysis_results.append(self._route_to_mutation_analyzer(test_result))
        
        elif test_result.status == TestResultStatus.COVERAGE_AND_MUTATION_LOSS:
            # Route to both, coverage first (higher priority)
            analysis_results.append(self._route_to_coverage_analyzer(test_result))
            analysis_results.append(self._route_to_mutation_analyzer(test_result))
        
        elif test_result.status == TestResultStatus.PASS:
            # Even though test passes, check if we need to improve coverage/mutation
            coverage_percentage = test_result.test_case.coverage_info.coverage_percentage
            mutation_kill_rate = test_result.test_case.mutation_info.kill_percentage
            
            coverage_threshold = config.framework.coverage_threshold * 100
            mutation_threshold = config.framework.mutation_threshold * 100
            
            # If coverage is below threshold, analyze it
            if coverage_percentage < coverage_threshold:
                self.log_info(f"Test passes but coverage {coverage_percentage:.1f}% < {coverage_threshold:.1f}%, routing to CoverageAnalysisAgent")
                analysis_results.append(self._route_to_coverage_analyzer(test_result))
            
            # If mutation kill rate is below threshold, analyze it
            if mutation_kill_rate < mutation_threshold:
                self.log_info(f"Test passes but mutation kill {mutation_kill_rate:.1f}% < {mutation_threshold:.1f}%, routing to MutationAnalysisAgent")
                analysis_results.append(self._route_to_mutation_analyzer(test_result))
        
        return analysis_results
    
    def _route_to_error_analyzer(self, test_result: TestResultInfo) -> AnalysisResult:
        """Route to error analysis agent or create simplified result if disabled"""
        from config import config
        
        # Skip if all_ablation_disable_error or ablation_disable_error
        if config.framework.all_ablation_disable_error or config.framework.ablation_disable_error:
            # Ablation: skip error_analyze_agent, return simplified result with raw error log
            self.log_info("Ablation: error_analyze_agent disabled, using raw error log")
            analysis = AnalysisResult(
                agent_name="ErrorAnalyzeAgent (Disabled)",
                analysis_type="error"
            )
            
            # Set error_type and root_cause based on error status
            if test_result.status == TestResultStatus.COMPILE_ERROR:
                analysis.error_type = "compilation_error"
                analysis.root_cause = "Compilation error"
            elif test_result.status == TestResultStatus.RUN_FAIL:
                # Check if it's assertion failure based on error message
                error_msg = test_result.error_message or ""
                if "assert" in error_msg.lower() or "expected" in error_msg.lower():
                    analysis.error_type = "assertion_failure"
                    analysis.root_cause = "Assertion failure"
                else:
                    analysis.error_type = "runtime_error"
                    analysis.root_cause = "Runtime error"
            else:
                analysis.error_type = "unknown"
                analysis.root_cause = "Test failure"
            
            analysis.explanation = f"Raw error log:\n{test_result.error_message or 'No error message available'}"
            analysis.retrieval_result = None
            return analysis
        
        # Normal mode: use error_analyze_agent
        from agents.error_analyze_agent import ErrorAnalyzeAgent
        
        # Pass retrieval_agent=None if retrieval is disabled
        retrieval_agent = None if config.framework.ablation_disable_retrieval else self.retrieval_agent
        
        agent = ErrorAnalyzeAgent(
            retrieval_agent=retrieval_agent,
            repo_path=self.repo_path,
            project_name=self.project_name
        )
        return agent.execute(test_result, filtered_hunks=self.filtered_hunks)
    
    def _route_to_coverage_analyzer(self, test_result: TestResultInfo) -> AnalysisResult:
        """Route to coverage analysis agent or create simplified result if disabled"""
        from config import config
        
        # Skip if all_ablation_disable_coverage or all_ablation_disable_error
        if config.framework.all_ablation_disable_coverage or config.framework.all_ablation_disable_error:
            # Ablation: skip coverage_analysis_agent, return simplified result
            self.log_info("Ablation: coverage_analysis_agent disabled, using simplified message")
            coverage_percentage = test_result.test_case.coverage_info.coverage_percentage
            analysis = AnalysisResult(
                agent_name="CoverageAnalysisAgent (Disabled)",
                analysis_type="coverage"
            )
            analysis.strategies = ["Improve coverage"]
            analysis.recommendations = {}
            analysis.metadata = {
                "current_coverage": coverage_percentage,
                "priority": "HIGH"
            }
            return analysis
        
        if config.framework.ablation_disable_coverage:
            # Ablation: skip coverage_analysis_agent, return simplified result
            self.log_info("Ablation: coverage_analysis_agent disabled, using simplified message")
            coverage_percentage = test_result.test_case.coverage_info.coverage_percentage
            analysis = AnalysisResult(
                agent_name="CoverageAnalysisAgent (Disabled)",
                analysis_type="coverage"
            )
            analysis.strategies = ["Improve coverage"]
            analysis.recommendations = {}
            analysis.metadata = {
                "current_coverage": coverage_percentage,
                "priority": "HIGH"
            }
            return analysis
        
        if config.framework.ablation_coverage_lines_only:
            # Ablation: use coverage_analysis_agent but only process line coverage, skip branch coverage
            self.log_info("Ablation: coverage_analysis_agent enabled (lines only mode, skipping branch coverage)")
            
            # Normal mode: use coverage_analysis_agent with lines_only=True
            from agents.coverage_analysis_agent import CoverageAnalysisAgent
            agent = CoverageAnalysisAgent(lines_only=True)
            return agent.execute(test_result)
        
        # Normal mode: use coverage_analysis_agent
        from agents.coverage_analysis_agent import CoverageAnalysisAgent
        agent = CoverageAnalysisAgent()
        return agent.execute(test_result)
    
    def _route_to_mutation_analyzer(self, test_result: TestResultInfo) -> AnalysisResult:
        """Route to mutation analysis agent or create simplified result if disabled"""
        from config import config
        
        # Skip if all_ablation_disable_mutation or all_ablation_disable_error
        if config.framework.all_ablation_disable_mutation or config.framework.all_ablation_disable_error:
            # Ablation: skip mutation_analysis_agent, return simplified result
            self.log_info("Ablation: mutation_analysis_agent disabled, using simplified message")
            kill_percentage = test_result.test_case.mutation_info.kill_percentage
            analysis = AnalysisResult(
                agent_name="MutationAnalysisAgent (Disabled)",
                analysis_type="mutation"
            )
            analysis.strategies = ["Improve mutation killing"]
            analysis.recommendations = {}
            analysis.metadata = {
                "current_kill_rate": kill_percentage,
                "priority": "HIGH"
            }
            return analysis
        
        if config.framework.ablation_disable_mutation:
            # Ablation: skip mutation_analysis_agent, return simplified result
            self.log_info("Ablation: mutation_analysis_agent disabled, using simplified message")
            kill_percentage = test_result.test_case.mutation_info.kill_percentage
            analysis = AnalysisResult(
                agent_name="MutationAnalysisAgent (Disabled)",
                analysis_type="mutation"
            )
            analysis.strategies = ["Improve mutation killing"]
            analysis.recommendations = {}
            analysis.metadata = {
                "current_kill_rate": kill_percentage,
                "priority": "HIGH"
            }
            return analysis
        
        # Normal mode: use mutation_analysis_agent
        from agents.mutation_analysis_agent import MutationAnalysisAgent
        agent = MutationAnalysisAgent()
        return agent.execute(test_result)
    
    def synthesize_instructions(self, analysis_results: List[AnalysisResult]) -> List[UpdateInstruction]:
        """
        Synthesize analysis results into update instructions (simplified).
        
        Now just packages AnalysisResult into UpdateInstruction without priority filtering.
        
        Args:
            analysis_results: List of analysis results from various agents
            
        Returns:
            List of update instructions (one per analysis result)
        """
        self.log_info(f"Synthesizing {len(analysis_results)} analysis results into instructions")
        
        instructions = []
        
        for result in analysis_results:
            if result.analysis_type == "error":
                instructions.append(self._create_error_instruction(result))
            elif result.analysis_type == "coverage":
                instructions.extend(self._create_coverage_instructions(result))
            elif result.analysis_type == "mutation":
                instructions.extend(self._create_mutation_instructions(result))
        
        self.log_info(f"Generated {len(instructions)} update instruction(s)")
        for i, instr in enumerate(instructions):
            self.log_info(f"  {i+1}. {instr.instruction_type}: {instr.reasoning}")
        
        return instructions
    
    def _create_error_instruction(self, analysis: AnalysisResult) -> UpdateInstruction:
        """
        Create simplified instruction from error analysis (directly packaging AnalysisResult).
        
        Args:
            analysis: AnalysisResult from ErrorAnalyzeAgent
            
        Returns:
            Single UpdateInstruction with all analysis details
        """
        # Determine instruction type based on error type
        if analysis.error_type == 'assertion_failure':
            instruction_type = 'fix_assertion_failure'
            reasoning = 'Fix assertion failure based on code changes'
        elif analysis.error_type in ['compilation_error', 'project_symbol_missing', 
                                     'import_missing', 'common_library_missing']:
            instruction_type = 'fix_compilation_error'
            reasoning = 'Fix compilation error based on analysis'
        else:
            instruction_type = 'fix_error'
            reasoning = 'Fix general error'
        
        # Package all AnalysisResult fields into details
        details = {
            'error_type': analysis.error_type,
            'known_symbols': analysis.known_symbols,
            'unknown_symbols': analysis.unknown_symbols,
            'error_locations': analysis.error_locations,
            'root_cause': analysis.root_cause,
            'explanation': analysis.explanation,
            'retrieval_result': analysis.retrieval_result,
            'annotated_test_code': analysis.annotated_test_code
        }
        
        return UpdateInstruction(
            instruction_type=instruction_type,
            details=details,
            reasoning=reasoning
        )
    
    def _create_coverage_instructions(self, analysis: AnalysisResult) -> List[UpdateInstruction]:
        """Create instructions from coverage analysis"""
        instructions = []
        
        recommendations = analysis.recommendations
        strategies = analysis.strategies
        metadata = analysis.metadata or {}
        
        # Check if this is ablation_coverage_lines_only mode (has uncovered data)
        if not recommendations and strategies and "uncovered" in metadata:
            # Ablation mode: return uncovered lines only
            uncovered_data = metadata["uncovered"]
            
            instructions.append(UpdateInstruction(
                instruction_type="improve_coverage",
                details={
                    "focus": "uncovered_lines",
                    "strategy": strategies[0] if strategies else "Improve coverage",
                    "uncovered_lines": uncovered_data
                },
                reasoning=f"Cover {len(uncovered_data)} uncovered lines in focal method"
            ))
            return instructions
        
        # Check if this is ablation_disable_coverage mode (empty recommendations)
        if not recommendations and strategies:
            # Ablation mode: create a detailed instruction with general guidance
            # Provide comprehensive coverage improvement strategies
            general_guidance = {
                "focus": "general",
                "strategy": strategies[0] if strategies else "Improve coverage",
                "general_strategies": [
                    "Add test cases to exercise different code paths and branches",
                    "Test conditional statements with both true and false branches",
                    "Cover exception handling paths (try-catch blocks)",
                    "Test method calls with various input combinations",
                    "Include edge cases and boundary conditions",
                    "Test loops with different iteration counts (0, 1, many)"
                ],
                "coverage_priorities": [
                    {
                        "type": "Control flow branches",
                        "strategy": "Ensure all if/else, switch cases, and conditional operators are tested"
                    },
                    {
                        "type": "Loop variations",
                        "strategy": "Test loops with empty, single, and multiple iterations"
                    },
                    {
                        "type": "Exception paths",
                        "strategy": "Test both normal execution and exception handling paths"
                    },
                    {
                        "type": "Method calls",
                        "strategy": "Invoke all public methods and verify their behavior"
                    },
                    {
                        "type": "Return statements",
                        "strategy": "Cover all possible return paths in methods"
                    }
                ],
                "priority_actions": [
                    "First, identify uncovered lines and branches in the focal method",
                    "Second, add test cases that trigger these uncovered paths",
                    "Third, use different input values to exercise various code paths",
                    "Finally, add assertions to verify the behavior of newly covered code"
                ]
            }
            
            instructions.append(UpdateInstruction(
                instruction_type="improve_coverage",
                details=general_guidance,
                reasoning="Improve code coverage with comprehensive test scenarios"
            ))
            return instructions
        
        # Normal mode: detailed coverage analysis
        # Get branch improvements (分支优先)
        branch_improvements = recommendations.get("branch_improvements", [])
        easy_to_cover = recommendations.get("easy_to_cover", [])
        moderate = recommendations.get("moderate_difficulty", [])
        
        # Priority-based coverage instructions (only add the first available)
        # Priority: easy (branches + lines) > moderate (branches + lines)
        # Merge branches and lines of the same difficulty into one instruction
        easy_branches = [b for b in branch_improvements if self._is_easy_branch(b)]
        moderate_branches = [b for b in branch_improvements if not self._is_easy_branch(b)]
        
        # Check if we have any easy items (branches or lines)
        if easy_branches or easy_to_cover:
            details = {"difficulty": "easy"}
            reasoning_parts = []
            
            if easy_branches:
                details["branches_to_cover"] = easy_branches
                reasoning_parts.append("easy-to-reach branches")
            
            if easy_to_cover:
                details["lines_to_cover"] = easy_to_cover
                reasoning_parts.append("easy-to-reach lines")
            
            # Set target_type based on what we have
            if easy_branches and easy_to_cover:
                details["target_type"] = "mixed"
            elif easy_branches:
                details["target_type"] = "branch"
            else:
                details["target_type"] = "line"
            
            reasoning = f"Add tests to cover {' and '.join(reasoning_parts)}"
            
            instructions.append(UpdateInstruction(
                instruction_type="improve_coverage",
                details=details,
                reasoning=reasoning
            ))
        
        # Check if we have any moderate items (branches or lines)
        elif moderate_branches or moderate:
            details = {"difficulty": "moderate"}
            reasoning_parts = []
            
            if moderate_branches:
                details["branches_to_cover"] = moderate_branches
                reasoning_parts.append("moderate difficulty branches")
            
            if moderate:
                details["lines_to_cover"] = moderate
                reasoning_parts.append("moderate difficulty lines")
            
            # Set target_type based on what we have
            if moderate_branches and moderate:
                details["target_type"] = "mixed"
            elif moderate_branches:
                details["target_type"] = "branch"
            else:
                details["target_type"] = "line"
            
            reasoning = f"Add tests for {' and '.join(reasoning_parts)}"
            
            instructions.append(UpdateInstruction(
                instruction_type="improve_coverage",
                details=details,
                reasoning=reasoning
            ))
        
        return instructions
    
    def _is_easy_branch(self, branch_info: Dict[str, Any]) -> bool:
        """Determine if a branch is easy to reach based on difficulty field or description"""
        if not isinstance(branch_info, dict):
            return True  # Default to easy if format is unexpected
        
        # First check if LLM provided explicit difficulty field
        difficulty = branch_info.get("difficulty", "").lower()
        if difficulty:
            return difficulty == "easy"
        
        # Fallback: infer from description
        text = str(branch_info.get("missing_condition", "")) + str(branch_info.get("suggested_test", ""))
        text_lower = text.lower()
        
        # Indicators of moderate/hard difficulty
        hard_indicators = [
            "mock", "exception", "error", "complex", "multiple", 
            "json", "parse", "malformed", "network", "io"
        ]
        
        return not any(indicator in text_lower for indicator in hard_indicators)
    
    def _create_mutation_instructions(self, analysis: AnalysisResult) -> List[UpdateInstruction]:
        """Create instructions from mutation analysis
        
        Note: Mutations on uncovered lines are excluded as they should be handled
        by coverage improvements first.
        """
        instructions = []
        
        recommendations = analysis.recommendations
        strategies = analysis.strategies
        
        # Check if this is ablation mode (empty recommendations)
        if not recommendations and strategies:
            # Ablation mode: create a detailed instruction with general guidance
            # Provide comprehensive mutation killing strategies
            general_guidance = {
                "focus": "general",
                "strategy": strategies[0] if strategies else "Improve mutation killing",
                "general_strategies": [
                    "Add stronger assertions to verify return values, object states, and side effects",
                    "Test boundary conditions (null, empty, zero, negative values, edge cases)",
                    "Verify exception handling and error conditions are tested",
                    "Add assertions for intermediate states and method call effects",
                    "Use multiple assertions per test to catch different mutation types",
                    "Test both positive and negative scenarios thoroughly"
                ],
                "common_mutation_types": [
                    {
                        "type": "Return value mutations",
                        "strategy": "Add assertions on return values (e.g., assertNotNull, assertEquals, assertTrue/False)"
                    },
                    {
                        "type": "Conditional boundary mutations",
                        "strategy": "Test boundary values (e.g., ==, !=, <, >, <=, >=) with edge cases"
                    },
                    {
                        "type": "Arithmetic operator mutations",
                        "strategy": "Verify calculation results with specific expected values"
                    },
                    {
                        "type": "Logical operator mutations (&&, ||, !)",
                        "strategy": "Test all combinations of boolean conditions"
                    },
                    {
                        "type": "Method call removals",
                        "strategy": "Assert side effects and state changes after method calls"
                    }
                ],
                "priority_actions": [
                    "First, ensure all return values are explicitly asserted",
                    "Second, add boundary condition tests for conditionals",
                    "Third, verify state changes and side effects",
                    "Finally, strengthen existing assertions with more specific checks"
                ]
            }
            
            instructions.append(UpdateInstruction(
                instruction_type="improve_mutation_kill",
                details=general_guidance,
                reasoning="Improve mutation killing with comprehensive testing strategies"
            ))
            return instructions
        
        # Normal mode: detailed mutation analysis
        # Get survived mutations (已覆盖但未杀死的变异体)
        survived_mutation_fixes = recommendations.get("survived_mutation_fixes", [])
        
        # NO_COVERAGE mutations should NOT be included here - they belong to coverage
        # no_coverage_fixes = recommendations.get("no_coverage_fixes", [])
        
        mutation_types_summary = recommendations.get("mutation_types_summary", [])
        boundary_tests = recommendations.get("boundary_tests", [])
        test_improvements = recommendations.get("test_improvements", [])
        
        # Priority-based mutation instructions (only add the first available)
        # Priority: easy mutations > moderate mutations > strengthen assertions > boundary tests
        easy_mutations = []
        moderate_mutations = []
        
        if survived_mutation_fixes:
            easy_mutations = [m for m in survived_mutation_fixes if self._is_easy_mutation(m)]
            moderate_mutations = [m for m in survived_mutation_fixes if not self._is_easy_mutation(m)]
        
        if easy_mutations:
            instructions.append(UpdateInstruction(
                instruction_type="improve_mutation_kill",
                details={
                    "survived_mutations": easy_mutations,
                    "difficulty": "easy",
                    "mutation_types_summary": mutation_types_summary,
                    "test_improvements": test_improvements
                },
                reasoning="Fix easy-to-kill survived mutations with targeted assertions"
            ))
        elif moderate_mutations:
            instructions.append(UpdateInstruction(
                instruction_type="improve_mutation_kill",
                details={
                    "survived_mutations": moderate_mutations,
                    "difficulty": "moderate",
                    "mutation_types_summary": mutation_types_summary,
                    "test_improvements": test_improvements
                },
                reasoning="Fix moderate difficulty survived mutations"
            ))
        elif "strengthen_assertions" in strategies:
            instructions.append(UpdateInstruction(
                instruction_type="improve_mutation_kill",
                details={
                    "focus": "assertion_strength",
                    "test_improvements": test_improvements
                },
                reasoning="Strengthen existing assertions to kill more mutations"
            ))
        elif "add_boundary_tests" in strategies or boundary_tests:
            instructions.append(UpdateInstruction(
                instruction_type="improve_mutation_kill",
                details={
                    "focus": "boundary_cases",
                    "boundary_tests": boundary_tests
                },
                reasoning="Add boundary case tests"
            ))
        
        return instructions
    
    def _is_easy_mutation(self, mutation_info: Dict[str, Any]) -> bool:
        """Determine if a mutation is easy to kill"""
        if not isinstance(mutation_info, dict):
            return True
        
        # Check mutator type and reasoning
        mutator = mutation_info.get("mutator", "")
        reasoning = mutation_info.get("reasoning", "")
        
        # Some mutator types are generally easier to kill
        easy_mutators = [
            "BooleanFalseReturnValsMutator",
            "BooleanTrueReturnValsMutator",
            "EmptyObjectReturnValsMutator",
            "PrimitiveReturnsMutator"
        ]
        
        # Some require more complex setup
        hard_indicators = ["mock", "complex", "multiple", "side effect", "state"]
        
        is_easy_mutator = any(easy in mutator for easy in easy_mutators)
        has_hard_indicator = any(indicator in reasoning.lower() for indicator in hard_indicators)
        
        return is_easy_mutator or not has_hard_indicator
    
    def _get_uncovered_lines(self, test_result: TestResultInfo) -> List[int]:
        """
        Get uncovered lines from detailed coverage info.
        Extracted from CoverageAnalysisAgent for ablation mode.
        """
        coverage_info = test_result.test_case.coverage_info
        
        # Try to use detailed coverage first (focal method specific)
        if coverage_info.detailed_coverage:
            return coverage_info.detailed_coverage.uncovered_lines
        
        # Fallback to overall coverage
        all_lines = set(range(1, coverage_info.total_lines + 1))
        covered_lines = set(coverage_info.covered_lines)
        uncovered = list(all_lines - covered_lines)
        return sorted(uncovered)
    
    def _extract_uncovered_code(self, test_result: TestResultInfo, uncovered_lines: List[int]) -> List[Dict[str, Any]]:
        """
        Extract code for uncovered lines.
        Returns list of {"line": line_number, "code": line_code}
        """
        focal_method_info = test_result.focal_method_info
        focal_code = focal_method_info.current_code
        start_line = focal_method_info.start_line
        
        # Split focal code into lines
        code_lines = focal_code.split('\n')
        
        # Create a mapping from line number to code
        line_to_code = {}
        for i, code_line in enumerate(code_lines):
            line_num = start_line + i
            line_to_code[line_num] = code_line
        
        # Extract uncovered lines with their code
        uncovered_data = []
        for line_num in uncovered_lines:
            if line_num in line_to_code:
                uncovered_data.append({
                    "line": line_num,
                    "code": line_to_code[line_num].strip()  # Strip whitespace for cleaner output
                })
        
        return uncovered_data
