from typing import Optional, Callable, List
from models import (
    TestResultInfo, IterationResult, FrameworkState, 
    TestResultStatus, TestCase, FocalMethodInfo
)
from agents import CoordinatorAgent, TestUpdateAgent, InputPreprocessAgent
from agents.retrieval_agent import RetrievalAgent
from utils import logger, get_sample_logger
from config import config

class Orchestrator:
    """Main orchestrator for the test update framework"""
    
    def __init__(self, repo_path: Optional[str] = None, project_name: Optional[str] = None, 
                 commit_id: Optional[str] = None):
        """
        Initialize orchestrator.
        
        Args:
            repo_path: Path to the repository (for retrieval system)
            project_name: Name of the project (for retrieval system)
            commit_id: Commit ID to index (for retrieval system)
        """
        # Initialize retrieval agent if repo info is provided
        self.retrieval_agent = None
        if repo_path and project_name and commit_id:
            try:
                self.retrieval_agent = RetrievalAgent(repo_path, project_name, commit_id)
                # Ensure index is built (or start LSP server if using LSP)
                if self.retrieval_agent.use_lsp:
                    self.retrieval_agent.start_lsp_server()
                else:
                    self.retrieval_agent.ensure_index_built()
            except Exception as e:
                logger.warning(f"Failed to initialize retrieval agent: {e}")
                logger.warning("Continuing without retrieval system")
        
        # Initialize agents with retrieval agent and repo info
        self.coordinator = CoordinatorAgent(
            retrieval_agent=self.retrieval_agent,
            repo_path=repo_path,
            project_name=project_name
        )
        self.test_updater = TestUpdateAgent()
        # Input pre-processing agent: filters diff hunks and test context before other agents
        self.input_preprocessor = InputPreprocessAgent(retrieval_agent=self.retrieval_agent)
        
        self.logger = logger
        self.sample_logger = get_sample_logger()
        self.state = FrameworkState()
    
    def __del__(self):
        """Cleanup resources when orchestrator is destroyed"""
        if self.retrieval_agent and hasattr(self.retrieval_agent, 'use_lsp') and self.retrieval_agent.use_lsp:
            try:
                self.retrieval_agent.stop_lsp_server()
            except Exception as e:
                # Silently ignore errors during cleanup
                pass
    
    def run(self, test_case: TestCase, focal_method_info: FocalMethodInfo,
            test_executor: Callable[[str], TestResultInfo],
            max_iterations: Optional[int] = None,
            diff_hunks: List = None,
            prioritized_hunks: List = None,
            focal_method_changed: bool = None) -> IterationResult:
        """
        Run the test update framework
        
        Args:
            test_case: Initial test case
            focal_method_info: Information about the focal method
            test_executor: Callable that compiles, runs test and returns TestResultInfo
            max_iterations: Maximum iterations (uses config default if None)
            diff_hunks: List of DiffHunk objects showing code changes (test_method, focal_method, focal_file, high_frequency)
            prioritized_hunks: List of DiffHunk objects from prioritized_changes (for ablation experiments)
            focal_method_changed: Whether the focal method itself has changed (optional)
            
        Returns:
            Best iteration result
        """
        max_iterations = max_iterations or config.framework.max_iterations
        
        # If all_ablation_disable_error is True, force one-shot mode (only 1 iteration)
        if config.framework.all_ablation_disable_error:
            max_iterations = 1
            self.logger.info("Ablation: all_ablation_disable_error enabled - forcing one-shot mode (max_iterations=1)")
        
        # Auto-detect focal_method_changed if not provided
        if focal_method_changed is None:
            focal_method_changed = (focal_method_info.original_code and 
                                   focal_method_info.original_code != focal_method_info.current_code)
        
        self.state.test_case = test_case
        self.state.focal_method_info = focal_method_info
        
        self.logger.info("=" * 60)
        self.logger.info("Starting Test Update Framework")
        self.logger.info(f"Test Case: {test_case.name}")
        self.logger.info(f"Focal Method: {focal_method_info.name}")
        self.logger.info(f"Focal Method Changed: {focal_method_changed}")
        self.logger.info(f"Max Iterations: {max_iterations}")
        self.logger.info("=" * 60)
        
        # Log to sample logger
        self._log_framework_start(test_case, focal_method_info, max_iterations, focal_method_changed)
        
        current_test_code = test_case.code
        current_new_imports = []  # Track new imports across iterations
        test_imports = getattr(test_case, 'test_imports', [])  # Get existing imports from test case
        
        # Log initial test code
        if self.sample_logger.log_file:
            self.sample_logger.log_test_code(current_test_code, "INITIAL TEST CODE")
        
        for iteration in range(1, max_iterations + 1):
            self.state.current_iteration = iteration
            
            self.logger.info(f"\n{'=' * 60}")
            self.logger.info(f"ITERATION {iteration}")
            self.logger.info(f"{'=' * 60}")
            
            # Set iteration in sample logger
            if self.sample_logger.log_file:
                self.sample_logger.set_iteration(iteration)
            
            # Step 1: Generate/Update test code (before execution)
            if iteration == 1:
                # Determine which hunks to use based on configuration
                use_prioritized = config.framework.use_target_prioritized_changes
                
                if use_prioritized:
                    # Ablation experiment: use prioritized_changes directly
                    # prioritized_hunks may be None or empty list
                    actual_prioritized_hunks = prioritized_hunks if prioritized_hunks else []
                    
                    self.logger.info("Step 1a: Using prioritized_changes (ablation experiment mode)...")
                    if self.sample_logger.log_file:
                        self.sample_logger.set_step("Step 1a: Using Prioritized Changes")
                        self.sample_logger.log_info("Ablation experiment: Using prioritized_changes without root cause filtering")
                    
                    # Still need to filter class_fields and non_test_methods
                    # Create a minimal input preprocess result for class context filtering
                    from models import InputPreprocessResult
                    
                    # Filter only class_fields and non_test_methods using LLM
                    filtered_fields, filtered_methods = self.input_preprocessor._filter_test_class_context(
                        test_case=test_case,
                        focal_method_info=focal_method_info,
                        filtered_hunks=actual_prioritized_hunks,  # Use prioritized hunks as context (may be empty)
                        focal_method_changed=focal_method_changed
                    )
                    
                    # Use prioritized hunks directly (no filtering)
                    self.coordinator.filtered_hunks = actual_prioritized_hunks
                    self.coordinator.filtered_class_fields = filtered_fields
                    self.coordinator.filtered_non_test_methods = filtered_methods
                    
                    # Create a simple input_preprocess result for logging
                    root_cause_analysis = InputPreprocessResult(
                        filtered_hunks=actual_prioritized_hunks,
                        retrieval_result=None,
                        reasoning=f"Using {len(actual_prioritized_hunks)} prioritized changes directly (ablation experiment). "
                                 f"Filtered class fields and non-test methods using LLM.",
                        needs_retrieval=False,
                        filtered_class_fields=filtered_fields,
                        filtered_non_test_methods=filtered_methods
                    )
                    
                    # Log
                    if self.sample_logger.log_file:
                        self.sample_logger.log_analysis_result(
                            "InputPreprocessAgent (Prioritized Mode)",
                            root_cause_analysis
                        )
                    
                    self.logger.info(f"Step 1a Complete: Using {len(actual_prioritized_hunks)} prioritized hunks directly")
                else:
                    # Standard mode: use root_cause_analysis_agent to filter all hunks
                    self.logger.info("Step 1a: Preprocessing input (filtering hunks and class context)...")
                    if self.sample_logger.log_file:
                        self.sample_logger.set_step("Step 1a: Root Cause Analysis")
                    
                    # Preprocess input: filter hunks and test class context
                    root_cause_analysis = self.input_preprocessor.execute(
                        test_case=test_case,
                        focal_method_info=focal_method_info,
                        diff_hunks=diff_hunks or [],
                        focal_method_changed=focal_method_changed
                    )
                    
                    # Store filtered hunks and test class context in coordinator for later use
                    self.coordinator.filtered_hunks = root_cause_analysis.filtered_hunks
                    self.coordinator.filtered_class_fields = root_cause_analysis.filtered_class_fields
                    self.coordinator.filtered_non_test_methods = root_cause_analysis.filtered_non_test_methods
                    
                    # Log input preprocessing result
                    if self.sample_logger.log_file:
                        self.sample_logger.log_analysis_result(
                            "InputPreprocessAgent",
                            root_cause_analysis
                        )
                    
                    self.logger.info("Step 1a Complete: Input preprocessing done")
                
                self.logger.info("Step 1b: Generating initial improved test code...")
                if self.sample_logger.log_file:
                    self.sample_logger.set_step("Step 1b: Initial Test Generation")
                
                # Generate initial improved code BEFORE execution
                initial_result = TestResultInfo(
                    status=TestResultStatus.COVERAGE_LOSS,  # dummy status
                    test_case=test_case,
                    focal_method_info=focal_method_info,
                    test_imports=getattr(test_case, 'test_imports', [])
                )
                # Use is_initial=True to trigger special pre-execution prompt
                # Use filtered hunks from root cause analysis
                update_result = self.test_updater.execute(
                    test_result=initial_result,
                    instructions=None,
                    is_initial=True,
                    diff_hunks=root_cause_analysis.filtered_hunks,
                    root_cause_analysis=root_cause_analysis,
                    focal_method_changed=focal_method_changed
                )
                current_test_code = update_result.test_code
                current_new_imports = update_result.new_imports
                
                # Log generated test code
                if self.sample_logger.log_file:
                    self.sample_logger.log_test_code(current_test_code, "GENERATED TEST CODE (Initial)")
                
                self.logger.info("Step 1b Complete: Initial test code generated")
            
            else:
                # Store the initially generated code for non-first iterations
                updated_code = None
            
            # Step 2: Execute test
            self.logger.info("Step 2: Executing test...")
            if self.sample_logger.log_file:
                self.sample_logger.set_step("Step 2: Test Execution")
            
            # Update test case with current imports before execution
            test_case.test_imports = test_imports
            test_case.new_imports = current_new_imports
            
            test_result = test_executor(current_test_code)
            test_result.test_case.code = current_test_code
            test_result.test_case.test_imports = test_imports
            test_result.test_case.new_imports = current_new_imports
            test_result.test_imports = test_imports
            
            self.logger.info(f"Test Status: {test_result.status}")
            
            # Log test result details
            if self.sample_logger.log_file:
                self.sample_logger.log_test_result(test_result)
                
                # Log coverage details
                if test_result.test_case.coverage_info:
                    from java_test_executor import generate_coverage_annotated_code
                    coverage_info = test_result.test_case.coverage_info
                    annotated_code = None
                    if coverage_info.detailed_coverage and focal_method_info.start_line > 0:
                        annotated_code = generate_coverage_annotated_code(
                            focal_method_info.current_code,
                            focal_method_info.start_line,
                            coverage_info.detailed_coverage
                        )
                    self.sample_logger.log_coverage_details(coverage_info, annotated_code)
                
                # Log mutation details
                if test_result.test_case.mutation_info:
                    from java_test_executor import generate_mutation_annotated_code
                    mutation_info = test_result.test_case.mutation_info
                    annotated_code = None
                    if mutation_info.detailed_mutations and focal_method_info.start_line > 0:
                        annotated_code = generate_mutation_annotated_code(
                            focal_method_info.current_code,
                            focal_method_info.start_line,
                            mutation_info.detailed_mutations
                        )
                    self.sample_logger.log_mutation_details(mutation_info, annotated_code)
            
            # Create iteration result
            iteration_result = IterationResult(
                iteration=iteration, 
                test_result=test_result
            )
            
            # Calculate and set score
            iteration_result.score = self._score_result(iteration_result)
            
            # For first iteration, set the initially generated code and new imports
            if iteration == 1:
                iteration_result.updated_test_code = current_test_code
                iteration_result.new_imports = current_new_imports
            
            self.state.iteration_results.append(iteration_result)
            
            # Check if test passed
            if test_result.status == TestResultStatus.PASS:
                self.logger.info("[PASS] Test passed!")
                
                # Check if it meets all criteria
                if self._check_all_criteria_met(test_result):
                    self.logger.info("[PASS] All criteria met! Framework complete.")
                    
                    # Ensure updated_test_code is set before returning
                    if not iteration_result.updated_test_code:
                        iteration_result.updated_test_code = current_test_code
                    
                    self.state.best_result = iteration_result
                    self.state.best_score = self._score_result(iteration_result)
                    
                    # Log final result
                    self._log_final_result(iteration_result, current_test_code)
                    
                    return self.state.best_result
                else:
                    self.logger.info("⚠ Test passes but doesn't meet all criteria. Continuing...")
            
            # Step 3: Route to analysis agents
            if self.sample_logger.log_file:
                self.sample_logger.set_step("Step 3: Analysis")
            
            analysis_results = self.coordinator.route_test_result(test_result)
            iteration_result.analysis_results = analysis_results
            
            # Log analysis results
            if self.sample_logger.log_file:
                for analysis in analysis_results:
                    self.sample_logger.log_analysis_result(
                        analysis.agent_name,
                        analysis
                    )
            
            self.logger.info(f"Step 3 Complete: Completed {len(analysis_results)} analyses")
            
            # Step 4: Synthesize instructions
            if self.sample_logger.log_file:
                self.sample_logger.set_step("Step 4: Instruction Synthesis")
            
            instructions = self.coordinator.synthesize_instructions(analysis_results)
            iteration_result.update_instructions = instructions
            
            # Log instructions
            if self.sample_logger.log_file and instructions:
                self.sample_logger.log_instructions(instructions)
            
            self.logger.info(f"Step 4 Complete: Synthesized {len(instructions)} update instructions")
            
            # Score this result and update best
            score = self._score_result(iteration_result)
            iteration_result.score = score
            
            if score > self.state.best_score:
                self.state.best_score = score
                self.state.best_result = iteration_result
                self.logger.info(f"New best result! Score: {score}")
            
            # Log iteration summary
            if self.sample_logger.log_file:
                line_coverage = test_result.test_case.coverage_info.line_coverage_percentage
                branch_coverage = test_result.test_case.coverage_info.branch_coverage_percentage
                mutation = test_result.test_case.mutation_info.kill_percentage
                self.sample_logger.log_iteration_summary(
                    iteration=iteration,
                    status=str(test_result.status.value),
                    line_coverage=line_coverage,
                    branch_coverage=branch_coverage,
                    mutation=mutation,
                    score=score
                )
            
            # Check if we've reached max iterations or have no instructions to apply
            if iteration >= max_iterations:
                self.logger.warning(f"Reached maximum iterations ({max_iterations})")
                
                # Ensure updated_test_code is set before ending
                if not iteration_result.updated_test_code:
                    iteration_result.updated_test_code = current_test_code
                
                break
            
            if not instructions:
                self.logger.info("[PASS] No improvement instructions generated. Test update complete.")
                
                # Ensure updated_test_code is set
                if not iteration_result.updated_test_code:
                    iteration_result.updated_test_code = current_test_code
                
                break
            
            # Step 5 (for next iteration): Generate updated test code
            self.logger.info("Step 5: Generating updated test code for next iteration...")
            if self.sample_logger.log_file:
                self.sample_logger.set_step("Step 5: Test Code Update")
            
            # Set updated_test_code in iteration_result BEFORE using it as previous_result
            # This ensures the previous iteration's code is available to TestUpdateAgent
            iteration_result.updated_test_code = current_test_code
            iteration_result.new_imports = current_new_imports
            
            # Get previous iteration result (current iteration)
            previous_result = iteration_result
            
            # Log context information
            if self.state.best_result:
                self.logger.info(f"  Providing best result (iteration {self.state.best_result.iteration}, score {self.state.best_score:.1f}) as reference")
            self.logger.info(f"  Providing previous result (iteration {iteration}, score {score:.1f}) for learning")
            
            # Use filtered hunks and test class context from coordinator
            update_result = self.test_updater.execute(
                test_result=test_result,
                instructions=instructions,
                is_initial=False,
                diff_hunks=self.coordinator.filtered_hunks,
                focal_method_changed=focal_method_changed,
                best_result=self.state.best_result,  # Pass best result
                previous_result=previous_result,      # Pass previous result (now with updated_test_code)
                filtered_class_fields=self.coordinator.filtered_class_fields,
                filtered_non_test_methods=self.coordinator.filtered_non_test_methods
            )
            # Generate new test code for next iteration
            current_test_code = update_result.test_code
            # Accumulate new imports: merge previous imports with new ones from this iteration
            # This ensures ErrorAnalyzeAgent knows all imports that have been added so far
            previous_new_imports = set(current_new_imports)
            new_imports_from_this_iteration = set(update_result.new_imports or [])
            # Merge: keep all previous imports and add new ones (avoid duplicates)
            current_new_imports = list(new_imports_from_this_iteration)
            
            # Log updated test code
            if self.sample_logger.log_file:
                self.sample_logger.log_test_code(current_test_code, f"UPDATED TEST CODE (Iteration {iteration})")
                if current_new_imports:
                    self.sample_logger.log_info(f"New imports to add: {current_new_imports}")
            
            self.logger.info("Step 5 Complete: Updated test code generated")
        
        # Return best result
        if self.state.best_result:
            self.logger.info(f"\nFramework completed. Best result from iteration {self.state.best_result.iteration}")
            self.logger.info(f"Best score: {self.state.best_score}")
            
            # Final safety check: ensure updated_test_code is set
            if not self.state.best_result.updated_test_code and self.state.iteration_results:
                # Try to find it from iteration results
                for iter_result in self.state.iteration_results:
                    if iter_result.iteration == self.state.best_result.iteration and iter_result.updated_test_code:
                        self.state.best_result.updated_test_code = iter_result.updated_test_code
                        break
                
                # If still not found, use current_test_code as fallback
                if not self.state.best_result.updated_test_code:
                    self.state.best_result.updated_test_code = current_test_code
            
            # Log final result
            self._log_final_result(self.state.best_result, 
                                   self.state.best_result.updated_test_code or current_test_code)
            
            return self.state.best_result
        else:
            self.logger.error("No valid result found")
            return self.state.iteration_results[0] if self.state.iteration_results else None
    
    def _log_framework_start(self, test_case: TestCase, focal_method_info: FocalMethodInfo,
                             max_iterations: int, focal_method_changed: bool):
        """Log framework start information to sample logger"""
        if not self.sample_logger.log_file:
            return
        
        self.sample_logger.log_header("FRAMEWORK CONFIGURATION")
        self.sample_logger.log_info(f"Test Case: {test_case.name}")
        self.sample_logger.log_info(f"Focal Method: {focal_method_info.name}")
        self.sample_logger.log_info(f"Focal Method Class: {focal_method_info.class_name}")
        self.sample_logger.log_info(f"Focal Method Lines: {focal_method_info.start_line} - {focal_method_info.end_line}")
        self.sample_logger.log_info(f"Focal Method Changed: {focal_method_changed}")
        self.sample_logger.log_info(f"Max Iterations: {max_iterations}")
        self.sample_logger.log_info(f"Coverage Threshold: {config.framework.coverage_threshold * 100}%")
        self.sample_logger.log_info(f"Mutation Threshold: {config.framework.mutation_threshold * 100}%")
        
        # Log focal method code
        self.sample_logger.log_subheader("FOCAL METHOD CODE (Current)")
        if self.sample_logger.logger:
            for i, line in enumerate(focal_method_info.current_code.split('\n'), 
                                     focal_method_info.start_line or 1):
                self.sample_logger.logger.info(f"  {i:4d} | {line}")
        
        if focal_method_info.original_code and focal_method_info.original_code != focal_method_info.current_code:
            self.sample_logger.log_subheader("FOCAL METHOD CODE (Original)")
            if self.sample_logger.logger:
                for i, line in enumerate(focal_method_info.original_code.split('\n'), 1):
                    self.sample_logger.logger.info(f"  {i:4d} | {line}")
    
    def _log_final_result(self, result: IterationResult, final_code: str):
        """Log final result to sample logger"""
        if not self.sample_logger.log_file:
            return
        
        line_coverage = result.test_result.test_case.coverage_info.line_coverage_percentage
        branch_coverage = result.test_result.test_case.coverage_info.branch_coverage_percentage
        mutation = result.test_result.test_case.mutation_info.kill_percentage
        
        self.sample_logger.log_final_result(
            result={
                "status": str(result.test_result.status.value),
                "line_coverage": line_coverage,
                "branch_coverage": branch_coverage,
                "mutation_kill_rate": mutation,
                "score": result.score
            },
            best_iteration=result.iteration,
            final_code=final_code
        )
        
        # Close the sample logger
        self.sample_logger.close()
    
    def _check_all_criteria_met(self, test_result: TestResultInfo) -> bool:
        """
        Check if all criteria are met:
        - Coverage meets threshold (unless disabled by ablation)
        - Mutation kill rate meets threshold (unless disabled by ablation)
        
        Ablation rules:
        - all_ablation_disable_error: Don't check any thresholds (always return True)
        - all_ablation_disable_coverage: Don't check coverage threshold
        - all_ablation_disable_mutation: Don't check mutation threshold
        """
        from config import config
        
        # If all_ablation_disable_error is True, don't check any thresholds
        if config.framework.all_ablation_disable_error:
            return True
        
        coverage = test_result.test_case.coverage_info.coverage_percentage
        mutation = test_result.test_case.mutation_info.kill_percentage
        
        coverage_threshold = config.framework.coverage_threshold * 100
        mutation_threshold = config.framework.mutation_threshold * 100
        
        # Check which criteria to evaluate based on ablation settings
        check_coverage = not config.framework.all_ablation_disable_coverage
        check_mutation = not config.framework.all_ablation_disable_mutation
        
        # Build criteria list
        criteria_checks = []
        if check_coverage:
            criteria_checks.append(coverage >= coverage_threshold)
        if check_mutation:
            criteria_checks.append(mutation >= mutation_threshold)
        
        # If all ablations are disabled, always return True (no criteria to check)
        if not criteria_checks:
            return True
        
        criteria_met = all(criteria_checks)
        
        if not criteria_met:
            self.logger.info(f"Criteria check:")
            if check_coverage:
                self.logger.info(f"  Coverage: {coverage:.1f}% (need {coverage_threshold:.1f}%)")
            if check_mutation:
                self.logger.info(f"  Mutation kill: {mutation:.1f}% (need {mutation_threshold:.1f}%)")
        
        return criteria_met
    
    def _score_result(self, iteration_result: IterationResult) -> float:
        """
        Score an iteration result using simplified priority system
        
        Higher score is better
        
        Priorities:
        1. Compilation and execution success (weight: 1000)
        2. Line coverage percentage (weight: 10) - unless disabled by ablation
        3. Branch coverage percentage (weight: 10) - unless disabled by ablation
        4. Mutation kill percentage (weight: 10) - unless disabled by ablation
        
        Ablation rules:
        - all_ablation_disable_error: Don't use coverage or mutation in scoring
        - all_ablation_disable_coverage: Don't use coverage in scoring
        - all_ablation_disable_mutation: Don't use mutation in scoring
        """
        from config import config
        
        result = iteration_result.test_result
        score = 0.0
        
        # Priority 1: Compilation and execution success
        if result.status == TestResultStatus.COMPILE_ERROR:
            return -1000  # Worst case
        
        if result.status == TestResultStatus.RUN_FAIL:
            return -100  # Bad
        
        # Everything else passes compilation and execution
        score += 1000
        
        # If all_ablation_disable_error is True, don't use coverage or mutation in scoring
        if config.framework.all_ablation_disable_error:
            return score
        
        # Priority 2 & 3: Coverage (only if not disabled by all_ablation_disable_coverage)
        if not config.framework.all_ablation_disable_coverage:
            # Line coverage (higher is better)
            line_coverage = result.test_case.coverage_info.line_coverage_percentage
            score += line_coverage * 10
            
            # Branch coverage (higher is better)
            branch_coverage = result.test_case.coverage_info.branch_coverage_percentage
            if branch_coverage is not None:
                score += branch_coverage * 10
        
        # Priority 4: Mutation kill (only if not disabled by all_ablation_disable_mutation)
        if not config.framework.all_ablation_disable_mutation:
            mutation = result.test_case.mutation_info.kill_percentage
            score += mutation * 10
        
        return score
