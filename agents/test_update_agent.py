from typing import Dict, Any, List, Optional
from models import (
    TestResultInfo, UpdateInstruction, AnalysisResult,
    DiffHunk, InputPreprocessResult, RetrievalResult, TestUpdateResult
)
from agents.base_agent import BaseAgent
import difflib

class TestUpdateAgent(BaseAgent):
    """Agent responsible for generating updated test code"""
    
    def __init__(self):
        super().__init__("TestUpdateAgent", "updater")
    
    def execute(self, test_result: TestResultInfo, 
                instructions: List[UpdateInstruction] = None,
                is_initial: bool = False,
                diff_hunks: List[DiffHunk] = None,
                root_cause_analysis: InputPreprocessResult = None,
                focal_method_changed: bool = None,
                best_result: 'IterationResult' = None,
                previous_result: 'IterationResult' = None,
                filtered_class_fields=None,
                filtered_non_test_methods=None) -> TestUpdateResult:
        """
        Generate updated test code based on update instructions
        
        Args:
            test_result: Current test result
            instructions: List of update instructions (already prioritized)
            is_initial: Whether this is the initial update (before first test execution)
            diff_hunks: List of DiffHunk objects showing code changes
            root_cause_analysis: Preprocessed input info (from InputPreprocessAgent)
            focal_method_changed: Whether the focal method itself has changed (optional)
            best_result: Best iteration result so far (for context in iterative updates)
            previous_result: Previous iteration result (to show what was tried)
            
        Returns:
            TestUpdateResult containing updated test code and new imports
        """
        # Auto-detect if not provided
        if focal_method_changed is None:
            focal_info = test_result.focal_method_info
            focal_method_changed = (focal_info.original_code and 
                                   focal_info.original_code != focal_info.current_code)
        
        if is_initial:
            return self._execute_initial_update(test_result, diff_hunks, root_cause_analysis, focal_method_changed)
        else:
            return self._execute_iterative_update(test_result, instructions, diff_hunks, focal_method_changed, 
                                                 best_result, previous_result,
                                                 filtered_class_fields, filtered_non_test_methods)
    
    def _execute_initial_update(self, test_result: TestResultInfo,
                               diff_hunks: List[DiffHunk] = None,
                               root_cause_analysis: InputPreprocessResult = None,
                               focal_method_changed: bool = False) -> TestUpdateResult:
        """
        Execute initial test code generation before first test execution
        
        This is called BEFORE we have any test execution results.
        We analyze what might be wrong based on root cause analysis and diff hunks.
        """
        self.log_info("Executing INITIAL test update (pre-execution)")
        
        # Build prompt for initial generation
        prompt = self._build_initial_prompt(test_result, diff_hunks, root_cause_analysis, focal_method_changed)
        
        # Get LLM to generate updated code
        response = self.llm_client.generate(
            prompt,
            system_prompt=self._get_initial_system_prompt()
        )
        
        # Extract test code and new imports
        test_code = self._extract_code(response)
        new_imports = self._extract_new_imports(response)
        
        self.log_info("Initial test update generation complete")
        self.log_info(f"Generated test code: {len(test_code)} chars")
        self.log_info(f"New imports (before validation): {len(new_imports)} import(s)")
        
        # Validate imports usage (only deduplicate within new_imports, no existing_imports)
        validated_imports = self._validate_imports_usage(test_code, new_imports, [])
        
        self.log_info(f"New imports (after validation): {len(validated_imports)} import(s)")
        
        return TestUpdateResult(test_code=test_code, new_imports=validated_imports)
    
    def _execute_iterative_update(self, test_result: TestResultInfo,
                                   instructions: List[UpdateInstruction],
                                   diff_hunks: List[DiffHunk] = None,
                                   focal_method_changed: bool = False,
                                   best_result: 'IterationResult' = None,
                                   previous_result: 'IterationResult' = None,
                                   filtered_class_fields=None,
                                   filtered_non_test_methods=None) -> TestUpdateResult:
        """
        Execute iterative test code update based on test execution results
        
        This is called AFTER we have test execution results and analysis.
        We make targeted improvements based on specific issues found.
        
        Args:
            test_result: Current test execution result
            instructions: Update instructions from coordinator
            diff_hunks: Code change information
            focal_method_changed: Whether focal method changed
            best_result: Best iteration so far (to provide as reference)
            previous_result: Previous iteration (to show what didn't work)
        """
        self.log_info(f"Executing ITERATIVE test update with {len(instructions)} instructions")
        
        # NOTE: Instructions no longer have priority, use them as-is
        # (CoordinatorAgent already handles any necessary ordering)
        
        # Build prompt based on test status
        prompt = self._build_iterative_prompt(test_result, instructions, diff_hunks, 
                                              focal_method_changed, best_result, previous_result,
                                              filtered_class_fields, filtered_non_test_methods)
        
        # Get LLM to generate updated code
        response = self.llm_client.generate(
            prompt,
            system_prompt=self._get_iterative_system_prompt()
        )
        
        # Extract test code and new imports
        test_code = self._extract_code(response)
        new_imports = self._extract_new_imports(response)
        
        self.log_info("Iterative test update generation complete")
        self.log_info(f"Generated test code: {len(test_code)} chars")
        self.log_info(f"New imports (before validation): {len(new_imports)} import(s)")
        
        # Validate imports usage (only deduplicate within new_imports, no existing_imports)
        validated_imports = self._validate_imports_usage(test_code, new_imports, [])
        
        self.log_info(f"New imports (after validation): {len(validated_imports)} import(s)")
        
        return TestUpdateResult(test_code=test_code, new_imports=validated_imports)
    
    def _build_initial_prompt(self, test_result: TestResultInfo,
                             diff_hunks: List[DiffHunk] = None,
                             root_cause_analysis: InputPreprocessResult = None,
                             focal_method_changed: bool = False) -> str:
        """
        Build prompt for INITIAL test update (before first execution)
        
        Uses root cause analysis to guide improvements
        """
        focal = test_result.focal_method_info
        
        # Format diff hunks
        diff_hunks_text = self._format_diff_hunks(diff_hunks) if diff_hunks else "No diff hunks provided."
        
        # Format additional context if available
        if root_cause_analysis:
            additional_context = self._format_root_cause_analysis(root_cause_analysis)
            if additional_context:
                additional_context_text = f"""
ADDITIONAL CONTEXT:
{additional_context}
"""
            else:
                additional_context_text = ""
        else:
            additional_context_text = ""
        
        # Note: We no longer show test_imports as they may be outdated from bCommit
        # The aCommit test file already has the correct imports
        
        # Build focal method section based on whether it changed
        if not focal_method_changed:
            # Focal method not changed - show with line numbers
            focal_code_formatted = self._format_focal_method_with_diff(focal, show_line_numbers=True)
            focal_method_section = f"""
⚠️ CRITICAL: THE FOCAL METHOD HAS NOT CHANGED!

The focal method has IDENTICAL implementation in both versions:
```
{focal_code_formatted}
```

Key Points:
- Do NOT try to adapt the test to "new focal method behavior" - there is none!
- The test breakage is due to changes in DEPENDENCIES or RELATED CODE
- Focus on the related code changes below for clues
- The test assertions may need updating based on changed behavior of methods 
  CALLED BY the focal method (not the focal method itself)
- Look for changes in return types, data structures, or behavior of dependencies
"""
        else:
            # Focal method changed - show with diff hunks
            focal_code_formatted = self._format_focal_method_with_diff(focal, show_line_numbers=True)
            focal_method_section = f"""
FOCAL METHOD HAS BEEN MODIFIED:
```
{focal_code_formatted}
```

The focal method itself changed, so:
- Make minimal changes to adapt test to the new behavior
- Update only what's necessary to fix compilation/execution errors
- Adjust assertions only if they cause failures

Note: Lines marked with [DELETE] show removed code, [ADD] shows added code.
Unchanged lines show line numbers from the current version.
"""
        
        # Format test class context (fields and non-test methods)
        # Use filtered versions from root_cause_analysis if available
        filtered_class_fields = None
        filtered_non_test_methods = None
        if root_cause_analysis:
            filtered_class_fields = root_cause_analysis.filtered_class_fields
            filtered_non_test_methods = root_cause_analysis.filtered_non_test_methods
        
        class_context_text = self._format_test_class_context(
            test_result.test_case,
            filtered_class_fields=filtered_class_fields,
            filtered_non_test_methods=filtered_non_test_methods
        )
        
        test_method_name = test_result.test_case.name.split('.')[-1].replace('()', '')
        
        return f"""You are updating a test case that has become OUTDATED due to code changes in the codebase.

CONTEXT:
- This test was previously working correctly on the old version of the code
- Code changes have made this test outdated or broken
- Your goal is to ADAPT the test to work with the NEW code version
- This is NOT about improving a poorly written test, but about updating it for code changes

IMPORTANT: This test has NOT been executed yet on the new code. We are generating an improved version BEFORE execution.

TEST METHOD NAME: {test_method_name}
⚠️ **CRITICAL**: You MUST keep this exact test method name. Do NOT rename it!

Original Test Code (written for OLD code version):
```java
{test_result.test_case.code}
```

{class_context_text}

{focal_method_section}

CODE CHANGES THAT AFFECTED THIS TEST:
{diff_hunks_text}

{additional_context_text}

YOUR TASK - MINIMAL UPDATE (First Iteration Only):
**[CRITICAL]** Make the MINIMUM changes necessary to ensure the test can compile and execute successfully.

IMPORTANT GUIDELINES FOR MINIMAL UPDATE:
- Make ONLY the changes required to fix compilation and execution errors
- Do NOT add new test cases or expand test coverage at this stage
- Do NOT try to improve mutation killing or coverage metrics
- Preserve as much of the original test code as possible
- Focus ONLY on making the test work with the changed code
- Keep changes minimal and targeted - avoid unnecessary modifications

⚠️ NOTE: This is the first iteration. Coverage and mutation improvements will be handled in subsequent iterations if needed.

OUTPUT FORMAT:
You MUST provide your response in the following format:

```java
// NEW_IMPORTS (if you need to add new import statements)
import com.example.NewClass;
import org.junit.Assert;
// END_NEW_IMPORTS

// TEST_CODE (the updated test method)
@Test
public void {test_method_name}() {{
    // your test code here
}}
// END_TEST_CODE
```

CRITICAL FORMATTING REQUIREMENTS:
**YOU MUST FORMAT YOUR CODE PROPERLY WITH INDENTATION AND LINE BREAKS!**
- Use proper indentation (4 spaces per level)
- Put each statement on a separate line
- Use line breaks after opening braces {{
- Use line breaks before closing braces }}
- Do NOT compress all code into a single line
- Format your code as you would in a professional IDE

Example of GOOD formatting:
```java
@Test
public void exampleTest() {{
    Scanner scanner = new Scanner();
    List<Item> items = scanner.getItems();
    
    Assert.assertEquals(3, items.size());
    Assert.assertNotNull(items.get(0));
}}
```

Example of BAD formatting (DO NOT DO THIS):
```java
@Test public void exampleTest ( ) {{ Scanner scanner = new Scanner ( ) ; List < Item > items = scanner.getItems ( ) ; Assert.assertEquals ( 3 , items.size ( ) ) ; Assert.assertNotNull ( items.get ( 0 ) ) ; }}
```

CRITICAL RULES:
1. If you DON'T need any new imports, write "// NEW_IMPORTS" followed immediately by "// END_NEW_IMPORTS" (empty section)
2. The TEST_CODE section must contain EXACTLY ONE test method (with @Test annotation)
3. **IMPORTANT**: The test method MUST be named "{test_method_name}" - do NOT change the method name!
4. **IMPORTANT**: ALL test logic must be within this SINGLE test method - do NOT create multiple @Test methods
5. If you need to test multiple scenarios, put them all in the same test method with different test cases
6. Do NOT include the entire test class, package declaration, or existing imports
7. Do NOT duplicate imports that are already in the EXISTING IMPORT STATEMENTS section above
8. **CRITICAL**: Only add imports for classes that YOUR TEST CODE directly uses. Do NOT add imports for classes used internally by the focal method (e.g., if the focal method uses JsonNode internally, but your test only calls the method, you don't need to import JsonNode)
"""
    
    def _build_iterative_prompt(self, test_result: TestResultInfo,
                                instructions: List[UpdateInstruction],
                                diff_hunks: List[DiffHunk] = None,
                                focal_method_changed: bool = False,
                                best_result: 'IterationResult' = None,
                                previous_result: 'IterationResult' = None,
                                filtered_class_fields=None,
                                filtered_non_test_methods=None) -> str:
        """
        Build prompt for ITERATIVE test update (after execution and analysis)
        
        Args:
            test_result: Current test execution result
            instructions: Update instructions from coordinator
            diff_hunks: Code change information
            focal_method_changed: Whether focal method changed
            best_result: Best iteration so far (not used in prompt, kept for compatibility)
            previous_result: Previous iteration (to show current test code for improvement)
        """
        focal = test_result.focal_method_info
        
        # Build status summary
        status_summary = self._build_status_summary(test_result)
        
        # Format diff hunks
        diff_hunks_text = self._format_diff_hunks(diff_hunks) if diff_hunks else ""
        
        # Format instructions with detailed context
        instructions_text = self._format_instructions_detailed(instructions)
        
        # Build previous result section (current test code)
        previous_result_section = ""
        
        if previous_result and previous_result.updated_test_code:
            prev_test_result = previous_result.test_result
            
            # Format new imports if any
            prev_new_imports_text = ""
            if previous_result.new_imports:
                imports_list = "\n".join(previous_result.new_imports)
                prev_new_imports_text = f"""

New Imports Added in Current Version:
```java
{imports_list}
```
"""
            
            # Check if we have annotated test code from error analysis
            test_code_to_show = previous_result.updated_test_code
            code_note = ""
            
            # Try to get annotated test code from current instructions
            for instr in instructions:
                annotated = instr.details.get('annotated_test_code', '')
                if annotated:
                    test_code_to_show = annotated
                    code_note = "\n⚠️ Error locations are marked with comments in the code below.\n"
                    break
            
            # Format coverage details - only show if test PASSED (otherwise metrics are meaningless 0/0)
            prev_cov_info = prev_test_result.test_case.coverage_info
            status_details = f"Status: {prev_test_result.status.name}\n"
            
            # Only show coverage and mutation metrics if test passed
            if prev_test_result.status.name == "PASS":
                status_details += f"- Line Coverage: {prev_cov_info.line_coverage_percentage:.1f}% ({prev_cov_info.covered_lines_count}/{prev_cov_info.total_lines} lines)\n"
                if prev_cov_info.branch_coverage_percentage is not None:
                    status_details += f"- Branch Coverage: {prev_cov_info.branch_coverage_percentage:.1f}% ({prev_cov_info.covered_branches_count}/{prev_cov_info.total_branches} branches)\n"
                status_details += f"- Mutation Kill Rate: {prev_test_result.test_case.mutation_info.kill_percentage:.1f}%"
            else:
                # Test failed - don't show 0/0 coverage metrics
                status_details = status_details.rstrip()  # Remove trailing newline
            
            previous_result_section = f"""
═══════════════════════════════════════════════════════════════
CURRENT TEST VERSION (Iteration {previous_result.iteration}, Score: {previous_result.score:.1f}):
═══════════════════════════════════════════════════════════════

This is the CURRENT test code that needs improvement based on the UPDATE INSTRUCTIONS below.{code_note}

{status_details}

Current Test Code:
```java
{test_code_to_show}
```
{prev_new_imports_text}

⚠️ YOUR TASK:
- Review the CURRENT TEST CODE above
- Follow the UPDATE INSTRUCTIONS below to improve it
- Make targeted changes to address the specific issues identified
═══════════════════════════════════════════════════════════════
"""
        
        # Note: We no longer show test_imports as they may be outdated from bCommit
        # The aCommit test file already has the correct imports
        
        # Format test class context (fields and non-test methods)
        # Use filtered versions if provided
        class_context_text = self._format_test_class_context(
            test_result.test_case,
            filtered_class_fields=filtered_class_fields,
            filtered_non_test_methods=filtered_non_test_methods
        )
        
        # Build focal method context
        if not focal_method_changed:
            # Focal method not changed - show with line numbers
            focal_code_formatted = self._format_focal_method_with_diff(focal, show_line_numbers=True)
            focal_context = f"""
⚠️ CRITICAL: THE FOCAL METHOD HAS NOT CHANGED!

The focal method has IDENTICAL implementation in both versions:
```
{focal_code_formatted}
```

Key Points:
- Do NOT try to adapt the test to "new focal method behavior" - there is none!
- The test breakage is due to changes in DEPENDENCIES or RELATED CODE
- Focus on the related code changes below for clues
- The test assertions may need updating based on changed behavior of methods 
  CALLED BY the focal method (not the focal method itself)
- Look for changes in return types, data structures, or behavior of dependencies
"""
        else:
            # Focal method changed - show with diff hunks
            focal_code_formatted = self._format_focal_method_with_diff(focal, show_line_numbers=True)
            focal_context = f"""
FOCAL METHOD HAS BEEN MODIFIED:
```
{focal_code_formatted}
```

The focal method itself changed, so:
- Adapt test assertions to the new behavior
- Cover new code paths introduced by the changes
- Adjust expected values based on the modified logic

Note: Lines marked with [DELETE] show removed code, [ADD] shows added code.
Unchanged lines show line numbers from the current version.
"""
        
        
        diff_section = f"""
CODE CHANGES:
{diff_hunks_text}
""" if diff_hunks_text else ""
        
        test_method_name = test_result.test_case.name.split('.')[-1].replace('()', '')
        
        return f"""Update the following test case based on execution results and analysis:

TEST METHOD NAME: {test_method_name}
⚠️ **CRITICAL**: You MUST keep this exact test method name. Do NOT rename it!

{previous_result_section}

{class_context_text}

{focal_context}
{diff_section}

UPDATE INSTRUCTIONS (in priority order):
{instructions_text}

YOUR TASK:
**Carefully follow the UPDATE INSTRUCTIONS above** - they contain specific, actionable guidance based on the analysis results.

IMPORTANT STRATEGY:
- Start from the CURRENT TEST CODE shown above
- Apply the specific changes suggested in UPDATE INSTRUCTIONS
- Make targeted improvements to address the identified issues
- Preserve what's working well in the current version

IMPLEMENTATION CHECKLIST:
1. Review the CURRENT TEST CODE and understand what it does
2. Review the UPDATE INSTRUCTIONS carefully - they provide specific details (line numbers, suggestions, etc.)
3. Apply the suggested changes to improve the current test
4. Ensure all necessary imports are added (check UPDATE INSTRUCTIONS for import candidates)
5. Maintain code syntax correctness
6. Add brief comments for significant changes

OUTPUT FORMAT:
You MUST provide your response in the following format:

```java
// NEW_IMPORTS (if you need to add new import statements)
import com.example.NewClass;
import org.junit.Assert;
// END_NEW_IMPORTS

// TEST_CODE (the updated test method)
@Test
public void {test_method_name}() {{
    // your test code here
}}
// END_TEST_CODE
```

CRITICAL FORMATTING REQUIREMENTS:
**YOU MUST FORMAT YOUR CODE PROPERLY WITH INDENTATION AND LINE BREAKS!**
- Use proper indentation (4 spaces per level)
- Put each statement on a separate line
- Use line breaks after opening braces {{
- Use line breaks before closing braces }}
- Do NOT compress all code into a single line
- Format your code as you would in a professional IDE

Example of GOOD formatting:
```java
@Test
public void exampleTest() {{
    Scanner scanner = new Scanner();
    List<Item> items = scanner.getItems();
    
    Assert.assertEquals(3, items.size());
    Assert.assertNotNull(items.get(0));
}}
```

Example of BAD formatting (DO NOT DO THIS):
```java
@Test public void exampleTest ( ) {{ Scanner scanner = new Scanner ( ) ; List < Item > items = scanner.getItems ( ) ; Assert.assertEquals ( 3 , items.size ( ) ) ; Assert.assertNotNull ( items.get ( 0 ) ) ; }}
```

CRITICAL RULES:
1. If you DON'T need any new imports, write "// NEW_IMPORTS" followed immediately by "// END_NEW_IMPORTS" (empty section)
2. The TEST_CODE section must contain EXACTLY ONE test method (with @Test annotation)
3. **IMPORTANT**: The test method MUST be named "{test_method_name}" - do NOT change the method name!
4. **IMPORTANT**: ALL test logic must be within this SINGLE test method - do NOT create multiple @Test methods
5. If you need to test multiple scenarios, put them all in the same test method with different test cases
6. Do NOT include the entire test class, package declaration, or existing imports
7. Do NOT duplicate imports that are already in the EXISTING IMPORT STATEMENTS section above
8. **CRITICAL**: Only add imports for classes that YOUR TEST CODE directly uses. Do NOT add imports for classes used internally by the focal method (e.g., if the focal method uses JsonNode internally, but your test only calls the method, you don't need to import JsonNode)
"""
    
    def _format_root_cause_analysis(self, analysis: InputPreprocessResult) -> str:
        """Format root cause analysis for the prompt"""
        # Note: Retrieval results are now processed by ErrorAnalyzeAgent
        # This method is kept for backward compatibility but returns empty
        return ""
    
    def _format_diff_hunks(self, diff_hunks: List[DiffHunk]) -> str:
        """Format diff hunks for inclusion in prompt"""
        if not diff_hunks:
            return "No relevant diff hunks."
        
        # Type descriptions mapping
        type_descriptions = {
            "test_method": "hunks that call methods also called by the test case",
            "focal_method": "hunks that call methods also called by the focal method (before or after changes)",
            "focal_file": "hunks in the same file as the focal method",
            "high_frequency": "hunks that appear frequently in this change"
        }
        
        formatted = []
        for i, hunk in enumerate(diff_hunks, 1):
            hunk_type = hunk.hunk_type or "unknown"
            type_desc = type_descriptions.get(hunk_type, hunk_type)
            
            formatted.append(f"""HUNK {i} [{type_desc}]:
File: {hunk.file_path}

{hunk.context}
""")
        
        return "\n".join(formatted)
    
    def _format_assertion_failure_details(self, details: Dict[str, Any]) -> str:
        """Format assertion failure details with failing code lines"""
        parts = []
        
        # Show error locations with failing code
        error_locations = details.get("error_locations", [])
        if error_locations:
            parts.append(f"   Failed assertion(s): {len(error_locations)} location(s)\n")
            for i, loc in enumerate(error_locations, 1):
                # Show the failing code (without line number in prompt, but logged with line number)
                code = loc.get("code", "")
                error_msg = loc.get("error_message", "")
                
                parts.append(f"   Assertion {i}:\n")
                parts.append(f"     Failing code: {code}\n")
                if error_msg:
                    parts.append(f"     Error: {error_msg}\n")
        
        # Show root cause if available
        root_cause = details.get("root_cause", "")
        if root_cause:
            parts.append(f"   Root cause: {root_cause}\n")
        
        # Show explanation if available
        explanation = details.get("explanation", "")
        if explanation:
            parts.append(f"   Explanation: {explanation}\n")
        
        return "".join(parts)
    
    def _format_error_analysis_details(self, details: Dict[str, Any]) -> str:
        """Format error analysis details (used by add_imports, add_mocking, add_fields, add_methods, fix_error)"""
        parts = []
        
        # Show error locations with failing code (for compilation errors)
        error_locations = details.get("error_locations", [])
        if error_locations:
            parts.append(f"   Compilation error(s): {len(error_locations)} location(s)\n")
            for i, loc in enumerate(error_locations[:5], 1):  # Show first 5
                code = loc.get("code", "")
                error_msg = loc.get("error_message", "")
                
                parts.append(f"   Error {i}:\n")
                if code:
                    parts.append(f"     Failing code: {code}\n")
                parts.append(f"     Error: {error_msg}\n")
            
            if len(error_locations) > 5:
                parts.append(f"   ... and {len(error_locations) - 5} more error(s)\n")
        
        # Show missing candidates if available
        missing_candidates = details.get("missing_candidates", [])
        if missing_candidates:
            parts.append("   Missing candidates:\n")
            for candidate in missing_candidates[:5]:  # Show first 5
                parts.append(f"     - {candidate}\n")
            if len(missing_candidates) > 5:
                parts.append(f"     ... and {len(missing_candidates) - 5} more\n")
        
        # Show root cause if available
        root_cause = details.get("root_cause", "")
        if root_cause:
            parts.append(f"   Root cause: {root_cause}\n")
        
        # Show explanation if available
        explanation = details.get("explanation", "")
        if explanation:
            parts.append(f"   Explanation: {explanation}\n")
        
        # Show retrieval targets if available
        retrieval_targets = details.get("retrieval_targets", [])
        if retrieval_targets:
            parts.append("   Suggested retrieval targets:\n")
            for target in retrieval_targets[:3]:  # Show first 3
                parts.append(f"     - {target}\n")
            if len(retrieval_targets) > 3:
                parts.append(f"     ... and {len(retrieval_targets) - 3} more\n")
        
        # Show strategies if available (for fix_error type)
        strategies = details.get("strategies", [])
        if strategies:
            parts.append(f"   Strategies: {', '.join(strategies)}\n")
        
        return "".join(parts)
    
    def _format_instructions_detailed(self, instructions: List[UpdateInstruction]) -> str:
        """Format instructions with detailed information (simplified for new structure)"""
        formatted_parts = []
        
        for i, instr in enumerate(instructions):
            # No more numbering as requested, just show instruction type
            part = f"[{instr.instruction_type}]\n"
            part += f"Reasoning: {instr.reasoning}\n"
            
            # Format details based on instruction type
            details = instr.details
            
            if instr.instruction_type == "fix_compilation_error":
                # New simplified compilation error format
                part += self._format_compilation_error_details(details)
            
            elif instr.instruction_type == "fix_assertion_failure":
                # New simplified assertion failure format
                part += self._format_assertion_failure_details_v2(details)
            
            elif instr.instruction_type == "fix_error":
                # Generic error fix
                part += self._format_general_error_details(details)
            
            elif instr.instruction_type == "improve_coverage":
                # Coverage instructions
                focus = details.get("focus")
                target_type = details.get("target_type", "line")
                difficulty = details.get("difficulty", "unknown")
                
                if focus == "uncovered_lines":
                    # New ablation mode: uncovered_lines_only
                    uncovered_lines = details.get("uncovered_lines", [])
                    part += f"Target: Cover {len(uncovered_lines)} uncovered line(s)\n"
                    part += "Uncovered lines:\n"
                    for line_info in uncovered_lines[:5]:  # Show first 5
                        if isinstance(line_info, dict):
                            line_num = line_info.get("line", "?")
                            code = line_info.get("code", "")
                            part += f"  - Line {line_num}: {code}\n"
                        else:
                            part += f"  - {line_info}\n"
                    if len(uncovered_lines) > 5:
                        part += f"  ... and {len(uncovered_lines) - 5} more\n"
                
                elif focus == "general":
                    # General coverage guidance (ablation mode)
                    part += "Target: Improve code coverage with comprehensive test scenarios\n"
                    
                    # General strategies
                    general_strategies = details.get("general_strategies", [])
                    if general_strategies:
                        part += "General Strategies:\n"
                        for strategy in general_strategies:
                            part += f"  - {strategy}\n"
                    
                    # Coverage priorities
                    coverage_priorities = details.get("coverage_priorities", [])
                    if coverage_priorities:
                        part += "\nCoverage Priorities & Strategies:\n"
                        for priority_info in coverage_priorities:
                            priority_type = priority_info.get("type", "Unknown")
                            strategy = priority_info.get("strategy", "")
                            part += f"  • {priority_type}\n"
                            part += f"    → {strategy}\n"
                    
                    # Priority actions
                    priority_actions = details.get("priority_actions", [])
                    if priority_actions:
                        part += "\nPriority Actions:\n"
                        for i, action in enumerate(priority_actions, 1):
                            part += f"  {i}. {action}\n"
                
                elif target_type == "branch":
                    branches = details.get("branches_to_cover", [])
                    part += f"Target: Cover {len(branches)} branch(es) [{difficulty} difficulty]\n"
                    part += "Branches to cover:\n"
                    for branch in branches[:5]:  # Show first 5
                        line = branch.get("line_number", "?")
                        condition = branch.get("missing_condition", "")
                        suggested = branch.get("suggested_test", "")
                        part += f"  - Line {line}: {condition}\n"
                        if suggested:
                            part += f"    Suggested: {suggested}\n"
                    if len(branches) > 5:
                        part += f"  ... and {len(branches) - 5} more\n"
                
                elif target_type == "line":
                    lines = details.get("lines_to_cover", [])
                    part += f"Target: Cover {len(lines)} line(s) [{difficulty} difficulty]\n"
                    part += "Lines to cover:\n"
                    for line_info in lines[:5]:  # Show first 5
                        if isinstance(line_info, dict):
                            line_num = line_info.get("line_number", "?")
                            reasoning = line_info.get("reasoning", "")
                            suggested = line_info.get("suggested_test", "")
                            part += f"  - Line {line_num}: {reasoning}\n"
                            if suggested:
                                part += f"    Suggested: {suggested}\n"
                        else:
                            part += f"  - {line_info}\n"
                    if len(lines) > 5:
                        part += f"  ... and {len(lines) - 5} more\n"
            
            elif instr.instruction_type == "improve_mutation_kill":
                # Mutation instructions
                focus = details.get("focus")
                
                if focus == "general":
                    # General mutation killing guidance (ablation mode)
                    part += "Target: Improve mutation killing with comprehensive testing strategies\n"
                    
                    # General strategies
                    general_strategies = details.get("general_strategies", [])
                    if general_strategies:
                        part += "General Strategies:\n"
                        for strategy in general_strategies:
                            part += f"  - {strategy}\n"
                    
                    # Common mutation types and how to kill them
                    common_mutations = details.get("common_mutation_types", [])
                    if common_mutations:
                        part += "\nCommon Mutation Types & Strategies:\n"
                        for mut_info in common_mutations:
                            mut_type = mut_info.get("type", "Unknown")
                            strategy = mut_info.get("strategy", "")
                            part += f"  • {mut_type}\n"
                            part += f"    → {strategy}\n"
                    
                    # Priority actions
                    priority_actions = details.get("priority_actions", [])
                    if priority_actions:
                        part += "\nPriority Actions:\n"
                        for i, action in enumerate(priority_actions, 1):
                            part += f"  {i}. {action}\n"
                
                elif focus == "assertion_strength":
                    part += "Target: Strengthen existing assertions\n"
                    test_improvements = details.get("test_improvements", [])
                    if test_improvements:
                        part += "Improvements:\n"
                        for imp in test_improvements[:3]:
                            part += f"  - {imp}\n"
                
                elif focus == "boundary_cases":
                    part += "Target: Add boundary case tests\n"
                    boundary_tests = details.get("boundary_tests", [])
                    if boundary_tests:
                        part += "Boundary cases:\n"
                        for boundary in boundary_tests[:3]:
                            part += f"  - {boundary}\n"
                
                else:
                    # Survived mutations
                    survived = details.get("survived_mutations", [])
                    difficulty = details.get("difficulty", "unknown")
                    
                    if survived:
                        part += f"Target: Kill {len(survived)} survived mutation(s) [{difficulty} difficulty]\n"
                        part += "Survived mutations:\n"
                        for mut in survived[:5]:  # Show first 5
                            line = mut.get("line_number", "?")
                            mutator = mut.get("mutator", "Unknown")
                            suggested = mut.get("suggested_assertion", "")
                            reasoning = mut.get("reasoning", "")
                            
                            part += f"  - Line {line} [{mutator}]\n"
                            if suggested:
                                part += f"    Suggested assertion: {suggested}\n"
                            if reasoning:
                                part += f"    Reasoning: {reasoning}\n"
                        
                        if len(survived) > 5:
                            part += f"  ... and {len(survived) - 5} more\n"
                    
                    mutation_types = details.get("mutation_types_summary", [])
                    if mutation_types:
                        part += "Mutation types summary:\n"
                        for mut_type in mutation_types[:3]:
                            if isinstance(mut_type, dict):
                                mutator = mut_type.get("mutator_type", "Unknown")
                                count = mut_type.get("count", 0)
                                strategy = mut_type.get("general_strategy", "")
                                part += f"  - {mutator} ({count} mutations): {strategy}\n"
                    
                    test_improvements = details.get("test_improvements", [])
                    if test_improvements:
                        part += "Test improvements:\n"
                        for imp in test_improvements[:3]:
                            part += f"  - {imp}\n"
            
            else:
                # Generic format for other instruction types
                part += f"Details: {details}\n"
            
            formatted_parts.append(part)
        
        return "\n".join(formatted_parts)
    
    def _build_status_summary(self, test_result: TestResultInfo) -> str:
        """Build a detailed summary of test execution status"""
        focal = test_result.focal_method_info
        
        status_text = f"Status: {test_result.status}\n"
        
        # Compilation/Execution status only
        if test_result.status.value == "compile_error":
            status_text += f"""
COMPILATION ERROR:
{test_result.error_message}

This test FAILS TO COMPILE. The following issues need to be fixed:
"""
            # Add key details from raw error output if available
            if test_result.raw_error_output:
                status_text += self._extract_key_compile_errors(test_result.raw_error_output)
        elif test_result.status.value == "run_fail":
            status_text += f"""
TEST EXECUTION FAILURE:
{test_result.error_message}

This test FAILS AT RUNTIME. The following issues need to be fixed:
"""
        elif test_result.status.value == "pass":
            status_text += "[PASS] Test PASSES compilation and execution\n"
        
        return status_text
    
    def _get_initial_system_prompt(self) -> str:
        """Get system prompt for INITIAL test update"""
        return """You are an expert test engineer specializing in minimal test adaptation to code changes.

Your goal is to make the MINIMUM changes necessary to ensure the test can compile and execute successfully.

Key Context:
- The test was working correctly on the previous version of the code
- Code changes have made it outdated or broken
- Your task is to make MINIMAL changes to fix compilation and execution issues ONLY
- This is the FIRST iteration - do NOT try to improve coverage or mutation killing yet

Key Responsibilities (MINIMAL UPDATE):
1. **Fix Compilation Errors**: Ensure the test compiles successfully
2. **Fix Execution Errors**: Ensure the test runs without runtime exceptions
3. **Preserve Original Test**: Keep as much of the original test code as possible
4. **Minimal Changes Only**: Do NOT add new test cases or expand coverage

Approach:
- Analyze what changed in the code
- Make ONLY the changes required to fix compilation/execution
- Do NOT add new test cases or assertions
- Do NOT try to improve test quality at this stage
- Keep the test as close to the original as possible

**CRITICAL CONSTRAINTS**:
1. You must generate EXACTLY ONE test method. All test logic must be within this single method.
2. Do NOT create multiple @Test methods. If you need to test multiple scenarios, include them all in one test method.
3. You MUST keep the original test method name - do NOT rename it under any circumstances.
4. **MINIMAL CHANGES ONLY** - Do not expand or improve the test beyond what's necessary to make it work.

Always return complete, working test code with minimal modifications."""
    
    def _get_iterative_system_prompt(self) -> str:
        """Get system prompt for ITERATIVE test update"""
        return """You are an expert test engineer specializing in fixing broken tests through iterative improvements.

Your goal is to improve the CURRENT test code based on detailed update instructions and analysis results.

IMPORTANT CONTEXT:
- You will be shown the CURRENT test code that needs improvement
- You will be given specific UPDATE INSTRUCTIONS on what to fix/improve
- Your task is to apply these instructions to improve the current test

Key responsibilities based on issue type:

COMPILATION ERRORS:
- Fix syntax errors and missing imports
- Add missing methods or fields that the test depends on
- Use mocking if needed for unavailable dependencies

EXECUTION FAILURES:
- Fix runtime errors and exceptions
- Correct assertion logic
- Handle edge cases properly

COVERAGE ISSUES:
- Add test cases to cover uncovered lines
- Trigger different code paths and branches
- Add assertions that verify behavior

MUTATION KILLING ISSUES:
- Strengthen existing assertions
- Add boundary condition tests
- Use multiple assertions to catch subtle bugs

STRATEGIC APPROACH:
1. **Understand Current Code**: Carefully review the current test code provided
2. **Follow Instructions**: Apply the specific improvements suggested in UPDATE INSTRUCTIONS
3. **Targeted Changes**: Make specific improvements based on the identified issues
4. **Preserve What Works**: Keep the parts of the current test that are working well
5. **Incremental Progress**: Build upon the current version, don't rewrite from scratch

**CRITICAL CONSTRAINTS**:
1. You must generate EXACTLY ONE test method. All test logic must be within this single method.
2. Do NOT create multiple @Test methods. If you need to test multiple scenarios, include them all in one test method.
3. You MUST keep the original test method name - do NOT rename it under any circumstances.

Always:
1. Return complete, working test code
2. Maintain consistency with the original test style
3. Add comments explaining significant changes
4. Test only the focal method"""
    
    def _extract_code(self, response: str) -> str:
        """
        Extract code from LLM response.
        
        New format expects:
        // NEW_IMPORTS
        import statements...
        // END_NEW_IMPORTS
        
        // TEST_CODE
        test method code...
        // END_TEST_CODE
        
        Returns only the TEST_CODE section (the test method).
        New imports are stored separately and will be handled by JavaTestExecutor.
        """
        import re
        
        # Try to extract code from markdown code blocks first
        code_match = re.search(r'```(?:java)?\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            content = code_match.group(1)
        else:
            # If no code block found, use the response as-is
            content = response
        
        # Extract TEST_CODE section
        test_code_match = re.search(r'//\s*TEST_CODE.*?\n(.*?)//\s*END_TEST_CODE', content, re.DOTALL)
        if test_code_match:
            return test_code_match.group(1).strip()
        
        # Fallback: if no TEST_CODE markers, return the content as-is
        # (for backward compatibility with old format)
        return content.strip()
    
    def _extract_new_imports(self, response: str) -> List[str]:
        """
        Extract new import statements from LLM response.
        
        Returns:
            List of new import statements (e.g., ["import com.example.NewClass;"])
        """
        import re
        
        # Try to extract code from markdown code blocks first
        code_match = re.search(r'```(?:java)?\n(.*?)\n```', response, re.DOTALL)
        if code_match:
            content = code_match.group(1)
        else:
            content = response
        
        # Extract NEW_IMPORTS section
        imports_match = re.search(r'//\s*NEW_IMPORTS.*?\n(.*?)//\s*END_NEW_IMPORTS', content, re.DOTALL)
        if imports_match:
            imports_text = imports_match.group(1).strip()
            if not imports_text:
                return []
            
            # Parse import statements
            import_lines = []
            for line in imports_text.split('\n'):
                line = line.strip()
                # Skip comments and empty lines
                if line and not line.startswith('//') and line.startswith('import '):
                    import_lines.append(line)
            
            return import_lines
        
        # No NEW_IMPORTS section found
        return []
    
    def _validate_imports_usage(self, test_code: str, new_imports: List[str], existing_imports: List[str]) -> List[str]:
        """
        验证新生成的 imports 是否真的在测试代码中被使用。
        同时检查已有的 imports 和新生成的 imports。
        
        支持：
        1. 普通导入: import java.util.List;
        2. 通配符导入: import java.util.*;
        3. 内部类导入: import com.example.Outer.Inner;
        4. 静态导入: import static org.junit.Assert.*;
        5. 完全限定名使用
        
        Args:
            test_code: 生成的测试代码
            new_imports: 新生成的 import 语句列表
            existing_imports: 已有的 import 语句列表
            
        Returns:
            过滤后的新 imports 列表（只包含真正被使用的）
        """
        import re
        
        def parse_import(import_statement: str) -> dict:
            """
            解析 import 语句，返回详细信息
            
            Returns:
                {
                    'type': 'normal' | 'wildcard' | 'static' | 'static_wildcard',
                    'package': 'java.util',
                    'class_name': 'List' or None (for wildcard),
                    'full_path': 'java.util.List'
                }
            """
            import_statement = import_statement.strip()
            
            # 移除末尾的分号
            if import_statement.endswith(';'):
                import_statement = import_statement[:-1].strip()
            
            # 静态导入通配符: import static org.junit.Assert.*;
            match = re.match(r'import\s+static\s+([\w.]+)\.\*', import_statement)
            if match:
                return {
                    'type': 'static_wildcard',
                    'package': match.group(1),
                    'class_name': None,
                    'full_path': match.group(1)
                }
            
            # 静态导入: import static org.junit.Assert.assertEquals;
            match = re.match(r'import\s+static\s+([\w.]+)\.(\w+)', import_statement)
            if match:
                return {
                    'type': 'static',
                    'package': match.group(1),
                    'class_name': match.group(2),
                    'full_path': match.group(1) + '.' + match.group(2)
                }
            
            # 通配符导入: import java.util.*;
            match = re.match(r'import\s+([\w.]+)\.\*', import_statement)
            if match:
                return {
                    'type': 'wildcard',
                    'package': match.group(1),
                    'class_name': None,
                    'full_path': match.group(1)
                }
            
            # 普通导入: import java.util.List; 或 import com.example.Outer.Inner;
            match = re.match(r'import\s+([\w.]+)\.(\w+)', import_statement)
            if match:
                full_package = match.group(1)
                class_name = match.group(2)
                return {
                    'type': 'normal',
                    'package': full_package,
                    'class_name': class_name,
                    'full_path': full_package + '.' + class_name
                }
            
            return None
        
        def is_import_used(import_info: dict, code: str) -> bool:
            """检查 import 是否在代码中被使用"""
            if not import_info:
                return False
            
            import_type = import_info['type']
            
            # 通配符导入：检查包名是否以完全限定名形式出现
            # 例如：import java.util.*; 检查代码中是否有 java.util.List 这样的用法
            if import_type == 'wildcard':
                package = import_info['package']
                # 检查是否有完全限定名使用该包
                pattern = r'\b' + re.escape(package) + r'\.\w+'
                if re.search(pattern, code):
                    return True
                # 通配符导入很难验证，保守起见保留
                # 因为无法知道代码中使用了哪个具体的类
                return True
            
            # 静态通配符导入：保守保留
            if import_type == 'static_wildcard':
                return True
            
            # 普通导入和静态导入：检查类名/方法名是否出现
            class_name = import_info['class_name']
            if not class_name:
                return False
            
            # 1. 检查简单类名是否作为独立标识符出现
            pattern = r'\b' + re.escape(class_name) + r'\b'
            if re.search(pattern, code):
                return True
            
            # 2. 检查完全限定名是否出现
            full_path = import_info['full_path']
            if full_path in code:
                return True
            
            # 3. 对于内部类，检查外部类.内部类的形式
            # 例如：import com.example.Outer.Inner;
            # 代码中可能写成 Outer.Inner
            if '.' in import_info['package']:
                parts = import_info['package'].split('.')
                # 尝试匹配 OuterClass.InnerClass 的形式
                if len(parts) >= 2:
                    outer_class = parts[-1]
                    inner_class = class_name
                    pattern = r'\b' + re.escape(outer_class) + r'\.' + re.escape(inner_class) + r'\b'
                    if re.search(pattern, code):
                        return True
            
            return False
        
        # 验证新生成的 imports
        validated_new_imports = []
        removed_imports = []
        
        for import_stmt in new_imports:
            import_info = parse_import(import_stmt)
            if import_info and is_import_used(import_info, test_code):
                validated_new_imports.append(import_stmt)
            else:
                removed_imports.append(import_stmt)
                self.log_info(f"Removed unused import: {import_stmt}")
        
        # 同时检查已有的 imports，记录哪些可能未被使用（仅用于日志，不实际移除）
        unused_existing_imports = []
        for import_stmt in existing_imports:
            import_info = parse_import(import_stmt)
            if import_info and not is_import_used(import_info, test_code):
                unused_existing_imports.append(import_stmt)
        
        if unused_existing_imports:
            self.log_info(f"Note: {len(unused_existing_imports)} existing import(s) appear unused in test code")
            for imp in unused_existing_imports[:3]:  # 只显示前3个
                self.log_info(f"  - {imp}")
        
        # 统计信息
        if removed_imports:
            self.log_info(f"Import validation: Removed {len(removed_imports)} unused import(s) from {len(new_imports)} new import(s)")
        else:
            self.log_info(f"Import validation: All {len(new_imports)} new import(s) are used in test code")
        
        return validated_new_imports
    
    def _extract_key_compile_errors(self, raw_error_output: str) -> str:
        """
        Extract key compilation error details from Maven output.
        
        Directly extracts the complete error section including symbol and location details.
        
        Args:
            raw_error_output: Full Maven compilation error output
            
        Returns:
            Complete error details with symbol and location information
        """
        lines = raw_error_output.split('\n')
        error_lines = []
        
        # Extract all lines that contain error information
        # This includes [ERROR] lines and indented detail lines (symbol, location, etc.)
        in_error_section = False
        for i, line in enumerate(lines):
            # Start of error section
            if '[ERROR] COMPILATION ERROR' in line or 'COMPILATION ERROR' in line:
                in_error_section = True
                continue
            
            # Skip BUILD FAILURE and help messages
            if any(skip in line for skip in ['BUILD FAILURE', 'Failed to execute goal', '-> [Help', 'To see the full', 'Re-run Maven', 'For more information', 'After correcting']):
                continue
            
            # In error section, capture [ERROR] lines and indented detail lines
            if in_error_section:
                # [ERROR] lines or indented lines (symbol, location, etc.)
                if line.strip().startswith('[ERROR]') or (line.startswith('  ') and line.strip()):
                    error_lines.append(line)
                # Stop at [INFO] lines that indicate end of error details
                elif line.strip().startswith('[INFO]') and 'error' in line.lower():
                    break
        
        if error_lines:
            result = "\nCOMPILATION ERROR DETAILS:\n"
            for line in error_lines:
                result += f"{line}\n"
            return result
        
        return "\n(No detailed error information available)\n"
    
    def _format_test_class_context(self, test_case, 
                                   filtered_class_fields=None, 
                                   filtered_non_test_methods=None) -> str:
        """
        Format test class context including class fields and non-test methods.
        
        Args:
            test_case: TestCase object containing class_fields and non_test_methods
            filtered_class_fields: Optional filtered list of class fields to use (if None, uses test_case.class_fields)
            filtered_non_test_methods: Optional filtered list of non-test methods to use (if None, uses test_case.non_test_methods)
            
        Returns:
            Formatted string showing test class context
        """
        # Use filtered versions if provided, otherwise fall back to original
        class_fields = filtered_class_fields if filtered_class_fields is not None else test_case.class_fields
        non_test_methods = filtered_non_test_methods if filtered_non_test_methods is not None else test_case.non_test_methods
        
        if not class_fields and not non_test_methods:
            return ""
        
        sections = []
        
        # Format class fields
        if class_fields:
            fields_text = "\n".join(f"  {field}" for field in class_fields)
            sections.append(f"""TEST CLASS FIELDS:
```java
{fields_text}
```

These are class-level fields available to the test method. You can reference them directly in your test code.""")
        
        # Format non-test methods (show all methods' code)
        if non_test_methods:
            methods_codes = []
            for method in non_test_methods:  # Show all methods
                code = method.get('code', '')
                if code:
                    methods_codes.append(code)
            
            methods_text = "\n\n".join(methods_codes)
            sections.append(f"""TEST CLASS HELPER METHODS:
```java
{methods_text}
```

These are non-test methods in the same test class. They may include:
- @Before/@BeforeEach methods that set up test fixtures
- @After/@AfterEach methods that clean up
- Helper methods that can be called from your test
Pay attention to what these methods do, as they affect the test execution context.""")
        
        return "\n\n".join(sections) + "\n"
    
    def _format_focal_method_with_diff(self, focal, show_line_numbers: bool = True) -> str:
        """
        Format focal method with diff hunks if modified, otherwise just show with line numbers.
        
        Args:
            focal: FocalMethodInfo object
            show_line_numbers: Whether to show line numbers for current code
            
        Returns:
            Formatted string showing focal method with diff or line numbers
        """
        # Check if focal method was modified
        is_modified = (focal.original_code and 
                      focal.original_code != focal.current_code)
        
        if not is_modified:
            # Focal method not modified - just show current code with line numbers
            if show_line_numbers and focal.start_line > 0:
                lines = focal.current_code.split('\n')
                numbered_lines = []
                for i, line in enumerate(lines):
                    line_num = focal.start_line + i
                    numbered_lines.append(f"{line_num:6d} | {line}")
                return '\n'.join(numbered_lines)
            else:
                return focal.current_code
        
        # Focal method was modified - generate unified diff
        original_lines = focal.original_code.split('\n') if focal.original_code else []
        current_lines = focal.current_code.split('\n')
        
        # Generate unified diff
        diff = difflib.unified_diff(
            original_lines,
            current_lines,
            lineterm='',
            n=3  # context lines
        )
        
        diff_lines = list(diff)
        
        # If diff is too complex or empty, fall back to showing both versions
        if not diff_lines or len(diff_lines) > len(current_lines) + 10:
            # Fall back to simple line-by-line diff
            result = []
            
            # Use simple line-by-line comparison
            # Track line number for CURRENT version only
            for i, (orig_line, curr_line) in enumerate(zip(original_lines, current_lines)):
                line_num = focal.start_line + i if focal.start_line > 0 else i + 1
                if orig_line != curr_line:
                    # Show [DELETE] without line number (it's from old version)
                    # Show [ADD] with line number (it's from current version)
                    result.append(f"       | [DELETE] {orig_line}")
                    result.append(f"{line_num:6d} | [ADD] {curr_line}")
                else:
                    result.append(f"{line_num:6d} | {curr_line}")
            
            # Handle extra lines in original (deleted lines)
            if len(original_lines) > len(current_lines):
                for i in range(len(current_lines), len(original_lines)):
                    # No line number for deleted lines
                    result.append(f"       | [DELETE] {original_lines[i]}")
            
            # Handle extra lines in current (added lines)
            if len(current_lines) > len(original_lines):
                for i in range(len(original_lines), len(current_lines)):
                    line_num = focal.start_line + i if focal.start_line > 0 else i + 1
                    result.append(f"{line_num:6d} | [ADD] {current_lines[i]}")
            
            return '\n'.join(result)
        
        # Build the result with current code and inline diffs
        result = []
        current_line_num = focal.start_line if focal.start_line > 0 else 1
        
        # Parse the unified diff
        i = 0
        while i < len(diff_lines):
            line = diff_lines[i]
            
            # Skip header lines (---, +++, @@)
            if line.startswith('---') or line.startswith('+++'):
                i += 1
                continue
            
            # Hunk header (@@ -a,b +c,d @@)
            if line.startswith('@@'):
                # Extract starting line number from hunk header
                parts = line.split('+')[1].split(',')[0].strip()
                try:
                    current_line_num = int(parts)
                    if focal.start_line > 0:
                        current_line_num += focal.start_line - 1
                except:
                    pass
                i += 1
                continue
            
            # Context line (unchanged)
            if line.startswith(' '):
                result.append(f"{current_line_num:6d} | {line[1:]}")
                current_line_num += 1
                i += 1
                continue
            
            # Changed lines - group consecutive changes
            if line.startswith('-') or line.startswith('+'):
                deleted_lines = []
                added_lines = []
                
                # Collect all consecutive deletions
                while i < len(diff_lines) and diff_lines[i].startswith('-'):
                    deleted_lines.append(diff_lines[i][1:])
                    i += 1
                
                # Collect all consecutive additions
                while i < len(diff_lines) and diff_lines[i].startswith('+'):
                    added_lines.append(diff_lines[i][1:])
                    i += 1
                
                # Format the changes
                # IMPORTANT: Only current version lines get line numbers
                # [DELETE] lines are from old version, so no line number
                # [ADD] lines and unchanged lines are from current version, so they get line numbers
                if deleted_lines and added_lines:
                    # Both deletion and addition - show as modification
                    if len(deleted_lines) == 1 and len(added_lines) == 1:
                        # Single line change - show both on one line for brevity
                        result.append(f"       | [DELETE] {deleted_lines[0]}")
                        result.append(f"{current_line_num:6d} | [ADD] {added_lines[0]}")
                        current_line_num += 1
                    else:
                        # Multi-line change
                        for del_line in deleted_lines:
                            result.append(f"       | [DELETE] {del_line}")
                        for add_line in added_lines:
                            result.append(f"{current_line_num:6d} | [ADD] {add_line}")
                            current_line_num += 1
                elif deleted_lines:
                    # Only deletions - no line numbers
                    for del_line in deleted_lines:
                        result.append(f"       | [DELETE] {del_line}")
                elif added_lines:
                    # Only additions - with line numbers
                    for add_line in added_lines:
                        result.append(f"{current_line_num:6d} | [ADD] {add_line}")
                        current_line_num += 1
                
                continue
            
            i += 1
        
        return '\n'.join(result)
    
    def _format_compilation_error_details(self, details: Dict[str, Any]) -> str:
        """Format compilation error details from new AnalysisResult structure"""
        parts = []
        
        error_type = details.get('error_type', 'unknown')
        parts.append(f"Error Type: {error_type}\n")
        
        # Show known symbols (need imports)
        known_symbols = details.get('known_symbols', [])
        if known_symbols:
            parts.append(f"Known Symbols (need imports): {', '.join(known_symbols)}\n")
        
        # Show unknown symbols only (retrieval results already processed by ErrorAnalyzeAgent)
        unknown_symbols = details.get('unknown_symbols', [])
        
        if unknown_symbols:
            parts.append(f"Unknown Symbols (project-specific): {', '.join(unknown_symbols)}\n")
        
        # Show root cause and explanation
        root_cause = details.get('root_cause', '')
        if root_cause:
            parts.append(f"\nRoot Cause: {root_cause}\n")
        
        explanation = details.get('explanation', '')
        if explanation:
            parts.append(f"\nExplanation:\n{explanation}\n")
        
        return "".join(parts)
    
    def _format_assertion_failure_details_v2(self, details: Dict[str, Any]) -> str:
        """Format assertion failure details from new AnalysisResult structure"""
        parts = []
        
        error_type = details.get('error_type', 'assertion_failure')
        parts.append(f"Error Type: {error_type}\n")
        
        # Note: Retrieval results already processed by ErrorAnalyzeAgent, no need to show here
        
        # Show root cause and explanation
        root_cause = details.get('root_cause', '')
        if root_cause:
            parts.append(f"\nRoot Cause: {root_cause}\n")
        
        explanation = details.get('explanation', '')
        if explanation:
            parts.append(f"\nExplanation:\n{explanation}\n")
        
        return "".join(parts)
    
    def _format_general_error_details(self, details: Dict[str, Any]) -> str:
        """Format general error details"""
        parts = []
        
        error_type = details.get('error_type', 'unknown')
        parts.append(f"Error Type: {error_type}\n")
        
        root_cause = details.get('root_cause', '')
        if root_cause:
            parts.append(f"Root Cause: {root_cause}\n")
        
        explanation = details.get('explanation', '')
        if explanation:
            parts.append(f"Explanation:\n{explanation}\n")
        
        return "".join(parts)
