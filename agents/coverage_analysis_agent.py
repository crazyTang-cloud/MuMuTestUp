from typing import Dict, Any, List
from models import AnalysisResult, TestResultInfo
from agents.base_agent import BaseAgent
from config import config
from java_test_executor import generate_coverage_annotated_code

class CoverageAnalysisAgent(BaseAgent):
    """Agent for analyzing and improving coverage"""
    
    def __init__(self, lines_only: bool = False):
        super().__init__("CoverageAnalysisAgent", "analyzer")
        self.lines_only = lines_only  # If True, only process line coverage, skip branch coverage
    
    def execute(self, test_result: TestResultInfo) -> AnalysisResult:
        """
        Analyze coverage loss and provide improvement strategies
        
        Requirements:
        - Overall coverage should meet the threshold set in config
        
        Args:
            test_result: The test result with coverage loss
            
        Returns:
            AnalysisResult with coverage analysis and strategies
        """
        self.log_info("Analyzing coverage loss")
        
        analysis = AnalysisResult(
            agent_name=self.name,
            analysis_type="coverage"
        )
        
        # Get overall coverage info
        coverage_info = test_result.test_case.coverage_info
        current_coverage = coverage_info.coverage_percentage
        threshold = config.framework.coverage_threshold * 100  # Convert to percentage
        
        # Extract uncovered lines from detailed coverage
        uncovered_lines = self._get_uncovered_lines(test_result)
        uncovered_branches = [] if self.lines_only else self._get_uncovered_branches(test_result)
        
        self.log_info(f"Current coverage: {current_coverage:.1f}%, Threshold: {threshold:.1f}%")
        self.log_info(f"Uncovered lines: {uncovered_lines}")
        if not self.lines_only:
            self.log_info(f"Uncovered branches: {uncovered_branches}")
        
        # Generate annotated focal method code
        annotated_code = self._generate_annotated_focal_code(test_result)
        
        # Build prompt for LLM
        prompt = self._build_analysis_prompt(
            test_result, uncovered_lines, uncovered_branches, 
            current_coverage, threshold, annotated_code
        )
        
        # Get LLM analysis
        response = self.llm_client.generate(prompt, system_prompt=self._get_system_prompt())
        
        # Parse response
        analysis.recommendations = self._parse_llm_response(response, test_result)
        
        # If lines_only mode, remove branch_improvements from recommendations
        if self.lines_only and analysis.recommendations:
            analysis.recommendations.pop("branch_improvements", None)
        
        analysis.strategies = self._extract_strategies(response)
        analysis.metadata = {
            "uncovered_lines": uncovered_lines,
            "uncovered_branches": uncovered_branches if not self.lines_only else [],
            "current_coverage": current_coverage,
            "target_coverage": threshold,
            "annotated_focal_code": annotated_code,
            "priority": "HIGH"
        }
        
        self.log_info(f"Coverage analysis complete. Strategies: {analysis.strategies}")
        
        return analysis
    
    def _get_uncovered_lines(self, test_result: TestResultInfo) -> List[int]:
        """Get uncovered lines from detailed coverage info"""
        coverage_info = test_result.test_case.coverage_info
        
        # Try to use detailed coverage first (focal method specific)
        if coverage_info.detailed_coverage:
            return coverage_info.detailed_coverage.uncovered_lines
        
        # Fallback to overall coverage
        all_lines = set(range(1, coverage_info.total_lines + 1))
        covered_lines = set(coverage_info.covered_lines)
        uncovered = list(all_lines - covered_lines)
        return sorted(uncovered)
    
    def _get_uncovered_branches(self, test_result: TestResultInfo) -> List[Dict[str, Any]]:
        """Get uncovered branches from detailed coverage info"""
        coverage_info = test_result.test_case.coverage_info
        
        if not coverage_info.detailed_coverage:
            return []
        
        uncovered_branches = []
        for line_cov in coverage_info.detailed_coverage.line_coverages:
            if line_cov.has_branch and line_cov.missed_branches > 0:
                uncovered_branches.append({
                    "line": line_cov.line_number,
                    "covered_branches": line_cov.covered_branches,
                    "missed_branches": line_cov.missed_branches,
                    "total_branches": line_cov.covered_branches + line_cov.missed_branches
                })
        
        return uncovered_branches
    
    def _generate_annotated_focal_code(self, test_result: TestResultInfo) -> str:
        """Generate focal method code with coverage annotations"""
        focal_method_info = test_result.focal_method_info
        coverage_info = test_result.test_case.coverage_info
        
        if not coverage_info.detailed_coverage:
            # No detailed coverage, return original code with note
            return f"// Note: No detailed coverage data available\n{focal_method_info.current_code}"
        
        return generate_coverage_annotated_code(
            focal_code=focal_method_info.current_code,
            start_line=focal_method_info.start_line,
            detailed_coverage=coverage_info.detailed_coverage
        )
    
    def _build_analysis_prompt(self, test_result: TestResultInfo, 
                               uncovered_lines: List[int],
                               uncovered_branches: List[Dict[str, Any]],
                               current_coverage: float,
                               threshold: float,
                               annotated_code: str) -> str:
        """Build prompt for coverage analysis with annotated code"""
        
        if self.lines_only:
            # Lines-only mode: focus only on line coverage, ignore branches
            return f"""
Analyze the coverage loss in this test case (LINE COVERAGE ONLY):

Test Case: {test_result.test_case.name}
Focal Method: {test_result.focal_method_info.name}

## Focal Method Code with Coverage Annotations
(✅ = covered, ❌ = not covered)

```java
{annotated_code}
```

## Current Test Code
```java
{test_result.test_case.code}
```

## Coverage Status
- Current coverage: {current_coverage:.1f}%
- Target coverage: {threshold:.1f}%
- Gap: {threshold - current_coverage:.1f}%
- Uncovered lines: {uncovered_lines}

## Analysis Tasks
Based on the annotated code above, please analyze:
1. Which uncovered lines (marked with ❌) are easiest to cover?
2. Which uncovered lines require moderate effort?
3. What test conditions or paths would cover the uncovered lines?
4. What specific assertions or method calls would help?
5. Prioritize ALL line improvements by difficulty level

CRITICAL CONSTRAINTS:
- You can ONLY suggest modifications to the TEST code
- You can ONLY suggest adding new test cases, test logic, or imports
- You CANNOT suggest modifying the focal method or any production code
- Focus on HOW to write tests that exercise uncovered lines
- IGNORE branch coverage - only focus on line coverage

CRITICAL RULES:
1. **Line Focus**: Only analyze uncovered lines, do not consider branches
2. **Line Difficulty**: Distinguish between:
   - Easy lines: Simple method calls, straightforward logic, basic operations
   - Moderate lines: Require specific setup, multiple conditions, or complex state
   - Hard lines: Exception handling, edge cases, complex dependencies
3. **Definitive Suggestions**: Provide CONCRETE and DEFINITIVE test suggestions, not conditional ones
   - GOOD: "Call method with null argument and verify result"
   - BAD: "If X is true, call method; otherwise try Y; if both fail, do Z"

Format response as JSON with keys:
- easy_to_cover: list of {{line_number, reasoning, suggested_test}}
- moderate_difficulty: list of {{line_number, reasoning}}
- hard_to_cover: list of {{line_number, reasoning}}
- test_strategies: list of suggested test improvements

NOTE: Do NOT include branch_improvements in your response.
"""
        else:
            # Normal mode: include branch coverage
            # Format uncovered branches info
            branch_info = ""
            if uncovered_branches:
                branch_lines = []
                for b in uncovered_branches:
                    branch_lines.append(f"  - Line {b['line']}: {b['covered_branches']}/{b['total_branches']} branches covered")
                branch_info = "\n".join(branch_lines)
            else:
                branch_info = "  No branch coverage data available"
            
            return f"""
Analyze the coverage loss in this test case:

Test Case: {test_result.test_case.name}
Focal Method: {test_result.focal_method_info.name}

## Focal Method Code with Coverage Annotations
(✅ = covered, ❌ = not covered, ⚠️ = partial branch coverage)

```java
{annotated_code}
```

## Current Test Code
```java
{test_result.test_case.code}
```

## Coverage Status
- Current coverage: {current_coverage:.1f}%
- Target coverage: {threshold:.1f}%
- Gap: {threshold - current_coverage:.1f}%
- Uncovered lines: {uncovered_lines}

## Branch Coverage Details
{branch_info}

## Analysis Tasks
Based on the annotated code above, please analyze:
1. Which uncovered branches (marked with ⚠️) are easiest to cover? (e.g., simple if-null checks, boolean conditions)
2. Which branches require moderate effort? (e.g., mocking, exception handling, complex setup)
3. Which uncovered lines (marked with ❌) are easiest to cover?
4. What test conditions or paths would cover the uncovered code?
5. What specific assertions or method calls would help?
6. Prioritize ALL improvements by difficulty level

CRITICAL CONSTRAINTS:
- You can ONLY suggest modifications to the TEST code
- You can ONLY suggest adding new test cases, test logic, or imports
- You CANNOT suggest modifying the focal method or any production code
- Focus on HOW to write tests that exercise uncovered code paths

CRITICAL RULES:
1. **Branch Priority**: Branches are ALWAYS more important than lines. Focus on branches first.
2. **Control Dependencies**: If a line is uncovered BECAUSE its controlling branch is uncovered, 
   DO NOT include that line in easy_to_cover or moderate_difficulty lists. 
   Only include the controlling branch in branch_improvements.
   Example: If line 50 is inside "if (x != null)" block and the branch at line 48 is uncovered,
   only report the branch at line 48, NOT line 50.
3. **Branch Difficulty**: Distinguish between:
   - Easy branches: Simple conditions (null checks, boolean flags, simple comparisons)
   - Moderate/Hard branches: Require mocking, exception handling, complex state setup
4. **Definitive Suggestions**: Provide CONCRETE and DEFINITIVE test suggestions, not conditional ones
   - GOOD: "Call method with null argument and verify result"
   - BAD: "If X is true, call method; otherwise try Y; if both fail, do Z"

Format response as JSON with keys:
- easy_to_cover: list of {{line_number, reasoning, suggested_test}} (for lines NOT controlled by uncovered branches)
- branch_improvements: list of {{line_number, missing_condition, suggested_test, difficulty: "easy"|"moderate"}}
- moderate_difficulty: list of {{line_number, reasoning}} (for lines NOT controlled by uncovered branches)
- hard_to_cover: list of {{line_number, reasoning}} (for lines NOT controlled by uncovered branches)
- test_strategies: list of suggested test improvements
"""
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for coverage analysis"""
        return """You are an expert in code coverage analysis. You understand test design 
and can identify what test conditions and code paths are needed to achieve high coverage. 
You know how to write effective test cases that cover edge cases and branch conditions.

When analyzing the annotated code:
- Lines marked with ✅ are already covered by the test
- Lines marked with ❌ need test coverage
- Lines marked with ⚠️ have partial branch coverage and need additional test conditions
- Focus on providing actionable suggestions that can improve coverage"""
    
    def _parse_llm_response(self, response: str, test_result: TestResultInfo) -> Dict[str, Any]:
        """Parse coverage analysis response"""
        try:
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "easy_to_cover": data.get("easy_to_cover", []),
                    "branch_improvements": data.get("branch_improvements", []),
                    "moderate_difficulty": data.get("moderate_difficulty", []),
                    "hard_to_cover": data.get("hard_to_cover", []),
                    "test_strategies": data.get("test_strategies", []),
                    "full_response": response
                }
        except Exception as e:
            self.log_error(f"Failed to parse coverage analysis: {e}")
        
        return {
            "analysis": response,
            "test_strategies": [],
            "full_response": response
        }
    
    def _extract_strategies(self, response: str) -> List[str]:
        """Extract actionable strategies"""
        strategies = []
        
        if "assertion" in response.lower():
            strategies.append("add_assertions")
        if "condition" in response.lower() or "branch" in response.lower():
            strategies.append("add_conditional_paths")
        if "loop" in response.lower() or "iterate" in response.lower():
            strategies.append("add_loop_coverage")
        if "exception" in response.lower() or "error" in response.lower():
            strategies.append("add_error_handling")
        if "null" in response.lower():
            strategies.append("add_null_checks")
        
        return strategies if strategies else ["improve_test_paths"]
