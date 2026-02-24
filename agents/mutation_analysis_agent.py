from typing import Dict, Any, List
from models import AnalysisResult, TestResultInfo, MutationDetail
from agents.base_agent import BaseAgent
from config import config
from java_test_executor import generate_mutation_annotated_code

class MutationAnalysisAgent(BaseAgent):
    """Agent for analyzing and improving mutation killing capability"""
    
    def __init__(self):
        super().__init__("MutationAnalysisAgent", "analyzer")
    
    def execute(self, test_result: TestResultInfo) -> AnalysisResult:
        """
        Analyze mutation loss and provide improvement strategies
        
        Requirements:
        - Overall mutation kill rate should meet the threshold set in config
        
        Args:
            test_result: The test result with mutation loss
            
        Returns:
            AnalysisResult with mutation analysis and strategies
        """
        self.log_info("Analyzing mutation loss")
        
        analysis = AnalysisResult(
            agent_name=self.name,
            analysis_type="mutation"
        )
        
        # Get overall mutation info
        mutation_info = test_result.test_case.mutation_info
        current_kill_rate = mutation_info.kill_percentage
        threshold = config.framework.mutation_threshold * 100  # Convert to percentage
        
        # Extract unkilled mutations with details
        survived_mutations = self._get_survived_mutations(test_result)
        no_coverage_mutations = self._get_no_coverage_mutations(test_result)
        
        self.log_info(f"Current kill rate: {current_kill_rate:.1f}%, Threshold: {threshold:.1f}%")
        self.log_info(f"Survived mutations: {len(survived_mutations)}, No coverage: {len(no_coverage_mutations)}")
        
        # Generate annotated focal method code
        annotated_code = self._generate_annotated_focal_code(test_result)
        
        # Build prompt for LLM
        prompt = self._build_analysis_prompt(
            test_result, survived_mutations, no_coverage_mutations,
            current_kill_rate, threshold, annotated_code
        )
        
        # Get LLM analysis
        response = self.llm_client.generate(prompt, system_prompt=self._get_system_prompt())
        
        # Parse response
        analysis.recommendations = self._parse_llm_response(response, test_result)
        analysis.strategies = self._extract_strategies(response)
        analysis.metadata = {
            "survived_mutations": [self._mutation_to_dict(m) for m in survived_mutations],
            "no_coverage_mutations": [self._mutation_to_dict(m) for m in no_coverage_mutations],
            "total_mutations": mutation_info.total_mutations,
            "current_kill_rate": current_kill_rate,
            "target_kill_rate": threshold,
            "annotated_focal_code": annotated_code,
            "priority": "HIGH"
        }
        
        self.log_info(f"Mutation analysis complete. Strategies: {analysis.strategies}")
        
        return analysis
    
    def _mutation_to_dict(self, mutation: MutationDetail) -> Dict[str, Any]:
        """Convert MutationDetail to dictionary"""
        return {
            "mutation_id": mutation.mutation_id,
            "line_number": mutation.line_number,
            "mutator": mutation.mutator_simple_name,
            "description": mutation.description,
            "status": mutation.status
        }
    
    def _get_survived_mutations(self, test_result: TestResultInfo) -> List[MutationDetail]:
        """Get survived mutations from detailed mutation info"""
        mutation_info = test_result.test_case.mutation_info
        
        if mutation_info.detailed_mutations:
            return mutation_info.detailed_mutations.survived_mutations
        
        return []
    
    def _get_no_coverage_mutations(self, test_result: TestResultInfo) -> List[MutationDetail]:
        """Get mutations with no coverage from detailed mutation info"""
        mutation_info = test_result.test_case.mutation_info
        
        if mutation_info.detailed_mutations:
            return mutation_info.detailed_mutations.no_coverage_mutations
        
        return []
    
    def _generate_annotated_focal_code(self, test_result: TestResultInfo) -> str:
        """Generate focal method code with mutation annotations"""
        focal_method_info = test_result.focal_method_info
        mutation_info = test_result.test_case.mutation_info
        
        if not mutation_info.detailed_mutations:
            # No detailed mutations, return original code with note
            return f"// Note: No detailed mutation data available\n{focal_method_info.current_code}"
        
        return generate_mutation_annotated_code(
            focal_code=focal_method_info.current_code,
            start_line=focal_method_info.start_line,
            detailed_mutations=mutation_info.detailed_mutations
        )
    
    def _build_analysis_prompt(self, test_result: TestResultInfo,
                               survived_mutations: List[MutationDetail],
                               no_coverage_mutations: List[MutationDetail],
                               current_kill_rate: float,
                               threshold: float,
                               annotated_code: str) -> str:
        """Build prompt for mutation analysis with annotated code"""
        
        # Format survived mutations info
        survived_info = ""
        if survived_mutations:
            survived_lines = []
            for m in survived_mutations:
                survived_lines.append(
                    f"  - Line {m.line_number}: [{m.mutator_simple_name}] {m.description}"
                )
            survived_info = "\n".join(survived_lines)
        else:
            survived_info = "  No survived mutations"
        
        # Format no-coverage mutations info
        no_coverage_info = ""
        if no_coverage_mutations:
            no_cov_lines = []
            for m in no_coverage_mutations:
                no_cov_lines.append(
                    f"  - Line {m.line_number}: [{m.mutator_simple_name}] {m.description}"
                )
            no_coverage_info = "\n".join(no_cov_lines)
        else:
            no_coverage_info = "  No uncovered mutations"
        
        return f"""
Analyze the mutation killing capability of this test case:

Test Case: {test_result.test_case.name}
Focal Method: {test_result.focal_method_info.name}

## Focal Method Code with Mutation Annotations
(🟢 KILLED = mutation killed, 🔴 SURVIVED = mutation not killed, ⚪ NO_COVERAGE = line not covered)

```java
{annotated_code}
```

## Current Test Code
```java
{test_result.test_case.code}
```

## Mutation Status
- Current kill rate: {current_kill_rate:.1f}%
- Target kill rate: {threshold:.1f}%
- Gap: {threshold - current_kill_rate:.1f}%
- Total mutations: {test_result.test_case.mutation_info.total_mutations}

## Survived Mutations (🔴 - need stronger assertions)
{survived_info}

## No Coverage Mutations (⚪ - need test coverage first)
{no_coverage_info}

CRITICAL CONSTRAINTS:
- You can ONLY suggest modifications to the TEST code
- You can ONLY suggest adding assertions, test cases, or test logic
- You CANNOT suggest modifying the focal method or any production code
- Focus on HOW to write assertions that detect the mutations

## Analysis Tasks
Based on the annotated code above, please analyze:
1. For each 🔴 SURVIVED mutation, what SPECIFIC assertion would kill it?
2. For each ⚪ NO_COVERAGE mutation, what test path would cover it?
3. What common mutation types are surviving and why?
4. What boundary conditions or edge cases are missing?
5. Provide CONCRETE and DEFINITIVE assertion suggestions (not conditional ones)

## Guidelines for Suggestions:
- Provide SPECIFIC assertions that can be directly added to the test
- GOOD: "Add assertion: assertEquals(5, result.getValue())"
- BAD: "If result is null, check X; otherwise check Y; if both fail, try Z"
- Each suggestion should be a single actionable step

## Mutation Types Reference
Common mutators and how to kill them:
- VoidMethodCallMutator: Verify the side effects of the removed method call
- BooleanFalseReturnValsMutator/BooleanTrueReturnValsMutator: Assert the expected boolean value
- NullReturnValsMutator: Assert return value is not null, or verify its properties
- ConditionalsBoundaryMutator: Test boundary values (e.g., x == 5 vs x > 5)
- NegateConditionalsMutator: Test both true and false branches of conditions
- MathMutator: Verify exact numeric results of calculations

Format response as JSON with keys:
- survived_mutation_fixes: list of {{line_number, mutator, suggested_assertion, reasoning}}
- no_coverage_fixes: list of {{line_number, mutator, suggested_test_path}}
- mutation_types_summary: list of {{mutator_type, count, general_strategy}}
- boundary_tests: list of boundary conditions to add
- test_improvements: list of specific test improvements
"""
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for mutation analysis"""
        return """You are an expert in mutation testing and assertion design. You understand 
how different types of mutations work and what assertions are needed to detect them. 

Key principles:
- Mutations marked 🔴 SURVIVED were executed but not detected - need stronger assertions
- Mutations marked ⚪ NO_COVERAGE were not even executed - need test paths to cover them
- Each mutation type has specific killing strategies
- Provide CONCRETE and DEFINITIVE assertions, not conditional suggestions
- Each suggestion should be directly applicable to the test code
- Avoid listing multiple if-else branches in your suggestions

You know how to write effective test cases that can detect subtle code changes 
introduced by mutation testing. Focus on providing specific assertions that can be 
immediately added to the test code."""
    
    def _parse_llm_response(self, response: str, test_result: TestResultInfo) -> Dict[str, Any]:
        """Parse mutation analysis response"""
        try:
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "survived_mutation_fixes": data.get("survived_mutation_fixes", []),
                    "no_coverage_fixes": data.get("no_coverage_fixes", []),
                    "mutation_types_summary": data.get("mutation_types_summary", []),
                    "boundary_tests": data.get("boundary_tests", []),
                    "test_improvements": data.get("test_improvements", []),
                    "full_response": response
                }
        except Exception as e:
            self.log_error(f"Failed to parse mutation analysis: {e}")
        
        return {
            "analysis": response,
            "test_improvements": [],
            "full_response": response
        }
    
    def _extract_strategies(self, response: str) -> List[str]:
        """Extract actionable strategies"""
        strategies = []
        
        if "assertion" in response.lower():
            strategies.append("strengthen_assertions")
        if "boundary" in response.lower() or "edge case" in response.lower():
            strategies.append("add_boundary_tests")
        if "null" in response.lower() or "empty" in response.lower():
            strategies.append("add_null_checks")
        if "condition" in response.lower():
            strategies.append("add_conditional_assertions")
        if "return" in response.lower():
            strategies.append("verify_return_values")
        if "side effect" in response.lower() or "method call" in response.lower():
            strategies.append("verify_side_effects")
        
        return strategies if strategies else ["improve_assertions"]
