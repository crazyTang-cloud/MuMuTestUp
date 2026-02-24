from typing import List, Dict, Any, Optional
from models import (
    TestCase, FocalMethodInfo, DiffHunk, 
    InputPreprocessResult, RetrievalResult
)
from agents.base_agent import BaseAgent
from agents.retrieval_agent import RetrievalAgent
import difflib


class InputPreprocessAgent(BaseAgent):
    """
    Agent for filtering relevant hunks and determining if retrieval is needed.
    
    This agent no longer analyzes root causes, but instead:
    1. Filters diff hunks to keep only relevant ones
    2. Determines if additional information retrieval is needed
    3. Calls retrieval system if necessary
    """
    
    def __init__(self, retrieval_agent: Optional[RetrievalAgent] = None):
        super().__init__("InputPreprocessAgent", "analyzer")
        self.retrieval_agent = retrieval_agent
    
    def execute(self, test_case: TestCase, 
                focal_method_info: FocalMethodInfo,
                diff_hunks: List[DiffHunk],
                focal_method_changed: bool = None) -> InputPreprocessResult:
        """
        Filter relevant hunks and determine if retrieval is needed.
        
        Args:
            test_case: The test case that needs updating
            focal_method_info: Information about the focal method (before and after)
            diff_hunks: List of diff hunks with types (test_method, focal_method, focal_file, high_frequency)
            focal_method_changed: Whether the focal method itself has changed (optional)
            
        Returns:
            InputPreprocessResult with filtered hunks, class fields, non-test methods, and retrieval results
        """
        self.log_info("Filtering relevant hunks and determining retrieval needs")
        self.log_info(f"Focal method: {focal_method_info.name}")
        self.log_info(f"Total diff hunks: {len(diff_hunks)}")
        
        # Determine if focal method changed
        if focal_method_changed is None:
            focal_method_changed = (focal_method_info.original_code and 
                                   focal_method_info.original_code != focal_method_info.current_code)
        
        self.log_info(f"Focal method changed: {focal_method_changed}")
        
        # Log hunk types distribution
        hunk_types = {}
        for hunk in diff_hunks:
            hunk_type = hunk.hunk_type or "unknown"
            hunk_types[hunk_type] = hunk_types.get(hunk_type, 0) + 1
        self.log_info(f"Hunk types distribution: {hunk_types}")
        
        # Step 1: Filter all relevant information (hunks, class fields, non-test methods) in one unified LLM call
        self.log_info("Filtering all relevant information (hunks, class fields, non-test methods) in one pass")
        
        filtered_hunks, filtered_class_fields, filtered_non_test_methods = self._filter_all_context(
            test_case, focal_method_info, diff_hunks, focal_method_changed
        )
        
        self.log_info(f"Filtered to {len(filtered_hunks)} relevant hunks")
        self.log_info(f"Filtered to {len(filtered_class_fields) if filtered_class_fields else 0} relevant class fields")
        self.log_info(f"Filtered to {len(filtered_non_test_methods) if filtered_non_test_methods else 0} relevant non-test methods")
        
        # Step 3: Determine if retrieval is needed
        # NOTE: Retrieval is temporarily disabled for InputPreprocessAgent
        needs_retrieval = False  # Disabled
        # needs_retrieval = self._determine_retrieval_need(
        #     test_case, focal_method_info, filtered_hunks, focal_method_changed
        # )
        
        self.log_info(f"Retrieval needed: {needs_retrieval} (disabled)")
        
        # Step 4: Perform retrieval if needed
        # NOTE: Retrieval is temporarily disabled for InputPreprocessAgent
        retrieval_result = None
        # if needs_retrieval and self.retrieval_agent:
        #     self.log_info("Performing retrieval...")
        #     retrieval_result = self.retrieval_agent.retrieve_for_root_cause_analysis(
        #         test_code=test_case.code,
        #         focal_method_info=focal_method_info,
        #         filtered_hunks=filtered_hunks,
        #         focal_method_changed=focal_method_changed
        #     )
        #     
        #     if retrieval_result.retrieval_successful:
        #         self.log_info(f"Retrieval successful: {len(retrieval_result.retrieved_methods)} methods, "
        #                     f"{len(retrieval_result.retrieved_fields)} fields")
        #     else:
        #         self.log_info(f"Retrieval unsuccessful: {retrieval_result.retrieval_reasoning}")
        # elif needs_retrieval and not self.retrieval_agent:
        #     self.log_warning("Retrieval needed but no retrieval agent available")
        
        # Create result
        result = InputPreprocessResult(
            filtered_hunks=filtered_hunks,
            retrieval_result=retrieval_result,
            reasoning=self._build_reasoning(filtered_hunks, needs_retrieval, retrieval_result, 
                                           filtered_class_fields, filtered_non_test_methods),
            needs_retrieval=needs_retrieval,
            filtered_class_fields=filtered_class_fields,
            filtered_non_test_methods=filtered_non_test_methods
        )
        
        return result
    
    def _filter_all_context(self, test_case: TestCase,
                           focal_method_info: FocalMethodInfo,
                           diff_hunks: List[DiffHunk],
                           focal_method_changed: bool) -> tuple[List[DiffHunk], Optional[List[str]], Optional[List[Dict[str, Any]]]]:
        """
        Filter all relevant context (diff hunks, class fields, non-test methods) in one unified LLM call.
        
        Args:
            test_case: The test case
            focal_method_info: Focal method information
            diff_hunks: All diff hunks with types
            focal_method_changed: Whether focal method changed
            
        Returns:
            Tuple of (filtered_hunks, filtered_class_fields, filtered_non_test_methods)
        """
        # Pre-filter high_frequency hunks to top 5 by frequency
        pre_filtered_hunks = self._pre_filter_high_frequency_hunks(diff_hunks, top_k=5) if diff_hunks else []
        
        # Build unified prompt for LLM to filter all context
        prompt = self._build_unified_filter_prompt(
            test_case, focal_method_info, pre_filtered_hunks, focal_method_changed
        )
        
        # Get LLM analysis
        response = self.llm_client.generate(
            prompt,
            system_prompt=self._get_unified_filter_system_prompt()
        )
        
        # Parse response to get all filtered results
        filtered_hunks, filtered_fields, filtered_methods = self._parse_unified_filter_response(
            response, pre_filtered_hunks, test_case.class_fields, test_case.non_test_methods
        )
        
        return filtered_hunks, filtered_fields, filtered_methods
    
    def _build_unified_filter_prompt(self, test_case: TestCase,
                                    focal_method_info: FocalMethodInfo,
                                    diff_hunks: List[DiffHunk],
                                    focal_method_changed: bool) -> str:
        """Build unified prompt for filtering all context information"""
        
        # Format focal method section
        focal_code_formatted = self._format_focal_method_with_diff(focal_method_info, show_line_numbers=True)
        
        if focal_method_changed:
            focal_section = f"""
**Focal Method (CHANGED):**
```java
{focal_code_formatted}
```

Note: Lines marked with [DELETE] show removed code, [ADD] shows added code.
Unchanged lines show line numbers from the current version.
"""
        else:
            focal_section = f"""
**Focal Method (UNCHANGED):**
```java
{focal_code_formatted}
```
"""
        
        # Format diff hunks with indices
        hunks_text = ""
        if diff_hunks:
            for i, hunk in enumerate(diff_hunks):
                hunk_type = hunk.hunk_type or "unknown"
                freq_info = f"\nFrequency: {hunk.frequency}" if hunk.hunk_type == "high_frequency" else ""
                hunks_text += f"""
**Hunk {i} [{hunk_type}]:**
File: {hunk.file_path}{freq_info}
```
{hunk.context}
```
"""
        else:
            hunks_text = "No diff hunks provided."
        
        # Format class fields with indices
        fields_text = ""
        if test_case.class_fields:
            for i, field in enumerate(test_case.class_fields):
                fields_text += f"Field {i}: {field}\n"
        else:
            fields_text = "No class fields."
        
        # Format non-test methods with indices
        methods_text = ""
        if test_case.non_test_methods:
            for i, method in enumerate(test_case.non_test_methods):
                method_name = method.get('name', f'method_{i}')
                method_code = method.get('code', '')
                methods_text += f"""
**Method {i} ({method_name}):**
```java
{method_code}
```
"""
        else:
            methods_text = "No non-test methods."
        
        return f"""Analyze and filter which information is truly relevant for updating the following test case.

**Test Case:**
```java
{test_case.code}
```

{focal_section}

**Diff Hunks (Code Changes):**
{hunks_text}

**Test Class Fields:**
{fields_text}

**Test Class Non-Test Methods (Helper Methods, @Before/@After, etc.):**
{methods_text}

**Task:**
Analyze ALL THREE categories of information above and determine which items are truly relevant for updating this test case.

**Filtering Criteria:**

For **Diff Hunks**:
1. Does the hunk affect the test's behavior or assertions?
2. Does the hunk change methods/classes that the test depends on?
3. Is the hunk related to the focal method or its dependencies?
4. Does the hunk introduce new functionality that should be tested?

For **Test Class Fields**:
1. Are they referenced by the test method?
2. Are they related to the focal method or code changes?
3. Do they provide necessary context for understanding the test?
4. Exclude inner class definitions or unrelated data structures that just clutter the context

For **Test Class Non-Test Methods**:
1. Are they called by the test method?
2. Are they helper methods that the test uses?
3. Are they @Before/@After methods that affect test execution?
4. Exclude unrelated helper methods that the test doesn't use

**Hunk Types Explanation:**
- **test_method**: Hunks that call the same methods as the test
- **focal_method**: Hunks that call the same methods as the focal method
- **focal_file**: Hunks in the same file as the focal method
- **high_frequency**: Frequently occurring hunks across the codebase

Return a JSON object with:
- relevant_hunk_indices: list of hunk indices (0-based) that are relevant
- relevant_field_indices: list of field indices (0-based) that are relevant (empty list means none are relevant)
- relevant_method_indices: list of method indices (0-based) that are relevant (empty list means none are relevant)
- reasoning: brief explanation for your filtering decisions

Example:
{{
  "relevant_hunk_indices": [0, 2],
  "relevant_field_indices": [1],
  "relevant_method_indices": [0],
  "reasoning": "Hunk 0 changes a method called by the test. Hunk 2 is in the same file as focal method. Field 1 is referenced in test. Method 0 is a @Before setup method. Other items are not relevant."
}}
"""
    
    def _get_unified_filter_system_prompt(self) -> str:
        """Get system prompt for unified filtering"""
        return """You are an expert at analyzing code changes and test code to determine which information is relevant for test updates.

Your task is to filter THREE categories of information:
1. Diff hunks (code changes in the codebase)
2. Test class fields (class-level variables)
3. Test class non-test methods (helper methods, setup/teardown methods)

Key principles:
1. Focus on what the test ACTUALLY uses or depends on
2. Include changes that affect the test's behavior or assertions
3. Include helper methods and fields that are referenced by the test
4. Exclude unrelated changes, inner class definitions, and unused helpers
5. Be conservative - when in doubt about relevance, include it rather than exclude it
6. Consider the hunk type as a hint, but make your own judgment

Always return valid JSON with the specified format."""
    
    def _parse_unified_filter_response(self, response: str,
                                      diff_hunks: List[DiffHunk],
                                      class_fields: List[str],
                                      non_test_methods: List[Dict[str, Any]]) -> tuple[List[DiffHunk], Optional[List[str]], Optional[List[Dict[str, Any]]]]:
        """Parse LLM response to extract all filtered results"""
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Get indices
                hunk_indices = data.get("relevant_hunk_indices", [])
                field_indices = data.get("relevant_field_indices", [])
                method_indices = data.get("relevant_method_indices", [])
                
                # Filter hunks
                filtered_hunks = []
                if diff_hunks:
                    valid_hunk_indices = [i for i in hunk_indices if isinstance(i, int) and 0 <= i < len(diff_hunks)]
                    filtered_hunks = [diff_hunks[i] for i in valid_hunk_indices]
                
                # Filter fields
                filtered_fields = None
                if class_fields:
                    valid_field_indices = [i for i in field_indices if isinstance(i, int) and 0 <= i < len(class_fields)]
                    if valid_field_indices:
                        filtered_fields = [class_fields[i] for i in valid_field_indices]
                    elif field_indices is not None and len(field_indices) == 0:
                        # Explicitly empty list means no relevant fields
                        filtered_fields = []
                    # If parsing fails or invalid indices, return None (use all fields)
                
                # Filter methods
                filtered_methods = None
                if non_test_methods:
                    valid_method_indices = [i for i in method_indices if isinstance(i, int) and 0 <= i < len(non_test_methods)]
                    if valid_method_indices:
                        filtered_methods = [non_test_methods[i] for i in valid_method_indices]
                    elif method_indices is not None and len(method_indices) == 0:
                        # Explicitly empty list means no relevant methods
                        filtered_methods = []
                    # If parsing fails or invalid indices, return None (use all methods)
                
                return filtered_hunks, filtered_fields, filtered_methods
        except Exception as e:
            self.log_error(f"Failed to parse unified filter response: {e}")
        
        # Fallback: return all items (no filtering)
        return diff_hunks, None, None
    
    def _filter_relevant_hunks(self, test_case: TestCase,
                               focal_method_info: FocalMethodInfo,
                               diff_hunks: List[DiffHunk],
                               focal_method_changed: bool) -> List[DiffHunk]:
        """
        Filter diff hunks to keep only relevant ones using LLM.
        
        Args:
            test_case: The test case
            focal_method_info: Focal method information
            diff_hunks: All diff hunks with types
            focal_method_changed: Whether focal method changed
            
        Returns:
            List of filtered relevant hunks
        """
        if not diff_hunks:
            return []
        
        # Pre-filter high_frequency hunks to top 5 by relevance_score
        filtered_diff_hunks = self._pre_filter_high_frequency_hunks(diff_hunks, top_k=5)
        
        # Build prompt for LLM to filter hunks
        prompt = self._build_filter_prompt(
            test_case, focal_method_info, filtered_diff_hunks, focal_method_changed
        )
        
        # Get LLM analysis
        response = self.llm_client.generate(
            prompt,
            system_prompt=self._get_filter_system_prompt()
        )
        
        # Parse response to get relevant hunk indices
        relevant_indices = self._parse_filter_response(response, len(filtered_diff_hunks))
        
        # Filter hunks
        filtered = [filtered_diff_hunks[i] for i in relevant_indices if 0 <= i < len(filtered_diff_hunks)]
        
        return filtered
    
    def _pre_filter_high_frequency_hunks(self, diff_hunks: List[DiffHunk], top_k: int = 5) -> List[DiffHunk]:
        """
        Pre-filter high_frequency hunks to keep only top K by frequency.
        Other hunk types (test_method, focal_method, focal_file) are kept as-is.
        
        Args:
            diff_hunks: All diff hunks with types
            top_k: Number of top high_frequency hunks to keep (default: 5)
            
        Returns:
            List of hunks with high_frequency hunks filtered to top K
        """
        # Separate high_frequency hunks from others
        high_frequency_hunks = []
        other_hunks = []
        
        for hunk in diff_hunks:
            if hunk.hunk_type == "high_frequency":
                high_frequency_hunks.append(hunk)
            else:
                other_hunks.append(hunk)
        
        # If there are high_frequency hunks, filter to top K by frequency
        if high_frequency_hunks:
            original_count = len(high_frequency_hunks)
            # Sort by frequency (descending) and keep top K
            high_frequency_hunks.sort(key=lambda h: h.frequency, reverse=True)
            high_frequency_hunks = high_frequency_hunks[:top_k]
            
            self.log_info(f"Pre-filtered high_frequency hunks: {original_count} -> {len(high_frequency_hunks)} (top {top_k} by frequency)")
        
        # Combine and return
        return other_hunks + high_frequency_hunks
    
    def _determine_retrieval_need(self, test_case: TestCase,
                                  focal_method_info: FocalMethodInfo,
                                  filtered_hunks: List[DiffHunk],
                                  focal_method_changed: bool) -> bool:
        """
        Determine if additional information retrieval is needed.
        
        Args:
            test_case: The test case
            focal_method_info: Focal method information
            filtered_hunks: Filtered relevant hunks
            focal_method_changed: Whether focal method changed
            
        Returns:
            True if retrieval is needed, False otherwise
        """
        # Build prompt for LLM to determine retrieval need
        prompt = self._build_retrieval_decision_prompt(
            test_case, focal_method_info, filtered_hunks, focal_method_changed
        )
        
        # Get LLM decision
        response = self.llm_client.generate(
            prompt,
            system_prompt=self._get_retrieval_decision_system_prompt()
        )
        
        # Parse decision
        needs_retrieval = self._parse_retrieval_decision(response)
        
        return needs_retrieval
    
    def _build_filter_prompt(self, test_case: TestCase,
                            focal_method_info: FocalMethodInfo,
                            diff_hunks: List[DiffHunk],
                            focal_method_changed: bool) -> str:
        """Build prompt for filtering relevant hunks"""
        
        # Format focal method section
        focal_code_formatted = self._format_focal_method_with_diff(focal_method_info, show_line_numbers=True)
        
        if focal_method_changed:
            focal_section = f"""
**Focal Method (CHANGED):**
```java
{focal_code_formatted}
```

Note: Lines marked with [DELETE] show removed code, [ADD] shows added code.
Unchanged lines show line numbers from the current version.
"""
        else:
            focal_section = f"""
**Focal Method (UNCHANGED):**
```java
{focal_code_formatted}
```
"""
        
        # Format hunks with indices
        hunks_text = ""
        for i, hunk in enumerate(diff_hunks):
            hunk_type = hunk.hunk_type or "unknown"
            # Show frequency for high_frequency hunks
            freq_info = f"\nFrequency: {hunk.frequency}" if hunk.hunk_type == "high_frequency" else ""
            hunks_text += f"""
**Hunk {i} [{hunk_type}]:**
File: {hunk.file_path}{freq_info}
```
{hunk.context}
```
"""
        
        # Format test class context (fields and non-test methods)
        class_context_text = self._format_test_class_context(test_case)
        
        return f"""Analyze which diff hunks are relevant for updating the following test case.

**Test Case:**
```java
{test_case.code}
```

{class_context_text}

{focal_section}

**Diff Hunks (with types):**
{hunks_text}

**Task:**
Determine which hunks are truly relevant for updating this test. Consider:
1. Does the hunk affect the test's behavior or assertions?
2. Does the hunk change methods/classes that the test depends on?
3. Is the hunk related to the focal method or its dependencies?
4. Does the hunk introduce new functionality that should be tested?

**Hunk Types Explanation:**
- **test_method**: Hunks that call the same methods as the test
- **focal_method**: Hunks that call the same methods as the focal method
- **focal_file**: Hunks in the same file as the focal method
- **high_frequency**: Frequently occurring hunks across the codebase

Return a JSON object with:
- relevant_indices: list of hunk indices (0-based) that are relevant
- reasoning: explanation for each selected/rejected hunk

Example:
{{
  "relevant_indices": [0, 2, 5],
  "reasoning": "Hunk 0 is relevant because..., Hunk 2 is relevant because..., Hunk 5 is relevant because..."
}}
"""
    
    def _build_retrieval_decision_prompt(self, test_case: TestCase,
                                        focal_method_info: FocalMethodInfo,
                                        filtered_hunks: List[DiffHunk],
                                        focal_method_changed: bool) -> str:
        """Build prompt for determining retrieval need"""
        
        # Format focal method section
        focal_code_formatted = self._format_focal_method_with_diff(focal_method_info, show_line_numbers=True)
        
        if focal_method_changed:
            focal_section = f"""
**Focal Method (CHANGED):**
```java
{focal_code_formatted}
```

Note: Lines marked with [DELETE] show removed code, [ADD] shows added code.
Unchanged lines show line numbers from the current version.
"""
        else:
            focal_section = f"""
**Focal Method (UNCHANGED):**
```java
{focal_code_formatted}
```
"""
        
        # Format filtered hunks
        hunks_text = ""
        for i, hunk in enumerate(filtered_hunks):
            hunk_type = hunk.hunk_type or "unknown"
            hunks_text += f"""
**Relevant Hunk {i} [{hunk_type}]:**
File: {hunk.file_path}
```
{hunk.context}
```
"""
        
        # Format test class context (fields and non-test methods)
        class_context_text = self._format_test_class_context(test_case)
        
        return f"""Determine if additional information retrieval is needed to update this test case.

**Test Case:**
```java
{test_case.code}
```

{class_context_text}

{focal_section}

**Relevant Code Changes:**
{hunks_text if hunks_text else "No relevant hunks identified."}

**Task:**
Analyze if the test case, focal method, and relevant hunks provide enough information to update the test, or if we need to retrieve additional methods/fields/classes from the codebase.

Consider:
1. Are there missing imports or undefined classes/methods?
2. Do we need to understand how certain methods work?
3. Are there dependencies that are not shown in the hunks?
4. Do we need examples of how to use certain APIs?

Return a JSON object with:
- needs_retrieval: boolean (true if retrieval is needed)
- reasoning: detailed explanation of why retrieval is/isn't needed
- what_to_retrieve: list of specific things to look for (if needs_retrieval is true)

Example:
{{
  "needs_retrieval": true,
  "reasoning": "The test references class 'Foo' which is not imported and not shown in the hunks...",
  "what_to_retrieve": ["class Foo definition", "method bar() signature", "field BAZ value"]
}}
"""
    
    def _get_filter_system_prompt(self) -> str:
        """Get system prompt for hunk filtering"""
        return """You are an expert at analyzing code changes and determining which changes are relevant for test updates.

Your task is to filter diff hunks to identify only those that are truly relevant for updating a test case.

Key principles:
1. Focus on changes that affect the test's behavior or assertions
2. Include changes to methods/classes that the test depends on
3. Include changes to the focal method or its direct dependencies
4. Exclude unrelated changes even if they're in the same file
5. Consider the hunk type as a hint, but make your own judgment

Always return valid JSON with the specified format."""
    
    def _get_retrieval_decision_system_prompt(self) -> str:
        """Get system prompt for retrieval decision"""
        return """You are an expert at determining what information is needed to update test cases.

Your task is to decide if additional information retrieval is needed from the codebase, beyond what's already provided in the test, focal method, and diff hunks.

Key principles:
1. If all necessary information is present, retrieval is NOT needed
2. If there are missing classes, methods, or fields that are referenced but not defined, retrieval IS needed
3. If the changes are complex and require understanding of dependencies, retrieval IS needed
4. If the test can be updated with only the provided information, retrieval is NOT needed

Always return valid JSON with the specified format."""
    
    def _parse_filter_response(self, response: str, total_hunks: int) -> List[int]:
        """Parse LLM response to extract relevant hunk indices"""
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                relevant_indices = data.get("relevant_indices", [])
                
                # Validate indices
                valid_indices = [i for i in relevant_indices if isinstance(i, int) and 0 <= i < total_hunks]
                
                return valid_indices
        except Exception as e:
            self.log_error(f"Failed to parse filter response: {e}")
        
        # Fallback: return all indices
        return list(range(total_hunks))
    
    def _parse_retrieval_decision(self, response: str) -> bool:
        """Parse LLM response to extract retrieval decision"""
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                needs_retrieval = data.get("needs_retrieval", False)
                
                return bool(needs_retrieval)
        except Exception as e:
            self.log_error(f"Failed to parse retrieval decision: {e}")
        
        # Fallback: assume retrieval is needed
        return True
    
    def _filter_test_class_context(self, test_case: TestCase,
                                   focal_method_info: FocalMethodInfo,
                                   filtered_hunks: List[DiffHunk],
                                   focal_method_changed: bool) -> tuple[Optional[List[str]], Optional[List[Dict[str, Any]]]]:
        """
        Filter test class context (class fields and non-test methods) to keep only relevant ones.
        
        Args:
            test_case: The test case
            focal_method_info: Focal method information
            filtered_hunks: Already filtered relevant hunks
            focal_method_changed: Whether focal method changed
            
        Returns:
            Tuple of (filtered_class_fields, filtered_non_test_methods), or (None, None) if no filtering performed
        """
        if not test_case.class_fields and not test_case.non_test_methods:
            return None, None
        
        # Build prompt for LLM to filter test class context
        prompt = self._build_test_class_context_filter_prompt(
            test_case, focal_method_info, filtered_hunks, focal_method_changed
        )
        
        # Get LLM analysis
        response = self.llm_client.generate(
            prompt,
            system_prompt=self._get_test_class_context_filter_system_prompt()
        )
        
        # Parse response to get relevant fields and methods
        filtered_fields, filtered_methods = self._parse_test_class_context_filter_response(
            response, test_case.class_fields, test_case.non_test_methods
        )
        
        return filtered_fields, filtered_methods
    
    def _build_test_class_context_filter_prompt(self, test_case: TestCase,
                                                focal_method_info: FocalMethodInfo,
                                                filtered_hunks: List[DiffHunk],
                                                focal_method_changed: bool) -> str:
        """Build prompt for filtering test class context"""
        
        # Format focal method section
        focal_code_formatted = self._format_focal_method_with_diff(focal_method_info, show_line_numbers=True)
        
        if focal_method_changed:
            focal_section = f"""
**Focal Method (CHANGED):**
```java
{focal_code_formatted}
```

Note: Lines marked with [DELETE] show removed code, [ADD] shows added code.
"""
        else:
            focal_section = f"""
**Focal Method (UNCHANGED):**
```java
{focal_code_formatted}
```
"""
        
        # Format filtered hunks
        hunks_text = ""
        if filtered_hunks:
            for i, hunk in enumerate(filtered_hunks):
                hunk_type = hunk.hunk_type or "unknown"
                hunks_text += f"""
**Relevant Hunk {i} [{hunk_type}]:**
File: {hunk.file_path}
```
{hunk.context}
```
"""
        else:
            hunks_text = "No relevant hunks identified."
        
        # Format class fields
        fields_text = ""
        if test_case.class_fields:
            for i, field in enumerate(test_case.class_fields):
                fields_text += f"Field {i}: {field}\n"
        else:
            fields_text = "No class fields."
        
        # Format non-test methods
        methods_text = ""
        if test_case.non_test_methods:
            for i, method in enumerate(test_case.non_test_methods):
                method_name = method.get('name', f'method_{i}')
                method_code = method.get('code', '')
                methods_text += f"""
**Method {i} ({method_name}):**
```java
{method_code}
```
"""
        else:
            methods_text = "No non-test methods."
        
        return f"""Analyze which test class context elements (class fields and non-test methods) are relevant for updating the following test case.

**Test Case:**
```java
{test_case.code}
```

{focal_section}

**Relevant Code Changes:**
{hunks_text}

**Test Class Fields:**
{fields_text}

**Test Class Non-Test Methods:**
{methods_text}

**Task:**
Determine which class fields and non-test methods are truly relevant for updating this test. Consider:
1. Are they referenced by the test method?
2. Are they related to the focal method or code changes?
3. Do they provide necessary context for understanding the test?
4. Are they helper methods that the test uses?

Some fields/methods might be inner classes or unrelated helpers that just clutter the context without providing useful information.

Return a JSON object with:
- relevant_field_indices: list of field indices (0-based) that are relevant (empty list means none are relevant)
- relevant_method_indices: list of method indices (0-based) that are relevant (empty list means none are relevant)
- reasoning: explanation for selections

Example:
{{
  "relevant_field_indices": [0, 2],
  "relevant_method_indices": [1],
  "reasoning": "Field 0 is used by the test to set up data, Field 2 is referenced in assertions. Method 1 is a helper method called by the test. Other fields/methods are not referenced."
}}
"""
    
    def _get_test_class_context_filter_system_prompt(self) -> str:
        """Get system prompt for test class context filtering"""
        return """You are an expert at analyzing test code and determining which class-level context is relevant.

Your task is to filter test class fields and non-test methods to identify only those that are truly relevant for updating a test case.

Key principles:
1. Focus on elements that are actually referenced by the test method
2. Include elements related to the focal method or code changes
3. Include helper methods and setup/teardown methods that the test uses
4. Exclude inner classes, unrelated data structures, and unused helpers
5. Be conservative - when in doubt about relevance, include it

Always return valid JSON with the specified format."""
    
    def _parse_test_class_context_filter_response(self, response: str,
                                                  class_fields: List[str],
                                                  non_test_methods: List[Dict[str, Any]]) -> tuple[Optional[List[str]], Optional[List[Dict[str, Any]]]]:
        """Parse LLM response to extract relevant class fields and non-test methods"""
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                
                # Get indices
                field_indices = data.get("relevant_field_indices", [])
                method_indices = data.get("relevant_method_indices", [])
                
                # Validate and filter fields
                filtered_fields = None
                if class_fields:
                    valid_field_indices = [i for i in field_indices if isinstance(i, int) and 0 <= i < len(class_fields)]
                    if valid_field_indices:  # Only return filtered list if there are relevant fields
                        filtered_fields = [class_fields[i] for i in valid_field_indices]
                    elif not field_indices:  # Empty list means explicitly no relevant fields
                        filtered_fields = []
                    # If parsing fails or invalid indices, return None (use all fields)
                
                # Validate and filter methods
                filtered_methods = None
                if non_test_methods:
                    valid_method_indices = [i for i in method_indices if isinstance(i, int) and 0 <= i < len(non_test_methods)]
                    if valid_method_indices:  # Only return filtered list if there are relevant methods
                        filtered_methods = [non_test_methods[i] for i in valid_method_indices]
                    elif not method_indices:  # Empty list means explicitly no relevant methods
                        filtered_methods = []
                    # If parsing fails or invalid indices, return None (use all methods)
                
                return filtered_fields, filtered_methods
        except Exception as e:
            self.log_error(f"Failed to parse test class context filter response: {e}")
        
        # Fallback: return None to use all fields and methods
        return None, None
    
    def _build_reasoning(self, filtered_hunks: List[DiffHunk],
                        needs_retrieval: bool,
                        retrieval_result: Optional[RetrievalResult],
                        filtered_class_fields: Optional[List[str]],
                        filtered_non_test_methods: Optional[List[Dict[str, Any]]]) -> str:
        """Build reasoning string for the result"""
        reasoning_parts = []
        
        reasoning_parts.append(f"Filtered to {len(filtered_hunks)} relevant hunks from the provided diff hunks.")
        
        # Add filtering info for class fields and methods
        if filtered_class_fields is not None:
            reasoning_parts.append(f"Filtered to {len(filtered_class_fields)} relevant class field(s).")
        if filtered_non_test_methods is not None:
            reasoning_parts.append(f"Filtered to {len(filtered_non_test_methods)} relevant non-test method(s).")
        
        if needs_retrieval:
            reasoning_parts.append("Determined that additional information retrieval is needed.")
            
            if retrieval_result:
                if retrieval_result.retrieval_successful:
                    reasoning_parts.append(
                        f"Successfully retrieved {len(retrieval_result.retrieved_methods)} methods "
                        f"and {len(retrieval_result.retrieved_fields)} fields."
                    )
                else:
                    reasoning_parts.append(
                        f"Retrieval was attempted but not fully successful: {retrieval_result.retrieval_reasoning}"
                    )
            else:
                reasoning_parts.append("Retrieval was needed but not performed (no retrieval agent available).")
        else:
            reasoning_parts.append("Determined that no additional information retrieval is needed.")
        
        return " ".join(reasoning_parts)
    
    def _format_test_class_context(self, test_case: TestCase) -> str:
        """
        Format test class context including class fields and non-test methods.
        
        Args:
            test_case: TestCase object containing class_fields and non_test_methods
            
        Returns:
            Formatted string showing test class context
        """
        if not test_case.class_fields and not test_case.non_test_methods:
            return ""
        
        sections = []
        
        # Format class fields
        if test_case.class_fields:
            fields_text = "\n".join(f"  {field}" for field in test_case.class_fields)
            sections.append(f"""**Test Class Fields:**
```java
{fields_text}
```
These are class-level fields available to the test method.""")
        
        # Format non-test methods (show all methods' code)
        if test_case.non_test_methods:
            methods_codes = []
            for method in test_case.non_test_methods:  # Show all methods
                code = method.get('code', '')
                if code:
                    methods_codes.append(code)
            
            methods_text = "\n\n".join(methods_codes)
            sections.append(f"""**Test Class Helper Methods:**
```java
{methods_text}
```
These are non-test methods in the same test class (e.g., @Before, @After, helper methods).""")
        
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
        
        # Use simple line-by-line comparison for better readability
        result = []
        
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
                result.append(f"       | [DELETE] {original_lines[i]}")
        
        # Handle extra lines in current (added lines)
        if len(current_lines) > len(original_lines):
            for i in range(len(original_lines), len(current_lines)):
                line_num = focal.start_line + i if focal.start_line > 0 else i + 1
                result.append(f"{line_num:6d} | [ADD] {current_lines[i]}")
        
        return '\n'.join(result)


