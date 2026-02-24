"""
Java Test Executor for BEAM Framework

Executes Java tests with JaCoCo coverage and PITest mutation testing.
Two-phase execution:
1. JaCoCo for code coverage (detects compile/run errors)
2. PITest for mutation testing (only if phase 1 succeeds)
"""

import subprocess
import os
import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
import shutil
import tempfile
import time
import platform

from models import (
    TestResultInfo, TestResultStatus, TestCase, FocalMethodInfo,
    CoverageInfo, MutationInfo, LineCoverage, DetailedCoverageInfo,
    MutationDetail, DetailedMutationInfo
)
from config import config
from utils import logger, get_sample_logger

# Maven skip parameters (consistent with TestUpdater)
MVN_SKIPS = [
    "-Djacoco.skip",
    "-Dcheckstyle.skip",
    "-Dspotless.apply.skip",
    "-Drat.skip",
    "-Denforcer.skip",
    "-Danimal.sniffer.skip",
    "-Dmaven.javadoc.skip",
    "-Dmaven.gitcommitid.skip",
    "-Dfindbugs.skip",
    "-Dwarbucks.skip",
    "-Dmodernizer.skip",
    "-Dimpsort.skip",
    "-Dpmd.skip",
    "-Dxjc.skip",
    "-Dair.check.skip-all",
    "-Dlicense.skip",
    "-Dfindbugs.skip",
    "-Denforcer.skip",
    "-Dremoteresources.skip",
]


class JavaTestExecutor:
    """
    Executes Java tests with coverage and mutation analysis.
    
    Two-phase execution:
    1. JaCoCo for code coverage (detects compile/run errors)
    2. PITest for mutation testing (only if phase 1 succeeds)
    """
    
    def __init__(self, project_path: str, java_version: str = "8", maven_repo_dir: str = None, project_name: str = None, sample_id: str = None):
        """
        Initialize Java test executor.
        
        Args:
            project_path: Path to the Java project
            java_version: Java version to use (8, 11, 17, 21)
            maven_repo_dir: Optional path to Maven local repository. If None, uses default ~/.m2/repository
            project_name: Optional project name to generate project-specific repo path (e.g., "dromara/hutool")
            sample_id: Optional sample ID for organizing reports (e.g., "13" or "dromara/hutool:13")
        """
        self.project_path = Path(project_path)
        self.java_version = str(java_version)
        self.java_home = config.java.java_homes.get(self.java_version)
        self.maven_home = config.java.maven_home
        self.project_name = project_name
        self.sample_id = sample_id
        
        # Determine Maven repository path
        if maven_repo_dir:
            # Use provided repo path directly
            self.maven_repo_dir = maven_repo_dir
        else:
            # Use project-specific or default repo path
            self.maven_repo_dir = config.java.get_maven_repo_path(project_name)
        
        # Determine reports directory with timestamp (for this sample execution)
        if project_name and sample_id:
            sample_base = Path(config.java.get_sample_reports_dir(project_name, sample_id, config.framework))
            sample_base.mkdir(parents=True, exist_ok=True)
            
            # Create timestamp directory for this sample execution
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.sample_execution_dir = sample_base / timestamp
            self.sample_execution_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info(f"  Sample Execution Dir: {self.sample_execution_dir}")
        else:
            self.sample_execution_dir = None
            logger.info(f"  Reports Dir: Not configured (will use default Maven target directories)")
        
        # Current iteration reports directory (will be set for each test execution)
        self.current_iteration_dir = None
        
        if not self.java_home:
            raise ValueError(f"Java {java_version} not configured in config.java.java_homes")
        
        if not Path(self.java_home).exists():
            raise ValueError(f"Java home not found: {self.java_home}")
        
        logger.info(f"JavaTestExecutor initialized")
        logger.info(f"  Project: {self.project_path}")
        logger.info(f"  Java {self.java_version}: {self.java_home}")
        logger.info(f"  Maven: {self.maven_home}")
        logger.info(f"  Maven Repo: {self.maven_repo_dir if self.maven_repo_dir else 'default (~/.m2/repository)'}")
    
    def _add_imports_to_file(self, file_content: str, new_imports: List[str]) -> tuple:
        """
        Add new import statements to a Java file, avoiding duplicates.
        
        Args:
            file_content: Current file content
            new_imports: List of new import statements (e.g., ["import com.example.NewClass;"])
            
        Returns:
            Tuple of (updated_content: str, lines_added: int)
        """
        if not new_imports:
            return (file_content, 0)
        
        lines = file_content.split('\n')
        
        # Find the position to insert imports (after package declaration, before class declaration)
        last_import_idx = -1
        package_idx = -1
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('package '):
                package_idx = i
            elif stripped.startswith('import '):
                last_import_idx = i
        
        # Determine insertion position
        if last_import_idx >= 0:
            # Insert after the last existing import
            insert_idx = last_import_idx + 1
        elif package_idx >= 0:
            # Insert after package declaration (with a blank line)
            insert_idx = package_idx + 1
            # Add a blank line after package if not present
            if insert_idx < len(lines) and lines[insert_idx].strip():
                lines.insert(insert_idx, '')
                insert_idx += 1
        else:
            # Insert at the beginning
            insert_idx = 0
        
        # Check for duplicate imports
        existing_imports = set()
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('import ') and stripped.endswith(';'):
                existing_imports.add(stripped)
        
        # Add new imports (avoiding duplicates)
        imports_to_add = []
        for new_import in new_imports:
            # Normalize the import statement
            normalized = new_import.strip()
            if not normalized.endswith(';'):
                normalized += ';'
            if not normalized.startswith('import '):
                normalized = 'import ' + normalized
            
            # Check if it's not a duplicate
            if normalized not in existing_imports:
                imports_to_add.append(normalized)
                existing_imports.add(normalized)
        
        # Insert the new imports
        lines_added = 0
        if imports_to_add:
            for import_stmt in reversed(imports_to_add):
                lines.insert(insert_idx, import_stmt)
            lines_added = len(imports_to_add)
        
        return ('\n'.join(lines), lines_added)
    
    def _extract_function_name(self, function_text: str) -> Optional[str]:
        """
        Extract function name from Java function text.
        Improved version from replace_method.py to handle various method definition patterns.
        
        Args:
            function_text: Java method text
            
        Returns:
            Method name or None if not found
        """
        # Method 1: Match complete method signature with modifiers
        patterns = [
            # Match method with access modifiers (public/private/protected + optional static/final + return type + method name)
            r'(?:public|private|protected)\s+(?:static\s+)?(?:final\s+)?(?:\w+(?:<[^>]*>)?(?:\[\])?\s+)(\w+)\s*\(',
            # Match method without access modifier but with return type (package-private methods)
            r'^\s*(?:static\s+)?(?:final\s+)?(?:\w+(?:<[^>]*>)?(?:\[\])?\s+)(\w+)\s*\(',
            # Match method after annotations like @Test
            r'@\w+(?:\([^)]*\))?\s+(?:public|private|protected)?\s*(?:\w+\s+)?(\w+)\s*\(',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, function_text, re.MULTILINE)
            if match:
                return match.group(1)
        
        # Fallback: simple pattern (compatible with old logic)
        match = re.search(r'\b(\w+)\s*\(', function_text)
        return match.group(1) if match else None
    
    def _simple_function_match(self, content: str, function_name: str, updated_text: str) -> Optional[Tuple[str, str]]:
        """
        Simple Java method matching and replacement with intelligent annotation handling.
        Based on replace_method.py's _simple_function_match logic.
        
        Args:
            content: File content
            function_name: Method name to replace
            updated_text: New method code
            
        Returns:
            Tuple of (new_content: str, match_info: str) or None if not found
        """
        # Match annotation part and function signature part separately
        # This allows us to intelligently handle annotations
        pattern = rf'((?:@\w+(?:\([^)]*\))?\s*)*)((?:(?:public|private|protected)\s+)?(?:(?:static|final|abstract|synchronized)\s+)*(?:\w+(?:<[^>]*>)?(?:\[\])?\s+)?{re.escape(function_name)}\s*\([^)]*\)\s*(?:throws\s+[^{{]+)?\s*\{{)'
        
        match = re.search(pattern, content)
        if not match:
            return None
        
        # Extract annotation part and function signature part
        annotations = match.group(1)  # Annotation part (e.g., "@Test\n")
        function_signature = match.group(2)  # Function signature part
        
        # Find method start position
        method_start = match.start()
        method_signature_end = match.end()
        
        # Simple brace matching to find method end
        brace_count = 1
        pos = method_signature_end
        
        while pos < len(content) and brace_count > 0:
            char = content[pos]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            pos += 1
        
        if brace_count > 0:
            logger.warning(f"Unable to find closing brace for function {function_name}")
            return None
        
        method_end = pos
        
        # Intelligently handle annotations: check if updated_text already contains annotations
        # If updated_text starts with @ (like @Test), don't preserve original annotations to avoid duplication
        updated_text_stripped = updated_text.lstrip()
        if updated_text_stripped.startswith('@'):
            # LLM generated code already contains annotations, use it directly
            new_content = content[:method_start] + updated_text + content[method_end:]
            match_info = f"(match position: {method_start}-{method_end}, annotations in generated code, discarded original: {annotations.strip()})"
        else:
            # LLM generated code doesn't contain annotations, preserve original annotations
            new_content = content[:method_start] + annotations + updated_text + content[method_end:]
            match_info = f"(match position: {method_start}-{method_end}, preserved original annotations: {annotations.strip()})"
        
        return new_content, match_info
    
    def _replace_test_method_in_file_simple(self, test_file: Path, original_test_code: str, new_test_code: str, new_imports: List[str] = None) -> tuple:
        """
        Replace test method using simple string replacement (like TestUpdater).
        More reliable than regex matching for complex nested code.
        
        Args:
            test_file: Path to the test file
            original_test_code: Complete original test method code (with @Test annotation)
            new_test_code: Complete new test method code (with @Test annotation)
            new_imports: List of new import statements to add
            
        Returns:
            Tuple of (success: bool, start_line: int)
        """
        try:
            original_content = test_file.read_text(encoding='utf-8')
            
            # Simple string replacement - most reliable method
            if original_test_code not in original_content:
                logger.error(f"Original test code not found in {test_file}")
                logger.debug(f"Looking for:\n{original_test_code[:200]}...")
                return (False, 0)
            
            # Calculate line number before replacement
            method_start = original_content.find(original_test_code)
            method_start_line = original_content[:method_start].count('\n') + 1
            
            # Replace
            new_content = original_content.replace(original_test_code, new_test_code, 1)  # Replace only first occurrence
            
            # Add new imports if provided
            import_lines_added = 0
            if new_imports:
                new_content, import_lines_added = self._add_imports_to_file(new_content, new_imports)
                logger.info(f"Added {len(new_imports)} new import(s) to {test_file}")
            
            # Adjust start line if imports were added before the method
            adjusted_start_line = method_start_line + import_lines_added
            
            # Write back
            test_file.write_text(new_content, encoding='utf-8')
            logger.info(f"Successfully replaced test method at line {adjusted_start_line} in {test_file}")
            return (True, adjusted_start_line)
            
        except Exception as e:
            logger.error(f"Error replacing test method: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return (False, 0)
    
    def _replace_test_method_in_file(self, test_file: Path, test_method: str, new_test_code: str, new_imports: List[str] = None) -> tuple:
        """
        Replace a specific test method in a test file while preserving the rest of the file.
        Uses intelligent annotation handling from replace_method.py.
        Also adds new import statements if provided.
        
        Args:
            test_file: Path to the test file
            test_method: Name of the test method to replace
            new_test_code: New test method code (just the method, with @Test annotation)
            new_imports: List of new import statements to add (e.g., ["import com.example.NewClass;"])
            
        Returns:
            Tuple of (success: bool, start_line: int)
            - success: True if replacement was successful, False otherwise
            - start_line: Line number where the test method starts (1-indexed), or 0 if failed
        """
        try:
            original_content = test_file.read_text(encoding='utf-8')
            
            # Use improved matching logic from replace_method.py
            replacement_result = self._simple_function_match(original_content, test_method, new_test_code)
            
            if not replacement_result:
                logger.warning(f"Could not find test method {test_method} in {test_file} using improved matcher")
                # Fallback to old pattern matching
                pattern = r'(@Test[^\n]*\n(?:\s*@[^\n]+\n)*\s*public\s+void\s+' + re.escape(test_method) + r'\s*\([^)]*\)\s*(?:throws[^{]*)?\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})'
                match = re.search(pattern, original_content, re.MULTILINE | re.DOTALL)
                
                if not match:
                    logger.warning(f"Could not find test method {test_method} in {test_file} using fallback pattern")
                    return (False, 0)
                
                # Get the indentation of the original method
                method_start = match.start()
                line_start = original_content.rfind('\n', 0, method_start) + 1
                indent = ''
                for char in original_content[line_start:method_start]:
                    if char in ' \t':
                        indent += char
                    else:
                        break
                
                # Calculate the line number where the method starts (1-indexed)
                method_start_line = original_content[:method_start].count('\n') + 1
                
                # Indent the new test code to match
                new_lines = new_test_code.split('\n')
                indented_new_code = '\n'.join(indent + line if line.strip() else line for line in new_lines)
                
                # Replace the method
                new_content = original_content[:match.start()] + indented_new_code + original_content[match.end():]
            else:
                # Use improved matching result
                new_content, match_info = replacement_result
                logger.info(f"Successfully matched test method {test_method} {match_info}")
                
                # Calculate line number from new_content
                # Find where the new method was inserted
                method_start = new_content.find(new_test_code)
                if method_start == -1:
                    # If new_test_code has annotations, the match_info tells us where it was inserted
                    # We need to search more carefully
                    # For now, use a simple approach: count lines before the method name
                    method_name_pattern = rf'\b{re.escape(test_method)}\s*\('
                    match = re.search(method_name_pattern, new_content)
                    if match:
                        method_start = match.start()
                    else:
                        method_start = 0
                
                method_start_line = new_content[:method_start].count('\n') + 1 if method_start > 0 else 1
            
            # Add new imports if provided
            import_lines_added = 0
            if new_imports:
                new_content, import_lines_added = self._add_imports_to_file(new_content, new_imports)
                logger.info(f"Added {len(new_imports)} new import(s) to {test_file}")
            
            # Adjust start line if imports were added before the method
            adjusted_start_line = method_start_line + import_lines_added
            
            # Write back
            test_file.write_text(new_content, encoding='utf-8')
            logger.info(f"Successfully replaced test method {test_method} at line {adjusted_start_line} in {test_file}")
            return (True, adjusted_start_line)
            
        except Exception as e:
            logger.error(f"Error replacing test method: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return (False, 0)
    
    def execute_test(self, 
                     test_code: str,
                     test_class: str,
                     test_method: str,
                     focal_method_info: FocalMethodInfo,
                     test_rel_path: str = None,
                     iteration: int = None,
                     new_imports: List[str] = None,
                     test_imports: List[str] = None,
                     class_fields: List[str] = None,
                     non_test_methods: List[Dict] = None,
                     original_test_code: str = None) -> TestResultInfo:
        """
        Execute a Java test with coverage and mutation analysis.
        
        Args:
            test_code: The test code to execute (just the test method with @Test annotation)
            test_class: Full qualified test class name (e.g., "cn.hutool.db.sql.ConditionTest")
            test_method: Test method name (e.g., "parseTest")
            focal_method_info: Information about the focal method
            test_rel_path: Relative path to test file (e.g., "hutool-db/src/test/java/...")
            iteration: Current iteration number (optional, for organizing reports)
            new_imports: List of new import statements to add (e.g., ["import com.example.NewClass;"])
            test_imports: List of existing import statements from input['test_import'] (NOT USED - kept for compatibility)
                         Note: This parameter is now ignored as we only use new_imports to avoid outdated imports
            original_test_code: Optional original test code for simple string replacement (more reliable than regex)
            
        Returns:
            TestResultInfo with execution results
        """
        # Get sample logger
        sample_logger = get_sample_logger()
        
        logger.info(f"Executing test: {test_class}.{test_method}")
        sample_logger.log_info(f"[JavaTestExecutor] Executing test: {test_class}.{test_method}")
        
        # Only use new_imports (from AI), not test_imports (which may contain outdated imports)
        # This ensures only two sources are included:
        # 1. new_imports - AI generated imports
        # 2. aCommit file imports - will be handled by _add_imports_to_file (deduplication)
        # We exclude test_imports (from input['test_import']) because it contains imports from bCommit
        # which may be outdated after the code change. The aCommit file already has the correct imports.
        all_imports = []
        if new_imports:
            all_imports.extend(new_imports)
            sample_logger.log_info(f"[JavaTestExecutor] Including {len(new_imports)} new import(s) from AI")
        if test_imports:
            sample_logger.log_info(f"[JavaTestExecutor] Skipping {len(test_imports)} import(s) from test_imports (may be outdated)")
        
        # Create iteration-specific reports directory under the sample execution timestamp directory
        if self.sample_execution_dir:
            iteration_name = f"iteration_{iteration}" if iteration is not None else "execution"
            self.current_iteration_dir = self.sample_execution_dir / iteration_name
            self.current_iteration_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Reports will be saved to: {self.current_iteration_dir}")
            sample_logger.log_info(f"[JavaTestExecutor] Reports directory: {self.current_iteration_dir}")
        else:
            self.current_iteration_dir = None
        
        # Replace test method in file if test_rel_path is provided
        backup_file = None
        test_file = None
        test_method_start_line = 0  # Track test method start line for error location calculation
        
        if test_rel_path:
            test_file = self.project_path / test_rel_path
            if test_file.exists():
                # Backup original file
                backup_file = test_file.with_suffix('.java.backup')
                shutil.copy2(test_file, backup_file)
                sample_logger.log_info(f"[JavaTestExecutor] Backed up test file: {test_file}")
                try:
                    # DISABLED: Simple string replacement (keeping code for future reference)
                    # Only use the improved regex matching from replace_method.py
                    USE_SIMPLE_STRING_REPLACEMENT = False
                    
                    if USE_SIMPLE_STRING_REPLACEMENT and original_test_code:
                        # Try simple string replacement first if original_test_code is provided
                        success, test_method_start_line = self._replace_test_method_in_file_simple(
                            test_file, original_test_code, test_code, all_imports
                        )
                        if success:
                            sample_logger.log_info(f"[JavaTestExecutor] Replaced test method using simple string replacement at line {test_method_start_line}")
                        else:
                            logger.warning(f"Simple replacement failed, falling back to regex method")
                            # Fallback to regex method
                            success, test_method_start_line = self._replace_test_method_in_file(test_file, test_method, test_code, all_imports)
                    else:
                        # Use improved regex method (integrated from replace_method.py)
                        success, test_method_start_line = self._replace_test_method_in_file(test_file, test_method, test_code, all_imports)
                        sample_logger.log_info(f"[JavaTestExecutor] Using improved regex matching (from replace_method.py)")
                    
                    if not success:
                        logger.warning(f"Could not replace test method, test file may be incomplete")
                        sample_logger.log_warning(f"[JavaTestExecutor] Could not replace test method")
                        test_method_start_line = 0
                    else:
                        sample_logger.log_info(f"[JavaTestExecutor] Replaced test method at line {test_method_start_line}")
                        if all_imports:
                            sample_logger.log_info(f"[JavaTestExecutor] Processed {len(all_imports)} total import(s) (after deduplication with file)")
                except Exception as e:
                    # Restore backup on error
                    if backup_file and backup_file.exists():
                        shutil.copy2(backup_file, test_file)
                    raise e
        
        try:
            # Phase 1: JaCoCo execution (always run for coverage computation)
            logger.info("Phase 1: Running JaCoCo for coverage...")
            sample_logger.log_info("[JavaTestExecutor] Phase 1: Running JaCoCo for coverage analysis...")
            jacoco_result = self._run_jacoco(test_class, test_method, focal_method_info, test_rel_path)
            
            # Check for compile/run errors
            if jacoco_result['status'] == 'COMPILE_ERROR':
                sample_logger.log_error("[JavaTestExecutor] Phase 1 FAILED: Compilation error")
                sample_logger.log_info(f"[JavaTestExecutor] Error: {jacoco_result.get('error_message', 'Compilation failed')}")
                return self._create_result(
                    status=TestResultStatus.COMPILE_ERROR,
                    test_code=test_code,
                    test_class=test_class,
                    test_method=test_method,
                    focal_method_info=focal_method_info,
                    error_message=jacoco_result.get('error_message', 'Compilation failed'),
                    raw_error_output=jacoco_result.get('raw_error_output'),
                    coverage_info=CoverageInfo(),
                    mutation_info=MutationInfo(),
                    test_file_path=test_file,
                    class_fields=class_fields,
                    non_test_methods=non_test_methods,
                    test_method_start_line=test_method_start_line
                )
            
            if jacoco_result['status'] == 'RUN_FAIL':
                sample_logger.log_error("[JavaTestExecutor] Phase 1 FAILED: Test execution failed")
                sample_logger.log_info(f"[JavaTestExecutor] Error: {jacoco_result.get('error_message', 'Test execution failed')}")
                return self._create_result(
                    status=TestResultStatus.RUN_FAIL,
                    test_code=test_code,
                    test_class=test_class,
                    test_method=test_method,
                    focal_method_info=focal_method_info,
                    error_message=jacoco_result.get('error_message', 'Test execution failed'),
                    raw_error_output=jacoco_result.get('raw_error_output'),
                    coverage_info=jacoco_result.get('coverage_info', CoverageInfo()),
                    mutation_info=MutationInfo(),
                    class_fields=class_fields,
                    non_test_methods=non_test_methods,
                    test_file_path=test_file,
                    test_method_start_line=test_method_start_line
                )
            
            # Log Phase 1 success
            coverage_info_phase1 = jacoco_result.get('coverage_info', CoverageInfo())
            sample_logger.log_info(f"[JavaTestExecutor] Phase 1 SUCCESS: Coverage = {coverage_info_phase1.coverage_percentage:.2f}%")
            
            # Phase 2: PITest execution (always run for mutation computation)
            logger.info("Phase 2: Running PITest for mutation testing...")
            sample_logger.log_info("[JavaTestExecutor] Phase 2: Running PITest for mutation testing...")
            pitest_result = self._run_pitest(test_class, test_method, focal_method_info, test_rel_path)
            
            # Combine results
            coverage_info = jacoco_result.get('coverage_info', CoverageInfo())
            mutation_info = pitest_result.get('mutation_info', MutationInfo())
            
            # Log Phase 2 completion
            sample_logger.log_info(f"[JavaTestExecutor] Phase 2 COMPLETED: Mutation kill rate = {mutation_info.kill_percentage:.2f}%")
            
            # Determine final status based on ablation settings
            line_coverage_threshold = config.framework.coverage_threshold * 100
            branch_coverage_threshold = config.framework.branch_coverage_threshold * 100
            mutation_threshold = config.framework.mutation_threshold * 100
            
            # Check which criteria to evaluate based on ablation settings
            check_coverage = not config.framework.all_ablation_disable_coverage
            check_mutation = not config.framework.all_ablation_disable_mutation
            
            # Coverage check (skip threshold check if disabled)
            if check_coverage:
                # 行覆盖率判断
                line_coverage_ok = coverage_info.line_coverage_percentage >= line_coverage_threshold
                
                # 分支覆盖率判断：只有当存在分支时才检查分支覆盖率
                if coverage_info.branch_coverage_percentage is not None:
                    branch_coverage_ok = coverage_info.branch_coverage_percentage >= branch_coverage_threshold
                else:
                    # 没有分支，分支覆盖率自动通过
                    branch_coverage_ok = True
                
                # 总的覆盖率判断：行覆盖和分支覆盖都要通过
                coverage_ok = line_coverage_ok and branch_coverage_ok
            else:
                # Coverage threshold check disabled, always pass
                coverage_ok = True
            
            # Mutation check (skip threshold check if disabled)
            if check_mutation:
                mutation_ok = mutation_info.kill_percentage >= mutation_threshold
            else:
                # Mutation threshold check disabled, always pass
                mutation_ok = True
            
            # Determine status
            if coverage_ok and mutation_ok:
                status = TestResultStatus.PASS
            elif not coverage_ok and not mutation_ok:
                status = TestResultStatus.COVERAGE_AND_MUTATION_LOSS
            elif not coverage_ok:
                status = TestResultStatus.COVERAGE_LOSS
            else:
                status = TestResultStatus.MUTATION_LOSS
            
            # Log final status (always log coverage and mutation, regardless of ablation)
            sample_logger.log_info(f"[JavaTestExecutor] Final Status: {status.name}")
            sample_logger.log_info(f"[JavaTestExecutor] Line Coverage: {coverage_info.line_coverage_percentage:.2f}% (threshold: {line_coverage_threshold:.2f}%)")
            if coverage_info.branch_coverage_percentage is not None:
                sample_logger.log_info(f"[JavaTestExecutor] Branch Coverage: {coverage_info.branch_coverage_percentage:.2f}% (threshold: {branch_coverage_threshold:.2f}%)")
            else:
                sample_logger.log_info(f"[JavaTestExecutor] Branch Coverage: None (no branches)")
            sample_logger.log_info(f"[JavaTestExecutor] Mutation Kill Rate: {mutation_info.kill_percentage:.2f}% (threshold: {mutation_threshold:.2f}%)")
            
            return self._create_result(
                status=status,
                test_code=test_code,
                test_class=test_class,
                test_method=test_method,
                focal_method_info=focal_method_info,
                error_message=None,
                coverage_info=coverage_info,
                mutation_info=mutation_info,
                test_file_path=test_file,
                class_fields=class_fields,
                non_test_methods=non_test_methods,
                test_method_start_line=test_method_start_line
            )
        finally:
            # Always restore the backup file after test execution
            if backup_file and backup_file.exists() and test_file:
                try:
                    shutil.copy2(backup_file, test_file)
                    backup_file.unlink()  # Delete backup file
                    logger.info(f"Restored original test file: {test_file}")
                    sample_logger.log_info(f"[JavaTestExecutor] Restored original test file")
                except Exception as e:
                    logger.error(f"Failed to restore backup file: {e}")
                    sample_logger.log_error(f"[JavaTestExecutor] Failed to restore backup file: {e}")

    def _run_jacoco(self, test_class: str, test_method: str, 
                    focal_method_info: FocalMethodInfo = None, test_rel_path: str = None) -> Dict:
        """
        Run test with JaCoCo coverage.
        
        Args:
            test_class: Full qualified test class name
            test_method: Test method name
            focal_method_info: Information about focal method for detailed coverage extraction
            test_rel_path: Relative path to test file
        
        Returns:
            Dict with status, coverage_info, error_message
        """
        sample_logger = get_sample_logger()
        
        try:
            # Find parent POM for the test
            if test_rel_path:
                test_path = self.project_path / test_rel_path
                pom_path = self._find_parent_pom(test_path)
            else:
                pom_path = self.project_path / "pom.xml"
            
            if not pom_path or not pom_path.exists():
                sample_logger.log_error("[JaCoCo] pom.xml not found")
                return {
                    'status': 'ERROR',
                    'error_message': 'pom.xml not found'
                }
            
            sample_logger.log_info(f"[JaCoCo] Using POM: {pom_path.relative_to(self.project_path)}")
            
            # Prepare POM modifications (must be done before building command)
            self._prepare_pom_for_jacoco(pom_path)
            sample_logger.log_info("[JaCoCo] POM prepared for JaCoCo execution")
            
            # Clean old JaCoCo files to avoid conflicts
            self._clean_old_jacoco_files(pom_path)
            sample_logger.log_info("[JaCoCo] Cleaned old JaCoCo files")
            
            # Build Maven command for JaCoCo
            cmd = self._build_jacoco_command(test_class, test_method, pom_path)
            sample_logger.log_info(f"[JaCoCo] Maven command: {' '.join(cmd)}")
            
            # Execute command
            sample_logger.log_info("[JaCoCo] Starting Maven execution...")
            result = self._execute_maven_command(cmd)
            sample_logger.log_info(f"[JaCoCo] Maven execution completed with return code: {result['returncode']}")
            
            # Parse output
            if result['returncode'] != 0:
                # Check for compilation error
                if self._is_compile_error(result['output']):
                    error_msg = self._extract_compile_error(result['output'], test_rel_path)
                    sample_logger.log_error(f"[JaCoCo] Compilation error detected")
                    sample_logger.log_info(f"[JaCoCo] Error message: {error_msg}")
                    return {
                        'status': 'COMPILE_ERROR',
                        'error_message': error_msg,
                        'raw_error_output': self._extract_full_compile_error(result['output'])
                    }
                # Check for test failure
                else:
                    error_msg = self._extract_test_failure(result['output'], test_class, test_method)
                    sample_logger.log_error(f"[JaCoCo] Test execution failed")
                    sample_logger.log_info(f"[JaCoCo] Error message: {error_msg}")
                    return {
                        'status': 'RUN_FAIL',
                        'error_message': error_msg,
                        'raw_error_output': self._extract_full_test_failure(result['output'], test_class, test_method),
                        'coverage_info': self._parse_jacoco_report(pom_path, focal_method_info)
                    }
            
            # Even if returncode is 0, check for test failures in surefire reports
            # Some Maven configurations may return 0 even when tests have Errors
            test_failure_info = self._check_surefire_for_failures(pom_path, test_class, test_method)
            if test_failure_info:
                error_msg = test_failure_info.get('error_message', 'Test execution failed')
                sample_logger.log_error(f"[JaCoCo] Test had errors despite Maven returning 0")
                sample_logger.log_info(f"[JaCoCo] Error message: {error_msg}")
                return {
                    'status': 'RUN_FAIL',
                    'error_message': error_msg,
                    'raw_error_output': test_failure_info.get('full_output', error_msg),
                    'coverage_info': self._parse_jacoco_report(pom_path, focal_method_info)
                }
            
            # Parse JaCoCo coverage report
            sample_logger.log_info("[JaCoCo] Parsing coverage report...")
            coverage_info = self._parse_jacoco_report(pom_path, focal_method_info)
            
            # Log focal method coverage info
            if coverage_info.total_lines > 0:
                sample_logger.log_info(f"[JaCoCo] Focal method line coverage: {len(coverage_info.covered_lines)}/{coverage_info.total_lines} lines ({coverage_info.coverage_percentage:.2f}%)")
                if coverage_info.total_branches > 0:
                    covered_branches = sum(1 for line in coverage_info.detailed_coverage.line_coverages 
                                         if line.has_branch and line.covered_branches > 0) if coverage_info.detailed_coverage else 0
                    branch_pct = (covered_branches / coverage_info.total_branches * 100) if coverage_info.total_branches > 0 else 0
                    sample_logger.log_info(f"[JaCoCo] Focal method branch coverage: {covered_branches}/{coverage_info.total_branches} branches ({branch_pct:.2f}%)")
            else:
                sample_logger.log_info(f"[JaCoCo] No coverage data found for focal method")
            
            # Copy JaCoCo reports from default location to custom directory if needed
            if self.current_iteration_dir:
                self._copy_jacoco_reports(pom_path)
            
            return {
                'status': 'SUCCESS',
                'coverage_info': coverage_info
            }
            
        except subprocess.TimeoutExpired:
            logger.error("JaCoCo execution timeout")
            sample_logger.log_error("[JaCoCo] Execution timeout")
            return {
                'status': 'TIMEOUT',
                'error_message': 'Test execution timeout'
            }
        except Exception as e:
            logger.error(f"JaCoCo execution error: {e}")
            sample_logger.log_error(f"[JaCoCo] Execution error: {e}")
            return {
                'status': 'ERROR',
                'error_message': str(e)
            }
    
    def _build_jacoco_command(self, test_class: str, test_method: str, pom_path: Path) -> List[str]:
        """Build Maven command for JaCoCo execution."""
        # Use platform-specific Maven script
        maven_script = "mvn.cmd" if platform.system() == "Windows" else "mvn"
        maven_cmd = str(Path(self.maven_home) / "bin" / maven_script)
        
        # Determine module path
        pom_module = str(pom_path.parent.relative_to(self.project_path))
        
        # Get simple class name for -Dtest parameter
        simple_class = test_class.split('.')[-1]
        
        # Maven test command with JaCoCo
        cmd = [
            maven_cmd,
            "org.jacoco:jacoco-maven-plugin:0.8.8:prepare-agent",
            "test",
            "org.jacoco:jacoco-maven-plugin:0.8.8:report",
            "-nsu",
            "-pl", pom_module,  # Split into two arguments
            "--also-make",
            f"-Dtest={simple_class}#{test_method}",
            "-Dsurefire.failIfNoSpecifiedTests=false",
            "-DfailIfNoTests=false",
            "-Dmaven.test.skip=false",
            "-DskipTests=false",
            "--batch-mode",
            "-Dcheckstyle.skip",
            "-Dspotless.apply.skip",
            "-Drat.skip",
            "-Denforcer.skip",
            "-Dmaven.javadoc.skip",
            "-Dpmd.skip"
        ]
        
        # Add custom JaCoCo output directory if current_iteration_dir is configured
        # Note: JaCoCo report plugin uses 'outputDirectory' parameter for XML/HTML reports
        if self.current_iteration_dir:
            jacoco_dir = self.current_iteration_dir / "jacoco"
            jacoco_dir.mkdir(parents=True, exist_ok=True)
            # dataFile: where to write the exec file during prepare-agent
            cmd.append(f"-Djacoco.destFile={jacoco_dir}/jacoco.exec")
            # dataFile: where to read the exec file during report generation
            cmd.append(f"-Djacoco.dataFile={jacoco_dir}/jacoco.exec")
            # outputDirectory: where to write the report files (XML/HTML)
            cmd.append(f"-Djacoco.outputDirectory={jacoco_dir}")
        
        # Add Maven repository parameter if configured
        if self.maven_repo_dir:
            cmd.append(f"-Dmaven.repo.local={self.maven_repo_dir}")
        
        return cmd
    
    def _clean_old_jacoco_files(self, pom_path: Path):
        """
        Clean old JaCoCo files before running tests to avoid conflicts.
        This is similar to maven_parser.py's approach (lines 836-841).
        """
        try:
            # Default location for jacoco.exec
            jacoco_exec_path = pom_path.parent / "target" / "jacoco.exec"
            if jacoco_exec_path.exists():
                jacoco_exec_path.unlink()
                logger.debug(f"Deleted old jacoco.exec: {jacoco_exec_path}")
            
            # Default location for jacoco reports
            jacoco_report_dir = pom_path.parent / "target" / "site" / "jacoco"
            if jacoco_report_dir.exists():
                import shutil
                shutil.rmtree(jacoco_report_dir, ignore_errors=True)
                logger.debug(f"Deleted old jacoco report dir: {jacoco_report_dir}")
            
            # Custom location if using iteration directory
            if self.current_iteration_dir:
                jacoco_dir = self.current_iteration_dir / "jacoco"
                if jacoco_dir.exists():
                    for item in jacoco_dir.iterdir():
                        if item.is_file():
                            item.unlink()
                logger.debug(f"Cleaned custom jacoco dir: {jacoco_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean old JaCoCo files: {e}")
    
    def _prepare_pom_for_jacoco(self, pom_path: Path):
        """Prepare POM file for JaCoCo execution."""
        parent_pom = self.project_path / "pom.xml"
        
        # Remove unnecessary plugins
        self._remove_unnecessary_plugins(self.project_path / "pom.xml")
        self._remove_unnecessary_plugins(pom_path)
        
        # Disable existing JaCoCo plugin to avoid version conflict
        self._disable_jacoco_in_pom(pom_path)
        if parent_pom != pom_path and parent_pom.exists():
            self._disable_jacoco_in_pom(parent_pom)
        
        # Fix hardcoded argLine (must fix both child and parent POM)
        # This is CRITICAL: without @{argLine}, JaCoCo agent won't be injected
        self._fix_hardcoded_argline(pom_path)
        if parent_pom != pom_path and parent_pom.exists():
            self._fix_hardcoded_argline(parent_pom)
    
    def _disable_jacoco_in_pom(self, pom_path: Path):
        """
        Disable existing JaCoCo plugin in pom.xml to avoid version conflict.
        
        Strategy: Comment out the entire jacoco-maven-plugin configuration block.
        Note: Cannot use <skip>true</skip> as it would set jacoco.skip property,
              affecting the command-line invoked JaCoCo version (0.8.8).
        """
        if not pom_path.exists():
            return
        
        try:
            content = pom_path.read_text(encoding='utf-8')
        except Exception:
            return
        
        # Check if already commented out
        if 'DISABLED BY TEST FRAMEWORK' in content and 'jacoco-maven-plugin' in content:
            # Already disabled, no need to process again
            return
        
        # Find jacoco-maven-plugin configuration block
        # Use negative lookahead (?!</plugin>) to ensure <plugin> and <artifactId> 
        # are in the same plugin block
        pattern = r'<plugin>(?:(?!</plugin>)[\s\S])*?<artifactId>jacoco-maven-plugin</artifactId>[\s\S]*?</plugin>'
        
        def comment_out_jacoco_plugin(match):
            plugin_block = match.group(0)
            
            # Remove all XML comments inside the plugin block to avoid nested comments
            # XML spec doesn't allow -- character sequence inside comments
            plugin_block_no_comments = re.sub(r'<!--[\s\S]*?-->', '', plugin_block)
            
            # Comment out the entire plugin block
            commented_block = '<!-- DISABLED BY TEST FRAMEWORK TO AVOID VERSION CONFLICT -->\n<!--\n' + plugin_block_no_comments + '\n-->'
            return commented_block
        
        new_content = re.sub(pattern, comment_out_jacoco_plugin, content)
        
        if new_content != content:
            pom_path.write_text(new_content, encoding='utf-8')
            logger.info(f"Disabled JaCoCo plugin in {pom_path} to avoid version conflict")
    
    def _fix_hardcoded_argline(self, pom_path: Path):
        """Fix hardcoded argLine in pom.xml to support JaCoCo."""
        if not pom_path.exists():
            return
        
        try:
            content = pom_path.read_text(encoding='utf-8')
        except Exception:
            return
        
        # Pattern to find argLine without @{argLine}
        pattern = r'(<argLine>)(?!.*@\{argLine\})(.*?)(</argLine>)'
        
        def replace_argline(match):
            original_value = match.group(2).strip()
            if not original_value:
                return f'{match.group(1)}@{{argLine}}{match.group(3)}'
            return f'{match.group(1)}@{{argLine}} {match.group(2)}{match.group(3)}'
        
        new_content = re.sub(pattern, replace_argline, content)
        
        if new_content != content:
            pom_path.write_text(new_content, encoding='utf-8')
    
    def _remove_unnecessary_plugins(self, pom_path: Path):
        """Remove plugins that may interfere with test execution."""
        if not pom_path.exists():
            return
        
        try:
            content = pom_path.read_text(encoding='utf-8')
        except Exception:
            return
        
        plugins = [
            (r"org\.codehaus\.mojo", r"findbugs-maven-plugin"),
            (r"pl\.project13\.maven", r"git-commit-id-plugin"),
            (r"io\.github\.git-commit-id", r"git-commit-id-maven-plugin"),
        ]
        
        new_content = content
        for group_id, artifact_id in plugins:
            regex = r"<plugin>\s*<groupId>{}</groupId>\s*<artifactId>{}</artifactId>.*?</plugin>".format(
                group_id, artifact_id
            )
            match = re.search(regex, new_content, re.DOTALL)
            if match:
                new_content = new_content.replace(match.group(), "")
        
        if new_content != content:
            pom_path.write_text(new_content, encoding='utf-8')
    
    def _execute_maven_command(self, cmd: List[str]) -> Dict:
        """Execute Maven command with proper environment."""
        env = os.environ.copy()
        env['JAVA_HOME'] = self.java_home
        env['PATH'] = f"{self.java_home}\\bin;{env['PATH']}"
        
        logger.info(f"Executing: {' '.join(cmd)}")
        
        try:
            process = subprocess.run(
                cmd,
                cwd=str(self.project_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=config.java.test_timeout
            )
            
            output = process.stdout + "\n" + process.stderr
            
            return {
                'returncode': process.returncode,
                'output': output,
                'stdout': process.stdout,
                'stderr': process.stderr
            }
            
        except subprocess.TimeoutExpired as e:
            logger.error("Command timeout")
            raise
    
    def _is_compile_error(self, output: str) -> bool:
        """Check if output contains compilation error."""
        return "COMPILATION ERROR" in output or "Compilation failure:" in output
    
    def _extract_compile_error(self, output: str, test_rel_path: str = None) -> str:
        """
        Extract compilation error message (simplified version for LLM).
        Returns a concise summary of the compilation errors.
        Now includes column information if available.
        """
        # Find error lines with column numbers
        error_pattern_with_col = r'\[ERROR\]\s+(.+\.java):\[(\d+),(\d+)\]\s+(.+)'
        matches = re.findall(error_pattern_with_col, output)
        
        if matches:
            errors = []
            for file, line, col, message in matches[:5]:  # Limit to 5 errors
                errors.append(f"Line {line}, Column {col}: {message}")
            return "\n".join(errors)
        
        # Fallback: try without column numbers
        error_pattern_no_col = r'\[ERROR\]\s+(.+\.java):\[(\d+)\]\s+(.+)'
        matches_no_col = re.findall(error_pattern_no_col, output)
        
        if matches_no_col:
            errors = []
            for file, line, message in matches_no_col[:5]:
                errors.append(f"Line {line}: {message}")
            return "\n".join(errors)
        
        return "Compilation failed"
    
    def _extract_full_compile_error(self, output: str) -> str:
        """
        Extract full compilation error output for logging.
        Returns complete Maven compilation error section.
        """
        # Find the compilation error section
        lines = output.split('\n')
        error_section = []
        in_error_section = False
        
        for line in lines:
            # Start capturing at COMPILATION ERROR
            if '[ERROR] COMPILATION ERROR' in line or 'COMPILATION ERROR' in line:
                in_error_section = True
                error_section.append(line)
            # Continue capturing error lines
            elif in_error_section:
                error_section.append(line)
                # Stop at BUILD FAILURE or end of error section
                if '[INFO] BUILD FAILURE' in line:
                    # Continue a bit more to capture summary
                    continue
                elif line.startswith('[INFO] --------') and len(error_section) > 10:
                    # End of Maven output section
                    break
        
        if error_section:
            return '\n'.join(error_section)
        
        # Fallback: return all ERROR lines
        error_lines = [line for line in lines if '[ERROR]' in line]
        if error_lines:
            return '\n'.join(error_lines[:50])  # Limit to 50 lines
        
        return "Compilation failed (no detailed error output available)"
    
    def _extract_test_failure(self, output: str, test_class: str, test_method: str) -> str:
        """Extract test failure message (concise summary for error_message field)."""
        # 策略1: 尝试从surefire报告中提取精确的错误信息
        if test_class:
            # 从Maven输出中查找surefire-reports路径
            surefire_paths = []
            for line in output.split('\n'):
                if 'Please refer to' in line and 'surefire-reports' in line:
                    match = re.search(r'(/[^\s]+/surefire-reports)', line)
                    if match:
                        surefire_dir = Path(match.group(1))
                        surefire_txt = surefire_dir / f"{test_class}.txt"
                        surefire_paths.append(surefire_txt)
            
            # 回退：尝试项目根目录
            surefire_paths.append(self.project_path / "target" / "surefire-reports" / f"{test_class}.txt")
            
            # 尝试读取surefire报告
            for surefire_txt in surefire_paths:
                if surefire_txt.exists():
                    try:
                        with open(surefire_txt, 'r', encoding='utf-8') as f:
                            report_content = f.read()
                        
                        # 提取第一个失败的断言错误信息
                        # 查找类似 "org.junit.ComparisonFailure: expected:<...> but was:<...>" 的行
                        for line in report_content.split('\n'):
                            if any(keyword in line for keyword in ['AssertionError', 'ComparisonFailure', 'Exception']):
                                # 提取错误消息（去掉包名前缀）
                                error_msg = line.strip()
                                # 移除包名前缀（如 org.junit.ComparisonFailure:）
                                if ':' in error_msg:
                                    error_msg = error_msg.split(':', 1)[1].strip()
                                logger.info(f"Extracted error from surefire report: {error_msg[:100]}")
                                return error_msg[:500]  # 限制长度
                    except Exception as e:
                        logger.warning(f"Failed to read surefire report {surefire_txt}: {e}")
        
        # 策略2: 从Maven输出中提取（原有逻辑）
        # Extract simple class name
        simple_class = test_class.split('.')[-1]
        
        # Look for assertion errors
        failure_pattern = rf'{simple_class}\.{test_method}.*?(?:AssertionError|Exception):\s*(.+?)\n'
        match = re.search(failure_pattern, output, re.DOTALL)
        
        if match:
            return match.group(1)[:500]  # Limit length
        
        # Look for test summary
        summary_pattern = r'Tests run: (\d+), Failures: (\d+), Errors: (\d+)'
        match = re.search(summary_pattern, output)
        
        if match:
            runs, failures, errors = match.groups()
            return f"Tests run: {runs}, Failures: {failures}, Errors: {errors}"
        
        return "Test execution failed"
    
    def _check_surefire_for_failures(self, pom_path: Path, test_class: str, test_method: str) -> Optional[dict]:
        """
        Check surefire reports for test failures even when Maven returns 0.
        
        Some Maven/Surefire configurations may return exit code 0 even when tests have Errors.
        This method reads the surefire report to detect such cases.
        
        Args:
            pom_path: Path to pom.xml
            test_class: Full test class name
            test_method: Test method name
            
        Returns:
            Dict with error_message and full_output if test failed, None if test passed
        """
        if not test_class:
            return None
        
        # Find surefire report paths
        surefire_paths = []
        
        # Try module-specific path
        module_dir = pom_path.parent
        surefire_paths.append(module_dir / "target" / "surefire-reports" / f"{test_class}.txt")
        
        # Try project root path (fallback)
        surefire_paths.append(self.project_path / "target" / "surefire-reports" / f"{test_class}.txt")
        
        # Try to read surefire report
        for surefire_txt in surefire_paths:
            if surefire_txt.exists():
                try:
                    with open(surefire_txt, 'r', encoding='utf-8') as f:
                        report_content = f.read()
                    
                    # Check for failures or errors in the summary line
                    # Format: "Tests run: 1, Failures: 0, Errors: 1, Skipped: 0"
                    summary_pattern = r'Tests run: (\d+), Failures: (\d+), Errors: (\d+)'
                    match = re.search(summary_pattern, report_content)
                    
                    if match:
                        runs, failures, errors = match.groups()
                        failures_count = int(failures)
                        errors_count = int(errors)
                        
                        # If there are any failures or errors, test didn't pass
                        if failures_count > 0 or errors_count > 0:
                            logger.info(f"Detected test failure in surefire report: {failures_count} failures, {errors_count} errors")
                            
                            # Extract error message (first line of exception)
                            error_msg = f"Tests run: {runs}, Failures: {failures}, Errors: {errors}"
                            
                            # Try to extract exception details
                            exception_pattern = rf'{test_method}.*?<<< (ERROR|FAILURE)!\n(.+?)(?:\n\n|\n-{3,}|\Z)'
                            exc_match = re.search(exception_pattern, report_content, re.DOTALL)
                            if exc_match:
                                exception_details = exc_match.group(2).strip()
                                # Get first line of exception
                                first_line = exception_details.split('\n')[0]
                                error_msg = f"{error_msg} - {first_line[:200]}"
                            
                            return {
                                'error_message': error_msg,
                                'full_output': report_content
                            }
                    
                    # Also check for explicit FAILURE or ERROR markers
                    if '<<< FAILURE!' in report_content or '<<< ERROR!' in report_content:
                        logger.warning(f"Detected FAILURE/ERROR marker in surefire report but no summary match")
                        return {
                            'error_message': 'Test execution failed (see surefire report)',
                            'full_output': report_content
                        }
                    
                    # Test passed
                    logger.info(f"Test passed according to surefire report: {surefire_txt}")
                    return None
                    
                except Exception as e:
                    logger.warning(f"Failed to read surefire report {surefire_txt}: {e}")
        
        # No surefire report found - assume test passed (trust Maven's return code)
        logger.debug(f"No surefire report found for {test_class}, trusting Maven return code")
        return None
    
    def _extract_full_test_failure(self, output: str, test_class: str = None, test_method: str = None) -> str:
        """
        智能提取测试失败的完整输出，过滤掉无关信息。
        
        策略：
        1. 优先读取surefire报告文件（最准确）
        2. 从Maven输出中智能提取测试执行部分
        3. 过滤掉编译阶段的WARNING和无关INFO
        4. 只保留测试相关的输出、ERROR和堆栈跟踪
        
        Returns:
            精简但完整的测试失败信息（通常只有几百字符，而不是12万字符）
        """
        # 策略1: 优先尝试读取surefire文本报告
        if test_class:
            # 尝试多个可能的surefire报告位置
            # 1. 从Maven输出中提取实际的surefire-reports路径
            surefire_paths = []
            
            # 从Maven输出中查找surefire-reports路径提示
            for line in output.split('\n'):
                if 'Please refer to' in line and 'surefire-reports' in line:
                    # 提取路径: "Please refer to /path/to/surefire-reports for ..."
                    import re
                    match = re.search(r'(/[^\s]+/surefire-reports)', line)
                    if match:
                        surefire_dir = Path(match.group(1))
                        surefire_txt = surefire_dir / f"{test_class}.txt"
                        surefire_paths.append(surefire_txt)
                        logger.info(f"Found surefire-reports path from Maven output: {surefire_dir}")
            
            # 2. 回退：尝试项目根目录下的target（单模块项目）
            surefire_paths.append(self.project_path / "target" / "surefire-reports" / f"{test_class}.txt")
            
            # 3. 尝试所有可能的路径
            for surefire_txt in surefire_paths:
                if surefire_txt.exists():
                    try:
                        with open(surefire_txt, 'r', encoding='utf-8') as f:
                            report_content = f.read()
                        # 如果报告包含失败信息，直接返回
                        if 'FAILURE' in report_content or 'ERROR' in report_content:
                            logger.info(f"Read test failure from surefire report: {surefire_txt} ({len(report_content)} chars)")
                            return report_content
                    except Exception as e:
                        logger.warning(f"Failed to read surefire report {surefire_txt}: {e}")
        
        # 策略2: 从Maven输出中智能提取
        lines = output.split('\n')
        test_section = []
        in_test_section = False
        
        for i, line in enumerate(lines):
            # 开始标记：surefire插件执行测试
            if 'maven-surefire-plugin' in line and ':test' in line:
                in_test_section = True
                test_section.append(line)
                continue
            
            # 如果在测试部分，选择性保留行
            if in_test_section:
                # 保留关键的INFO（过滤掉其他INFO）
                if line.startswith('[INFO]'):
                    # 只保留这些关键信息
                    if any(keyword in line for keyword in [
                        'Running', 'Tests run:', 'BUILD FAILURE', 
                        'Total time:', 'Finished at:', 'surefire'
                    ]):
                        test_section.append(line)
                # 保留所有ERROR和WARNING（在测试阶段的）
                elif line.startswith('[ERROR]') or (line.startswith('[WARNING]') and 'test' in line.lower()):
                    test_section.append(line)
                # 保留测试输出和堆栈跟踪（不以[开头的行）
                elif line.strip() and not line.startswith('['):
                    test_section.append(line)
                # 保留空行以保持格式（但限制连续空行）
                elif line.strip() == '' and test_section and test_section[-1].strip() != '':
                    test_section.append(line)
                
                # 结束标记：BUILD FAILURE之后
                if '[INFO] BUILD FAILURE' in line:
                    # 继续收集几行总结信息
                    for j in range(i+1, min(i+15, len(lines))):
                        next_line = lines[j]
                        if next_line.startswith('[INFO]'):
                            test_section.append(next_line)
                        # 到达分隔线，停止
                        if '--------' in next_line and len(test_section) > 10:
                            break
                    break
        
        if test_section:
            result = '\n'.join(test_section)
            logger.info(f"Extracted {len(test_section)} lines of test failure info ({len(result)} chars)")
            return result
        
        # 策略3: 回退方案 - 提取所有ERROR和测试相关行
        error_and_test_lines = []
        for line in lines:
            if any(keyword in line for keyword in [
                '[ERROR]', 'AssertionError', 'Exception:', 'Tests run:', 
                'Failures:', 'at org.junit', 'at sun.reflect', test_method if test_method else ''
            ]):
                error_and_test_lines.append(line)
        
        if error_and_test_lines:
            result = '\n'.join(error_and_test_lines[:100])  # 限制100行
            logger.info(f"Fallback: extracted {len(error_and_test_lines)} error lines")
            return result
        
        # 最后的回退：返回后5000字符（测试失败信息通常在输出末尾）
        logger.warning("Could not intelligently extract test failure, returning last 5000 chars")
        return output[-5000:]

    def _parse_jacoco_report(self, pom_path: Path, focal_method_info: FocalMethodInfo = None) -> CoverageInfo:
        """
        Parse JaCoCo XML report to extract coverage information.
        
        Args:
            pom_path: Path to the pom.xml
            focal_method_info: Information about focal method for filtering
            
        Returns:
            CoverageInfo with overall and detailed coverage
        """
        # JaCoCo report is always generated in the default location by Maven
        # We'll read it from there, then copy to custom directory if needed
        jacoco_xml = pom_path.parent / "target" / "site" / "jacoco" / "jacoco.xml"
        
        if not jacoco_xml.exists():
            logger.warning(f"JaCoCo report not found: {jacoco_xml}")
            return CoverageInfo()
        
        try:
            tree = ET.parse(jacoco_xml)
            root = tree.getroot()
            
            # Extract coverage metrics (overall)
            covered_lines = []
            total_lines = 0
            total_branches = 0
            covered_branches_count = 0
            
            # Detailed coverage for focal method
            focal_line_coverages = []
            
            # Get focal method info for filtering
            focal_start = focal_method_info.start_line if focal_method_info else 0
            focal_end = focal_method_info.end_line if focal_method_info else 0
            focal_class = focal_method_info.class_name if focal_method_info else ""
            focal_method_name = focal_method_info.name if focal_method_info else ""
            
            # Convert class name to package path format: cn.hutool.db.sql.Condition -> cn/hutool/db/sql
            focal_package = "/".join(focal_class.rsplit(".", 1)[0].split(".")) if focal_class else ""
            focal_source_file = focal_class.rsplit(".", 1)[-1] + ".java" if focal_class else ""
            
            logger.info(f"Looking for focal method coverage: {focal_class}.{focal_method_name} (lines {focal_start}-{focal_end})")
            
            # Find focal method coverage only (not entire project)
            for package in root.findall('.//package'):
                pkg_name = package.get('name', '')
                
                for sourcefile in package.findall('sourcefile'):
                    sf_name = sourcefile.get('name', '')
                    
                    # Check if this is the focal method's source file
                    is_focal_source = (pkg_name == focal_package and sf_name == focal_source_file)
                    
                    # Only process lines if this is the focal method's source file
                    if is_focal_source:
                        for line in sourcefile.findall('line'):
                            line_num = int(line.get('nr', 0))
                            ci = int(line.get('ci', 0))  # covered instructions
                            mi = int(line.get('mi', 0))  # missed instructions
                            cb = int(line.get('cb', 0))  # covered branches
                            mb = int(line.get('mb', 0))  # missed branches
                            
                            # Only count lines within focal method range
                            if focal_start <= line_num <= focal_end:
                                # Count lines (any line with instructions)
                                if ci > 0 or mi > 0:
                                    total_lines += 1
                                    if ci > 0:
                                        covered_lines.append(line_num)
                                
                                # Count branches
                                if cb > 0 or mb > 0:
                                    total_branches += (cb + mb)
                                    covered_branches_count += cb
                                
                                # Collect detailed coverage for focal method
                                if ci > 0 or mi > 0:  # Only lines with instructions
                                    line_cov = LineCoverage(
                                        line_number=line_num,
                                        is_covered=(ci > 0),
                                        covered_instructions=ci,
                                        missed_instructions=mi,
                                        covered_branches=cb,
                                        missed_branches=mb
                                    )
                                    focal_line_coverages.append(line_cov)
            
            # If we couldn't find by package/class match, try to find by method line range
            if not focal_line_coverages and focal_method_info:
                logger.info("Trying to match focal method by line range in all classes...")
                focal_line_coverages = self._find_focal_coverage_by_line_range(
                    root, focal_method_info
                )
                
                # Recalculate metrics from focal_line_coverages
                if focal_line_coverages:
                    covered_lines = []
                    total_lines = 0
                    total_branches = 0
                    covered_branches_count = 0
                    
                    for line_cov in focal_line_coverages:
                        total_lines += 1
                        if line_cov.is_covered:
                            covered_lines.append(line_cov.line_number)
                        
                        if line_cov.has_branch:
                            total_branches += (line_cov.covered_branches + line_cov.missed_branches)
                            covered_branches_count += line_cov.covered_branches
            
            # Calculate coverage percentages
            line_coverage_percentage = (len(covered_lines) / total_lines * 100) if total_lines > 0 else 0
            branch_coverage_percentage = (covered_branches_count / total_branches * 100) if total_branches > 0 else None
            
            # Create detailed coverage info
            detailed_coverage = DetailedCoverageInfo(
                line_coverages=sorted(focal_line_coverages, key=lambda x: x.line_number)
            ) if focal_line_coverages else None
            
            if detailed_coverage:
                logger.info(f"Focal method line coverage: {len(covered_lines)}/{total_lines} lines ({line_coverage_percentage:.2f}%)")
                if total_branches > 0:
                    logger.info(f"Focal method branch coverage: {covered_branches_count}/{total_branches} branches ({branch_coverage_percentage:.2f}%)")
                else:
                    logger.info(f"Focal method branch coverage: None (no branches)")
            
            return CoverageInfo(
                covered_lines=covered_lines,
                covered_branches=[],
                total_lines=total_lines,
                total_branches=total_branches,
                coverage_percentage=line_coverage_percentage,  # 向后兼容，使用行覆盖率
                line_coverage_percentage=line_coverage_percentage,
                branch_coverage_percentage=branch_coverage_percentage,
                covered_lines_count=len(covered_lines),
                covered_branches_count=covered_branches_count,
                detailed_coverage=detailed_coverage
            )
            
        except Exception as e:
            logger.error(f"Failed to parse JaCoCo report: {e}")
            import traceback
            traceback.print_exc()
            return CoverageInfo()
    
    def _find_focal_coverage_by_line_range(self, root: ET.Element, 
                                           focal_method_info: FocalMethodInfo) -> List[LineCoverage]:
        """
        Find focal method coverage by matching line range in classes.
        Used as fallback when package/class name matching fails.
        
        Args:
            root: XML root element
            focal_method_info: Focal method information
            
        Returns:
            List of LineCoverage for the focal method
        """
        focal_start = focal_method_info.start_line
        focal_end = focal_method_info.end_line
        focal_method_name = focal_method_info.name
        
        focal_line_coverages = []
        
        for package in root.findall('.//package'):
            for cls in package.findall('class'):
                # Try to match by method name first
                matched_method = None
                for method in cls.findall('method'):
                    method_name = method.get('name', '')
                    method_line = int(method.get('line', 0))
                    
                    # Match by name and check if line is in range
                    if method_name == focal_method_name:
                        if focal_start <= method_line <= focal_end:
                            matched_method = method
                            break
                
                # If no method name match, try to match by line range only
                if not matched_method:
                    for method in cls.findall('method'):
                        method_line = int(method.get('line', 0))
                        if focal_start <= method_line <= focal_end:
                            matched_method = method
                            break
                
                if matched_method:
                    # Found the method, get the corresponding sourcefile
                    sf_name = cls.get('sourcefilename', '')
                    
                    for sourcefile in package.findall('sourcefile'):
                        if sourcefile.get('name') == sf_name:
                            for line in sourcefile.findall('line'):
                                line_num = int(line.get('nr', 0))
                                if focal_start <= line_num <= focal_end:
                                    ci = int(line.get('ci', 0))
                                    mi = int(line.get('mi', 0))
                                    cb = int(line.get('cb', 0))
                                    mb = int(line.get('mb', 0))
                                    
                                    if ci > 0 or mi > 0:
                                        line_cov = LineCoverage(
                                            line_number=line_num,
                                            is_covered=(ci > 0),
                                            covered_instructions=ci,
                                            missed_instructions=mi,
                                            covered_branches=cb,
                                            missed_branches=mb
                                        )
                                        focal_line_coverages.append(line_cov)
                            
                            logger.info(f"Found focal method coverage via line range: {len(focal_line_coverages)} lines")
                            return focal_line_coverages
        
        return focal_line_coverages
    
    def _copy_jacoco_reports(self, pom_path: Path):
        """
        Copy JaCoCo reports from default Maven location to custom directory.
        
        JaCoCo Maven plugin always generates reports in target/site/jacoco/,
        and the outputDirectory parameter doesn't work for command-line execution.
        So we copy the reports to our custom directory after generation.
        
        Args:
            pom_path: Path to the pom.xml
        """
        sample_logger = get_sample_logger()
        
        if not self.current_iteration_dir:
            return
        
        try:
            # Source: Maven default location
            source_dir = pom_path.parent / "target" / "site" / "jacoco"
            
            # Destination: our custom directory
            dest_dir = self.current_iteration_dir / "jacoco"
            
            if not source_dir.exists():
                sample_logger.log_warning(f"[JaCoCo] Source report directory not found: {source_dir}")
                return
            
            # Copy all files from source to destination
            import shutil
            for item in source_dir.iterdir():
                if item.is_file():
                    dest_file = dest_dir / item.name
                    shutil.copy2(item, dest_file)
                    logger.debug(f"Copied {item.name} to {dest_dir}")
                elif item.is_dir():
                    # Copy subdirectories (like .resources/)
                    dest_subdir = dest_dir / item.name
                    if dest_subdir.exists():
                        shutil.rmtree(dest_subdir)
                    shutil.copytree(item, dest_subdir)
                    logger.debug(f"Copied directory {item.name} to {dest_dir}")
            
            sample_logger.log_info(f"[JaCoCo] Reports copied to custom directory: {dest_dir}")
            
        except Exception as e:
            logger.error(f"Failed to copy JaCoCo reports: {e}")
            sample_logger.log_warning(f"[JaCoCo] Failed to copy reports: {e}")
    
    def _run_pitest(self, test_class: str, test_method: str, 
                   focal_method_info: FocalMethodInfo, test_rel_path: str = None) -> Dict:
        """
        Run PITest for mutation testing.
        
        Returns:
            Dict with mutation_info
        """
        sample_logger = get_sample_logger()
        
        try:
            # Find parent POM for the test
            if test_rel_path:
                test_path = self.project_path / test_rel_path
                pom_path = self._find_parent_pom(test_path)
            else:
                pom_path = self.project_path / "pom.xml"
            
            if not pom_path or not pom_path.exists():
                sample_logger.log_error("[PITest] pom.xml not found")
                return {
                    'status': 'ERROR',
                    'mutation_info': MutationInfo()
                }
            
            sample_logger.log_info(f"[PITest] Using POM: {pom_path.relative_to(self.project_path)}")
            
            # Build dependencies first (install without tests)
            # This ensures all dependent modules are available in the local Maven repository
            # Note: JaCoCo (Phase 1) has already compiled and tested the code, so we only need
            # to ensure dependencies are installed for PITest to work correctly
            pom_module = str(pom_path.parent.relative_to(self.project_path))
            maven_script = "mvn.cmd" if platform.system() == "Windows" else "mvn"
            maven_cmd = str(Path(self.maven_home) / "bin" / maven_script)
            
            install_cmd = [
                maven_cmd,
                "-U",  # Force update snapshots (consistent with TestUpdater)
                "clean",  # Clean before build to avoid stale artifacts
                "install",
                "-nsu",
                "-pl", pom_module,  # Must be separate arguments for subprocess
                "--also-make",  # Build all dependency modules
                "-DskipTests",  # Skip test execution (already done in Phase 1)
                "--batch-mode",
            ]
            
            # Add skip options (exclude -Dmaven.test.skip and -DskipTests as they're already added)
            install_cmd.extend([skip for skip in MVN_SKIPS if skip not in ["-Dmaven.test.skip", "-DskipTests"]])
            
            if self.maven_repo_dir:
                install_cmd.append(f"-Dmaven.repo.local={self.maven_repo_dir}")
            
            sample_logger.log_info("[PITest] Building dependencies (install with --also-make)...")
            sample_logger.log_info(f"[PITest] Install command: {' '.join(install_cmd)}")
            install_result = self._execute_maven_command(install_cmd)
            
            if install_result['returncode'] != 0:
                sample_logger.log_error(f"[PITest] Dependency build failed with return code: {install_result['returncode']}")
                sample_logger.log_error("[PITest] Maven install output (last 100 lines):")
                output_lines = install_result['output'].split('\n')
                for line in output_lines[-100:]:
                    sample_logger.log_error(f"  {line}")
                return {
                    'status': 'ERROR',
                    'mutation_info': MutationInfo()
                }
            else:
                sample_logger.log_info("[PITest] Dependencies built and installed successfully")
            
            # Build PITest command
            cmd = self._build_pitest_command(test_class, test_method, focal_method_info, pom_path)
            sample_logger.log_info(f"[PITest] Maven command: {' '.join(cmd)}")
            
            # Execute command
            sample_logger.log_info("[PITest] Starting Maven execution...")
            result = self._execute_maven_command(cmd)
            sample_logger.log_info(f"[PITest] Maven execution completed with return code: {result['returncode']}")
            
            # Log Maven output if PITest failed or for debugging
            if result['returncode'] != 0:
                sample_logger.log_error("[PITest] Maven execution failed!")
                sample_logger.log_error("[PITest] Maven output (last 200 lines):")
                output_lines = result['output'].split('\n')
                for line in output_lines[-200:]:
                    sample_logger.log_error(f"  {line}")
            
            # Parse PITest report
            sample_logger.log_info("[PITest] Parsing mutation report...")
            mutation_info = self._parse_pitest_report(pom_path, focal_method_info)
            sample_logger.log_info(f"[PITest] Mutations parsed: {mutation_info.kill_percentage:.2f}% kill rate ({len(mutation_info.killed_mutations)}/{mutation_info.total_mutations} killed)")
            
            # If no mutations found, log detailed Maven output for debugging
            if mutation_info.total_mutations == 0:
                sample_logger.log_warning("[PITest] No mutations found! This may indicate a problem.")
                sample_logger.log_warning("[PITest] Maven output (last 200 lines for debugging):")
                output_lines = result['output'].split('\n')
                for line in output_lines[-200:]:
                    sample_logger.log_warning(f"  {line}")
                
                # Also check if PITest report directory exists
                if self.current_iteration_dir:
                    pitest_dir = self.current_iteration_dir / "pitest"
                    sample_logger.log_warning(f"[PITest] Checking report directory: {pitest_dir}")
                    if pitest_dir.exists():
                        sample_logger.log_warning(f"[PITest] Report directory exists, contents:")
                        for item in pitest_dir.iterdir():
                            sample_logger.log_warning(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
                    else:
                        sample_logger.log_warning(f"[PITest] Report directory does NOT exist!")
            
            return {
                'status': 'SUCCESS',
                'mutation_info': mutation_info
            }
            
        except Exception as e:
            logger.error(f"PITest execution error: {e}")
            sample_logger.log_error(f"[PITest] Execution error: {e}")
            return {
                'status': 'ERROR',
                'mutation_info': MutationInfo()
            }
    
    def _build_pitest_command(self, test_class: str, test_method: str, 
                             focal_method_info: FocalMethodInfo, pom_path: Path) -> List[str]:
        """Build Maven command for PITest execution."""
        # Use platform-specific Maven script
        maven_script = "mvn.cmd" if platform.system() == "Windows" else "mvn"
        maven_cmd = str(Path(self.maven_home) / "bin" / maven_script)
        
        # Determine module path
        pom_module = str(pom_path.parent.relative_to(self.project_path))
        
        # Get simple class name
        simple_class = test_class.split('.')[-1]
        
        cmd = [
            maven_cmd,
            "org.pitest:pitest-maven:1.9.11:mutationCoverage",
            "-nsu",
            "-pl", pom_module,  # Must be separate arguments for subprocess
            "-DfailIfNoTests=false",
            "-DskipFailingTests=true",
            "-Dmaven.test.skip=false",
            "-DskipTests=false",
            f"-DincludedTestMethods={test_method}",
            f"-DtargetTests={test_class}",
            "--batch-mode",
            "-DoutputFormats=XML",
            "-Dpitest.mutators=ALL",
            "-DtimeoutConst=10000",
            "-Dthreads=1",
            "-X",  # Debug mode (consistent with TestUpdater)
        ]
        
        # Avoid surefire ${argLine} placeholder issue
        cmd.append("-DargLine=")
        
        # Add targetClasses parameter to limit mutation analysis to focal method's class
        # This ensures PITest only mutates the focal method's class, not all classes
        if focal_method_info and focal_method_info.class_name:
            cmd.append(f"-DtargetClasses={focal_method_info.class_name}")
        
        # Add custom PITest reports directory if current_iteration_dir is configured
        if self.current_iteration_dir:
            pitest_dir = self.current_iteration_dir / "pitest"
            pitest_dir.mkdir(parents=True, exist_ok=True)
            cmd.append(f"-DreportsDirectory={pitest_dir}")
        
        # Add skip options (exclude -Dmaven.test.skip and -DskipTests as they're already added)
        cmd.extend([skip for skip in MVN_SKIPS if skip not in ["-Dmaven.test.skip", "-DskipTests"]])
        
        # Add Maven repository parameter if configured
        if self.maven_repo_dir:
            cmd.append(f"-Dmaven.repo.local={self.maven_repo_dir}")
        
        return cmd
    
    def _parse_pitest_report(self, pom_path: Path, focal_method_info: FocalMethodInfo) -> MutationInfo:
        """
        Parse PITest XML report to extract mutation information.
        
        Args:
            pom_path: Path to the pom.xml
            focal_method_info: Information about focal method for filtering
            
        Returns:
            MutationInfo with overall and detailed mutation info
        """
        # Try custom reports directory first, then fall back to default location
        if self.current_iteration_dir:
            pit_reports_dir = self.current_iteration_dir / "pitest"
        else:
            pit_reports_dir = pom_path.parent / "target" / "pit-reports"
        
        sample_logger = get_sample_logger()
        
        if not pit_reports_dir.exists():
            logger.warning(f"PITest reports directory not found: {pit_reports_dir}")
            sample_logger.log_warning(f"[PITest] Reports directory not found: {pit_reports_dir}")
            return MutationInfo()
        
        # Log directory contents for debugging
        sample_logger.log_info(f"[PITest] Searching for mutations.xml in: {pit_reports_dir}")
        dir_contents = list(pit_reports_dir.iterdir())
        sample_logger.log_info(f"[PITest] Directory contains {len(dir_contents)} items:")
        for item in dir_contents:
            sample_logger.log_info(f"  - {item.name} ({'dir' if item.is_dir() else 'file'})")
        
        # Find the latest report
        mutations_xml = None
        for report_dir in sorted(pit_reports_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if report_dir.is_dir():
                xml_file = report_dir / "mutations.xml"
                if xml_file.exists():
                    mutations_xml = xml_file
                    sample_logger.log_info(f"[PITest] Found mutations.xml in subdirectory: {report_dir.name}")
                    break
        
        # If no subdirectory found, try direct mutations.xml in pitest directory
        if not mutations_xml:
            direct_xml = pit_reports_dir / "mutations.xml"
            if direct_xml.exists():
                mutations_xml = direct_xml
                sample_logger.log_info(f"[PITest] Found mutations.xml directly in pitest directory")
        
        if not mutations_xml:
            logger.warning("PITest mutations.xml not found")
            sample_logger.log_warning("[PITest] mutations.xml not found in any location!")
            return MutationInfo()
        
        try:
            tree = ET.parse(mutations_xml)
            root = tree.getroot()
            
            killed_mutations = []
            total_mutations = 0
            
            # Detailed mutation info for focal method
            focal_mutation_details = []
            
            # Get focal method info for filtering
            focal_start = focal_method_info.start_line if focal_method_info else 0
            focal_end = focal_method_info.end_line if focal_method_info else 0
            focal_method_name = focal_method_info.name if focal_method_info else ""
            
            logger.info(f"Looking for focal method mutations: {focal_method_name} (lines {focal_start}-{focal_end})")
            sample_logger.log_info(f"[PITest] Parsing mutations for focal method: {focal_method_name} (lines {focal_start}-{focal_end})")
            
            # Parse mutation elements
            mutation_id = 0
            all_mutations_count = len(root.findall('.//mutation'))
            sample_logger.log_info(f"[PITest] Total mutations in XML: {all_mutations_count}")
            
            for mutation in root.findall('.//mutation'):
                mutation_id += 1
                status = mutation.get('status', '')
                detected = mutation.get('detected', 'false')
                
                # Extract mutation details
                mutated_method = mutation.find('mutatedMethod')
                line_number_elem = mutation.find('lineNumber')
                mutator_elem = mutation.find('mutator')
                description_elem = mutation.find('description')
                killing_test_elem = mutation.find('killingTest')
                
                mutated_method_name = mutated_method.text if mutated_method is not None else ""
                line_number = int(line_number_elem.text) if line_number_elem is not None else 0
                mutator = mutator_elem.text if mutator_elem is not None else ""
                description = description_elem.text if description_elem is not None else ""
                killing_test = killing_test_elem.text if killing_test_elem is not None and killing_test_elem.text else ""
                
                total_mutations += 1
                
                if status == 'KILLED' or detected == 'true':
                    killed_mutations.append(str(mutation_id))
                
                # Check if this mutation belongs to focal method
                is_focal_mutation = False
                
                # Method 1: Match by method name + line range
                if focal_method_name and mutated_method_name == focal_method_name:
                    if focal_start <= line_number <= focal_end:
                        is_focal_mutation = True
                
                # Method 2: If method name doesn't match, try line range only
                if not is_focal_mutation and focal_start > 0 and focal_end > 0:
                    if focal_start <= line_number <= focal_end:
                        is_focal_mutation = True
                
                if is_focal_mutation:
                    mutation_detail = MutationDetail(
                        mutation_id=str(mutation_id),
                        line_number=line_number,
                        mutator=mutator,
                        description=description,
                        status=status,
                        killing_test=killing_test
                    )
                    focal_mutation_details.append(mutation_detail)
            
            # Create detailed mutation info
            detailed_mutations = DetailedMutationInfo(
                mutations=sorted(focal_mutation_details, key=lambda x: x.line_number)
            ) if focal_mutation_details else None
            
            # Calculate kill_percentage based on focal method mutations only
            if detailed_mutations and len(detailed_mutations.mutations) > 0:
                focal_total = len(detailed_mutations.mutations)
                focal_killed = len(detailed_mutations.killed_mutations)
                kill_percentage = (focal_killed / focal_total * 100)
                # Update total_mutations and killed_mutations to reflect focal method only
                total_mutations = focal_total
                killed_mutations = [m.mutation_id for m in detailed_mutations.killed_mutations]
                
                logger.info(f"Focal method mutations: {focal_killed} killed, "
                           f"{len(detailed_mutations.survived_mutations)} survived, "
                           f"{len(detailed_mutations.no_coverage_mutations)} no coverage, "
                           f"kill rate: {kill_percentage:.1f}%")
            else:
                # Fallback: if no focal mutations found, use all mutations
                kill_percentage = (len(killed_mutations) / total_mutations * 100) if total_mutations > 0 else 0
                logger.warning(f"No focal method mutations found, using all mutations: {len(killed_mutations)}/{total_mutations}")
            
            return MutationInfo(
                killed_mutations=killed_mutations,
                total_mutations=total_mutations,
                kill_percentage=kill_percentage,
                detailed_mutations=detailed_mutations
            )
            
        except Exception as e:
            logger.error(f"Failed to parse PITest report: {e}")
            import traceback
            traceback.print_exc()
            return MutationInfo()
    
    def _find_parent_pom(self, test_path: Path) -> Optional[Path]:
        """Find the parent pom.xml for a test file."""
        current = test_path.parent
        while current != self.project_path and current.parent != current:
            pom = current / "pom.xml"
            if pom.exists():
                return pom
            current = current.parent
        return self.project_path / "pom.xml" if (self.project_path / "pom.xml").exists() else None
    
    def _create_result(self, status: TestResultStatus, test_code: str,
                      test_class: str, test_method: str,
                      focal_method_info: FocalMethodInfo,
                      error_message: Optional[str],
                      coverage_info: CoverageInfo,
                      mutation_info: MutationInfo,
                      raw_error_output: Optional[str] = None,
                      test_file_path: Optional[Path] = None,
                      class_fields: List[str] = None,
                      non_test_methods: List[Dict] = None,
                      test_method_start_line: int = 0) -> TestResultInfo:
        """Create TestResultInfo object."""
        # Extract test_imports from the actual test file if available
        test_imports = []
        if test_file_path and test_file_path.exists():
            test_imports = self._extract_imports_from_file(test_file_path)
        
        test_case = TestCase(
            name=f"{test_class}.{test_method}",
            code=test_code,
            focal_method=focal_method_info.name,
            coverage_info=coverage_info,
            mutation_info=mutation_info,
            test_imports=test_imports,
            class_fields=class_fields or [],
            non_test_methods=non_test_methods or []
        )
        
        return TestResultInfo(
            status=status,
            test_case=test_case,
            focal_method_info=focal_method_info,
            error_message=error_message,
            raw_error_output=raw_error_output,
            test_imports=test_imports,
            test_method_start_line=test_method_start_line
        )
    
    def _extract_imports_from_file(self, file_path: Path) -> List[str]:
        """
        Extract import statements from a Java file.
        
        Args:
            file_path: Path to the Java file
            
        Returns:
            List of import statements (e.g., ["import com.example.Class;"])
        """
        try:
            content = file_path.read_text(encoding='utf-8')
            imports = []
            for line in content.split('\n'):
                stripped = line.strip()
                if stripped.startswith('import ') and stripped.endswith(';'):
                    imports.append(stripped)
            return imports
        except Exception as e:
            logger.warning(f"Failed to extract imports from {file_path}: {e}")
            return []


# Helper functions for repository management

def ensure_repository(project_name: str, repo_url: str = None) -> Path:
    """
    Ensure repository exists locally, clone if necessary.
    
    Args:
        project_name: Project name (e.g., "dromara/hutool")
        repo_url: Repository URL (optional, will construct from project_name if not provided)
        
    Returns:
        Path to the repository
    """
    repos_dir = Path(config.java.repos_dir)
    repos_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct repo path
    repo_path = repos_dir / project_name.replace('/', os.sep)
    
    if repo_path.exists():
        logger.info(f"Repository already exists: {repo_path}")
        return repo_path
    
    # Clone repository
    if repo_url is None:
        repo_url = f"https://github.com/{project_name}.git"
    
    logger.info(f"Cloning repository: {repo_url}")
    
    # Use GitHub token if available
    if config.java.github_tokens:
        token = config.java.github_tokens[0]
        # Insert token into URL
        repo_url = repo_url.replace("https://", f"https://{token}@")
    
    # Retry logic for network failures
    max_retries = 3
    retry_delay = 2
    
    for attempt in range(max_retries):
        try:
            result = subprocess.run(
                ['git', 'clone', repo_url, str(repo_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            logger.info(f"Repository cloned successfully: {repo_path}")
            return repo_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository (attempt {attempt + 1}/{max_retries}): {e.stderr}")
            # Clean up partial clone directory
            if repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)
            
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise
        except subprocess.TimeoutExpired as e:
            logger.error(f"Clone timeout (attempt {attempt + 1}/{max_retries})")
            # Clean up partial clone directory
            if repo_path.exists():
                shutil.rmtree(repo_path, ignore_errors=True)
            
            if attempt < max_retries - 1:
                wait_time = retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                raise


def detect_java_version(project_path: Path) -> str:
    """
    Detect Java version from pom.xml.
    
    Args:
        project_path: Path to the project
        
    Returns:
        Java version as string (8, 11, 17, 21)
    """
    pom_xml = project_path / "pom.xml"
    
    if not pom_xml.exists():
        logger.warning("pom.xml not found, defaulting to Java 8")
        return "8"
    
    try:
        tree = ET.parse(pom_xml)
        root = tree.getroot()
        
        # Handle XML namespaces
        ns = {'maven': 'http://maven.apache.org/POM/4.0.0'}
        
        # Try to find maven.compiler.source
        for prop in root.findall('.//maven:properties/maven:maven.compiler.source', ns):
            version = prop.text
            if version:
                # Extract major version
                if '1.8' in version or version == '8':
                    return '8'
                elif '11' in version:
                    return '11'
                elif '17' in version:
                    return '17'
                elif '21' in version:
                    return '21'
        
        # Try without namespace
        for prop in root.findall('.//properties/maven.compiler.source'):
            version = prop.text
            if version:
                if '1.8' in version or version == '8':
                    return '8'
                elif '11' in version:
                    return '11'
                elif '17' in version:
                    return '17'
                elif '21' in version:
                    return '21'
        
        logger.warning("Could not detect Java version from pom.xml, defaulting to Java 8")
        return "8"
        
    except Exception as e:
        logger.error(f"Failed to parse pom.xml: {e}")
        return "8"


# ============================================================
# Annotated Code Generation Functions
# ============================================================

def generate_coverage_annotated_code(focal_code: str, 
                                     start_line: int, 
                                     detailed_coverage: DetailedCoverageInfo) -> str:
    """
    Generate focal method code with coverage annotations.
    
    Each line will be annotated with line number and coverage status:
    - Line numbers are shown at the start of each line (e.g., "131|")
    - ✅ COVERED: Line is covered
    - ❌ NOT_COVERED: Line is not covered
    - ⚠️ BRANCH: Branch coverage info (e.g., 2/4 branches covered)
    
    Args:
        focal_code: The focal method source code
        start_line: The starting line number in the source file
        detailed_coverage: Detailed coverage information
        
    Returns:
        Annotated focal method code with line numbers
    """
    if not detailed_coverage or not detailed_coverage.line_coverages:
        # No coverage data, return code with "no data" annotations
        lines = focal_code.split('\n')
        result_lines = []
        for i, line in enumerate(lines):
            line_num = start_line + i
            result_lines.append(f"{line_num:4d}| {line}  // ⚪ NO_DATA")
        return '\n'.join(result_lines)
    
    lines = focal_code.split('\n')
    result_lines = []
    
    for i, line in enumerate(lines):
        line_num = start_line + i
        line_cov = detailed_coverage.get_line_coverage(line_num)
        
        if line_cov:
            # Build annotation
            annotations = []
            
            # Coverage status
            if line_cov.is_covered:
                annotations.append("✅ COVERED")
            else:
                annotations.append("❌ NOT_COVERED")
            
            # Branch info
            if line_cov.has_branch:
                total_branches = line_cov.covered_branches + line_cov.missed_branches
                if line_cov.missed_branches > 0:
                    annotations.append(f"⚠️ BRANCH: {line_cov.covered_branches}/{total_branches} covered")
                else:
                    annotations.append(f"✅ BRANCH: {line_cov.covered_branches}/{total_branches} covered")
            
            annotation_str = " | ".join(annotations)
            result_lines.append(f"{line_num:4d}| {line}  // {annotation_str}")
        else:
            # Line not in coverage report (likely comment, blank, or declaration)
            stripped = line.strip()
            if stripped and not stripped.startswith('//') and not stripped.startswith('/*') and not stripped.startswith('*'):
                result_lines.append(f"{line_num:4d}| {line}  // ⚪ NO_INSTRUCTION")
            else:
                result_lines.append(f"{line_num:4d}| {line}")
    
    return '\n'.join(result_lines)


def generate_mutation_annotated_code(focal_code: str, 
                                     start_line: int, 
                                     detailed_mutations: DetailedMutationInfo) -> str:
    """
    Generate focal method code with mutation annotations.
    
    Each line is prefixed with line number (e.g., "131|").
    Mutations are annotated inline at the end of the mutated line:
    - 🟢 KILLED: Mutation was killed by the test
    - 🔴 SURVIVED: Mutation survived (test didn't detect it)
    - ⚪ NO_COVERAGE: Mutation not covered by test
    
    Args:
        focal_code: The focal method source code
        start_line: The starting line number in the source file
        detailed_mutations: Detailed mutation information
        
    Returns:
        Annotated focal method code with line numbers
    """
    if not detailed_mutations or not detailed_mutations.mutations:
        # No mutation data, return code with line numbers
        lines = focal_code.split('\n')
        result_lines = []
        for i, line in enumerate(lines):
            line_num = start_line + i
            result_lines.append(f"{line_num:4d}| {line}")
        return '\n'.join(result_lines)
    
    lines = focal_code.split('\n')
    
    # Group mutations by line number
    mutations_by_line: Dict[int, List[MutationDetail]] = {}
    for mutation in detailed_mutations.mutations:
        line_num = mutation.line_number
        if line_num not in mutations_by_line:
            mutations_by_line[line_num] = []
        mutations_by_line[line_num].append(mutation)
    
    result_lines = []
    
    for i, line in enumerate(lines):
        line_num = start_line + i
        
        # Add mutation annotations inline at the end of the line if any
        if line_num in mutations_by_line:
            mutations = mutations_by_line[line_num]
            mutation_annotations = []
            for mutation in mutations:
                # Status icon
                if mutation.is_killed:
                    status_icon = "🟢 KILLED"
                elif mutation.is_survived:
                    status_icon = "🔴 SURVIVED"
                else:
                    status_icon = "⚪ NO_COVERAGE"
                
                # Get simple mutator name
                mutator_name = mutation.mutator_simple_name
                
                # Build annotation
                mutation_annotations.append(f"{status_icon} [{mutator_name}] {mutation.description}")
            
            # Join all mutations for this line
            annotation_str = " | ".join(mutation_annotations)
            result_lines.append(f"{line_num:4d}| {line}  // {annotation_str}")
        else:
            result_lines.append(f"{line_num:4d}| {line}")
    
    return '\n'.join(result_lines)


def generate_combined_annotated_code(focal_code: str,
                                     start_line: int,
                                     detailed_coverage: DetailedCoverageInfo = None,
                                     detailed_mutations: DetailedMutationInfo = None) -> str:
    """
    Generate focal method code with both coverage and mutation annotations.
    
    Each line is prefixed with line number (e.g., "131|").
    Coverage and mutation annotations are added as inline comments at the end of lines.
    
    Args:
        focal_code: The focal method source code
        start_line: The starting line number in the source file
        detailed_coverage: Detailed coverage information (optional)
        detailed_mutations: Detailed mutation information (optional)
        
    Returns:
        Annotated focal method code with line numbers and both coverage and mutation info
    """
    lines = focal_code.split('\n')
    
    # Group mutations by line number
    mutations_by_line: Dict[int, List[MutationDetail]] = {}
    if detailed_mutations and detailed_mutations.mutations:
        for mutation in detailed_mutations.mutations:
            line_num = mutation.line_number
            if line_num not in mutations_by_line:
                mutations_by_line[line_num] = []
            mutations_by_line[line_num].append(mutation)
    
    result_lines = []
    
    for i, line in enumerate(lines):
        line_num = start_line + i
        
        # Collect all annotations for this line
        all_annotations = []
        
        # Add coverage annotation
        if detailed_coverage:
            line_cov = detailed_coverage.get_line_coverage(line_num)
            if line_cov:
                # Coverage status
                if line_cov.is_covered:
                    all_annotations.append("✅ COVERED")
                else:
                    all_annotations.append("❌ NOT_COVERED")
                
                # Branch info
                if line_cov.has_branch:
                    total = line_cov.covered_branches + line_cov.missed_branches
                    if line_cov.missed_branches > 0:
                        all_annotations.append(f"⚠️ BRANCH: {line_cov.covered_branches}/{total} covered")
                    else:
                        all_annotations.append(f"✅ BRANCH: {line_cov.covered_branches}/{total} covered")
        
        # Add mutation annotations
        if line_num in mutations_by_line:
            mutations = mutations_by_line[line_num]
            for mutation in mutations:
                # Status icon
                if mutation.is_killed:
                    status_icon = "🟢 KILLED"
                elif mutation.is_survived:
                    status_icon = "🔴 SURVIVED"
                else:
                    status_icon = "⚪ NO_COVERAGE"
                
                mutator_name = mutation.mutator_simple_name
                all_annotations.append(f"{status_icon} [{mutator_name}] {mutation.description}")
        
        # Build the final line with line number and annotations
        if all_annotations:
            annotation_str = " | ".join(all_annotations)
            result_lines.append(f"{line_num:4d}| {line}  // {annotation_str}")
        else:
            # No annotations, check if it's a code line
            stripped = line.strip()
            if stripped and not stripped.startswith('//') and not stripped.startswith('/*') and not stripped.startswith('*'):
                result_lines.append(f"{line_num:4d}| {line}  // ⚪ NO_INSTRUCTION")
            else:
                result_lines.append(f"{line_num:4d}| {line}")
    
    return '\n'.join(result_lines)