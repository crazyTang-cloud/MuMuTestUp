from typing import Dict, Any, List, Optional
from models import AnalysisResult, TestResultInfo, UpdateInstruction, DiffHunk, RetrievalResult
from agents.base_agent import BaseAgent
from agents.retrieval_agent import RetrievalAgent


class ErrorAnalyzeAgent(BaseAgent):
    """Agent for analyzing compile errors and run failures"""
    
    def __init__(self, retrieval_agent: Optional[RetrievalAgent] = None, 
                 repo_path: str = None, project_name: str = None):
        super().__init__("ErrorAnalyzeAgent", "analyzer")
        self.retrieval_agent = retrieval_agent
        self.repo_path = repo_path  # Repository path for reading test files
        self.project_name = project_name  # Project name for logging
    
    def execute(self, test_result: TestResultInfo,
                filtered_hunks: Optional[List[DiffHunk]] = None) -> AnalysisResult:
        """
        Analyze error with new workflow: classify → retrieve → LLM analyze.
        
        Args:
            test_result: The test result with error
            filtered_hunks: Filtered relevant hunks from InputPreprocessAgent
            
        Returns:
            AnalysisResult with structured error analysis
        """
        self.log_info(f"Analyzing error: {test_result.status}")
        
        # Step 1: Smart classification with original code filtering
        error_classification = self._classify_error_type_with_original_filter(
            test_result.error_message or "",
            test_result.raw_error_output or "",
            test_result,
            test_result.test_case.original_code
        )
        
        self.log_info(f"Error classification: {error_classification['type']}")
        self.log_info(f"Known symbols: {len(error_classification['known_symbols'])}, "
                     f"Unknown symbols: {len(error_classification['unknown_symbols'])}")
        
        # Step 2: Retrieve for unknown symbols if any
        retrieval_result = None
        if error_classification['unknown_symbols'] and self.retrieval_agent:
            self.log_info(f"Retrieving for {len(error_classification['unknown_symbols'])} unknown symbol(s)...")
            retrieval_result = self.retrieval_agent.retrieve_for_unknown_symbols(
                symbols=error_classification['unknown_symbols'],
                error_message=test_result.error_message or "",
                test_code=test_result.test_case.code,
                focal_method_info=test_result.focal_method_info,
                filtered_hunks=filtered_hunks or [],
                error_type=error_classification['type']
            )
            
            if retrieval_result.retrieval_successful:
                self.log_info(f"Retrieval successful: retrieved {len(retrieval_result.retrieved_items)} symbol(s)")
                if retrieval_result.failed_symbols:
                    self.log_info(f"Failed to retrieve: {', '.join(retrieval_result.failed_symbols)}")
            else:
                self.log_info(f"Retrieval unsuccessful: {retrieval_result.retrieval_reasoning}")
        
        # Step 3: LLM analysis based on error type
        if error_classification['type'] == 'assertion_failure':
            llm_response = self._analyze_assertion_failure(
                test_result, filtered_hunks, error_classification, retrieval_result
            )
        elif error_classification['type'] in ['compilation_error', 'project_symbol_missing', 
                                               'import_missing', 'common_library_missing']:
            llm_response = self._analyze_compilation_error(
                test_result, filtered_hunks, error_classification, retrieval_result
            )
        else:
            llm_response = self._analyze_general_error(
                test_result, filtered_hunks, error_classification, retrieval_result
            )
        
        # Step 4: Generate annotated test code with error comments
        annotated_test_code = self._inject_error_comments(
            test_result.test_case.code,
            error_classification.get('error_locations', []),
            test_result.test_method_start_line
        )
        
        # Step 5: Construct structured AnalysisResult
        analysis = AnalysisResult(
            agent_name=self.name,
            analysis_type="error",
            error_type=error_classification['type'],
            known_symbols=error_classification.get('known_symbols', []),
            unknown_symbols=error_classification.get('unknown_symbols', []),
            error_locations=error_classification.get('error_locations', []),
            root_cause=llm_response.get('root_cause', ''),
            explanation=llm_response.get('explanation', ''),
            retrieval_result=retrieval_result,
            annotated_test_code=annotated_test_code
        )
        
        self.log_info(f"Analysis complete. Error type: {analysis.error_type}")
        
        return analysis
    
    def _build_analysis_prompt(self, test_result: TestResultInfo,
                               filtered_hunks: Optional[List[DiffHunk]] = None) -> str:
        """Build prompt for LLM to analyze error"""
        
        # Build error details section
        error_details = f"Error Message: {test_result.error_message}"
        
        # Add detailed error information if available
        if test_result.raw_error_output:
            # Extract key compilation errors for better context
            key_errors = self._extract_key_error_details(test_result.raw_error_output)
            if key_errors:
                error_details += f"\n\nDetailed Error Information:\n{key_errors}"
        
        # Format filtered hunks
        hunks_text = ""
        if filtered_hunks:
            hunks_text = "\n\nRelevant Code Changes:\n"
            for i, hunk in enumerate(filtered_hunks, 1):
                hunk_type = hunk.hunk_type or "unknown"
                hunks_text += f"\n[{hunk_type}] Hunk {i} - {hunk.file_path}:\n{hunk.context}\n"
        
        # Format test class context (fields and non-test methods)
        class_context_text = self._format_test_class_context(test_result.test_case)
        
        return f"""
Analyze the following test failure:

Test Case: {test_result.test_case.name}
Focal Method: {test_result.focal_method_info.name}
Status: {test_result.status}
{error_details}

Current Test Code:
```
{test_result.test_case.code}
```

{class_context_text}

Focal Method Code:
```
{test_result.focal_method_info.current_code}
```
{hunks_text}

Based on the error message, code context, and relevant changes, please:
1. Identify the root cause of the failure
2. Suggest what methods or fields might be missing
3. Provide a candidate set of methods/fields that could help
4. Explain how these would help fix the test
5. Determine if additional information retrieval from the codebase is needed

Format your response as JSON with keys:
- root_cause: string
- missing_candidates: list of strings (method/field names)
- explanation: string
- needs_retrieval: boolean (true if we need to search the codebase for more information)
- retrieval_targets: list of strings (what to search for, if needs_retrieval is true)
"""
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for error analysis"""
        return """You are an expert at analyzing test failures and determining what information 
is missing from test cases. You understand compilation errors, test execution failures, 
and how to identify missing methods or fields needed to fix tests.

When analyzing errors:
1. Look for missing imports, undefined classes, or missing methods
2. Check if the error is due to API changes in the code
3. Identify if new dependencies or mocking is needed
4. Determine if additional information from the codebase would help

Always return valid JSON with the specified format."""
    
    def _parse_llm_response(self, response: str, test_result: TestResultInfo) -> Dict[str, Any]:
        """Parse LLM response into recommendations"""
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "root_cause": data.get("root_cause", ""),
                    "missing_candidates": data.get("missing_candidates", []),
                    "explanation": data.get("explanation", ""),
                    "error_status": test_result.status,
                    "needs_retrieval": data.get("needs_retrieval", False),
                    "retrieval_targets": data.get("retrieval_targets", [])
                }
        except Exception as e:
            self.log_error(f"Failed to parse LLM response: {e}")
        
        return {
            "root_cause": "Unknown error",
            "missing_candidates": [],
            "explanation": response,
            "error_status": test_result.status,
            "needs_retrieval": True,
            "retrieval_targets": []
        }
    
    def _determine_retrieval_need(self, response: str, test_result: TestResultInfo) -> bool:
        """
        Determine if retrieval is needed based on LLM response and error classification.
        
        This method intelligently distinguishes between:
        1. Import-related errors (no retrieval needed)
        2. Common library errors (no retrieval needed)
        3. Project-specific missing symbols (retrieval needed)
        """
        try:
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                llm_needs_retrieval = data.get("needs_retrieval", False)
                
                # If LLM says no retrieval needed, trust it
                if not llm_needs_retrieval:
                    return False
                
                # If LLM says retrieval needed, do smart classification
                # to avoid unnecessary retrieval for import-only issues
        except Exception as e:
            self.log_error(f"Failed to parse retrieval need: {e}")
        
        # Smart error classification to avoid unnecessary retrieval
        if test_result.error_message or test_result.raw_error_output:
            error_classification = self._classify_error_type(
                test_result.error_message or "",
                test_result.raw_error_output or "",
                test_result
            )
            
            self.log_info(f"Error classification: {error_classification['type']}")
            self.log_info(f"Classification reasoning: {error_classification['reasoning']}")
            
            # Check if there are unknown symbols that need retrieval
            unknown_symbols = error_classification.get('unknown_symbols', [])
            known_symbols = error_classification.get('known_symbols', [])
            
            if unknown_symbols:
                # Has unknown symbols - need retrieval for those
                self.log_info(f"Found {len(unknown_symbols)} unknown symbol(s) needing retrieval: {', '.join(unknown_symbols)}")
                if known_symbols:
                    self.log_info(f"Also found {len(known_symbols)} known symbol(s) that only need imports: {', '.join(known_symbols)}")
                return True
            elif known_symbols:
                # Only has known symbols - can be fixed with imports alone
                self.log_info(f"All {len(known_symbols)} symbol(s) are known, can be fixed with imports only: {', '.join(known_symbols)}")
                return False
            elif error_classification['type'] in ['import_missing', 'common_library_missing', 'assertion_failure']:
                # These can be fixed without retrieval
                self.log_info(f"Error type '{error_classification['type']}' can be fixed without retrieval, skipping")
                return False
        
        # Fallback: check if error suggests missing symbols
        if test_result.error_message:
            error_lower = test_result.error_message.lower()
            if any(keyword in error_lower for keyword in [
                "cannot find symbol", "symbol not found", "class not found",
                "method not found", "undefined", "unresolved"
            ]):
                # Default to True for safety, but classification above should handle most cases
                return True
        
        return False
    
    def _extract_strategies(self, response: str, test_result: TestResultInfo = None) -> List[str]:
        """
        Extract strategies from LLM response and error context.
        
        Uses intelligent error classification to prioritize import-only fixes
        over more complex retrieval-based strategies.
        """
        strategies = []
        
        response_lower = response.lower()
        
        # Also check the original error message if available
        error_context = ""
        raw_error = ""
        if test_result:
            if test_result.error_message:
                error_context = test_result.error_message.lower()
            if test_result.raw_error_output:
                raw_error = test_result.raw_error_output
        
        combined_text = response_lower + " " + error_context
        
        # Use smart error classification to determine if it's just an import issue
        if test_result and (test_result.error_message or test_result.raw_error_output):
            error_classification = self._classify_error_type(
                test_result.error_message or "",
                raw_error,
                test_result
            )
            
            # If it's assertion failure, return adjust_assertions strategy
            if error_classification['type'] == 'assertion_failure':
                strategies.append("adjust_assertions")
                self.log_info(f"Detected assertion failure: {error_classification['reasoning']}")
                return strategies  # Return early with clear strategy
            
            # If it's clearly an import/common library issue, prioritize add_imports
            if error_classification['type'] in ['import_missing', 'common_library_missing']:
                strategies.append("add_imports")
                self.log_info(f"Detected import-only issue: {error_classification['reasoning']}")
                return strategies  # Return early, no need for other strategies
        
        # Check for import-related strategies (check first as it's most common for compile errors)
        if any(keyword in combined_text for keyword in ["import", "cannot find symbol", "symbol not found", "package", "class not found", "cannot find class"]):
            strategies.append("add_imports")
        
        # Check for method-related strategies
        if any(keyword in combined_text for keyword in ["add method", "missing method", "method not found", "cannot find method"]):
            strategies.append("add_methods")
        
        # Check for field-related strategies
        if any(keyword in combined_text for keyword in ["add field", "missing field", "field not found", "cannot find field"]):
            strategies.append("add_fields")
        
        # Check for mocking strategies
        if any(keyword in combined_text for keyword in ["mock", "stub", "dependency"]):
            strategies.append("add_mocking")
        
        return strategies if strategies else ["manual_inspection"]
    
    def _classify_error_type(self, error_message: str, raw_error_output: str, 
                            test_result: 'TestResultInfo' = None) -> Dict[str, Any]:
        """
        Classify the type of compilation/runtime error to determine if retrieval is needed.
        
        Args:
            error_message: The error message
            raw_error_output: Raw Maven output
            test_result: TestResultInfo for accessing test file and context
        
        Returns:
            Dict with keys:
                - type: 'assertion_failure', 'import_missing', 'common_library_missing', 
                        'project_symbol_missing', 'other'
                - confidence: float (0-1)
                - missing_symbols: list of missing symbol names
                - reasoning: str
                - error_locations: list of dicts with failing line info (for assertion failures)
        """
        import re
        from pathlib import Path
        
        combined_error = error_message + "\n" + raw_error_output
        
        # PRIORITY 1: Check for compilation errors with line numbers
        compilation_error_locations = self._extract_compilation_error_locations(
            error_message, raw_error_output, test_result
        )
        if compilation_error_locations:
            self.log_info(f"Detected compilation error with {len(compilation_error_locations)} location(s)")
            for loc in compilation_error_locations:
                # Include column number in log if available
                if loc.get('column') is not None:
                    self.log_info(f"  Line {loc['file_line']}, Column {loc['column']}: {loc['error_message']}")
                else:
                    self.log_info(f"  Line {loc['file_line']}: {loc['error_message']}")
                if loc.get('code'):
                    self.log_info(f"    Code: {loc['code'][:80]}")
        
        # PRIORITY 2: Check for assertion failures FIRST (before symbol analysis)
        assertion_patterns = [
            'expected:', 'but was:', 'AssertionError', 'assertion failed',
            'expected but was', 'expected:<', 'but was:<', 'AssertionFailedError',
            'ComparisonFailure', 'org.junit.ComparisonFailure',
            'org.opentest4j.AssertionFailedError',
            # Mockito verification failures (also treated as assertion failures)
            'ArgumentsAreDifferent', 'WantedButNotInvoked', 'NeverWantedButInvoked',
            'TooManyActualInvocations', 'TooFewActualInvocations', 'NoInteractionsWanted',
            'org.mockito.exceptions.verification'
        ]
        
        if any(pattern in combined_error for pattern in assertion_patterns):
            self.log_info("Detected assertion failure, extracting failure locations...")
            
            # Extract failure locations from surefire reports or stack traces
            error_locations = self._extract_assertion_failure_locations(
                error_message, raw_error_output, test_result
            )
            
            if error_locations:
                self.log_info(f"Found {len(error_locations)} assertion failure location(s)")
                for loc in error_locations:
                    self.log_info(f"  Line {loc['file_line']}: {loc['code'][:80]}")
            
            return {
                'type': 'assertion_failure',
                'confidence': 1.0,
                'missing_symbols': [],
                'known_symbols': [],
                'unknown_symbols': [],
                'error_locations': error_locations,
                'reasoning': f'Test assertion failure - {len(error_locations)} assertion(s) failed. ' +
                           'Test expectations need updating to match new behavior.'
            }
        
        # PRIORITY 3: Check for compilation errors (missing symbols)
        missing_symbols = []
        
        # Extract missing symbol names from error messages
        # Pattern 1: "cannot find symbol: class/method/variable X"
        symbol_patterns = [
            r'cannot find symbol:\s*(?:class|method|variable)\s+(\w+)',
            r'symbol:\s*(?:class|method|variable)\s+(\w+)',
            r'package\s+(\S+)\s+does not exist',
            r'(\w+)\s+cannot be resolved',
        ]
        
        for pattern in symbol_patterns:
            matches = re.findall(pattern, combined_error, re.IGNORECASE)
            missing_symbols.extend(matches)
        
        # Remove duplicates while preserving order
        missing_symbols = list(dict.fromkeys(missing_symbols))
        
        # NEW: If no symbols extracted from error message, try extracting from error code lines
        if not missing_symbols and compilation_error_locations:
            self.log_info("No symbols found in error message, extracting from error code lines...")
            missing_symbols = self._extract_symbols_from_error_locations(
                compilation_error_locations, 
                test_result
            )
            if missing_symbols:
                self.log_info(f"Extracted {len(missing_symbols)} symbol(s) from error code: {', '.join(missing_symbols)}")
        
        if not missing_symbols:
            # Check if we have compilation error locations but no symbols extracted
            if compilation_error_locations:
                return {
                    'type': 'compilation_error',
                    'confidence': 1.0,
                    'missing_symbols': [],
                    'known_symbols': [],
                    'unknown_symbols': [],
                    'error_locations': compilation_error_locations,
                    'reasoning': f'Compilation error at {len(compilation_error_locations)} location(s), but no symbols could be extracted'
                }
            
            return {
                'type': 'other',
                'confidence': 0.5,
                'missing_symbols': [],
                'known_symbols': [],
                'unknown_symbols': [],
                'error_locations': [],
                'reasoning': 'No clear missing symbols detected'
            }
        
        # Check if symbols are from common test libraries (JUnit, AssertJ, Mockito, etc.)
        common_test_symbols = {
            # JUnit 4
            'Test', 'Before', 'After', 'BeforeClass', 'AfterClass', 'Ignore', 'Rule',
            'Assert', 'assertEquals', 'assertTrue', 'assertFalse', 'assertNull',
            'assertNotNull', 'assertSame', 'assertNotSame', 'assertArrayEquals',
            'assertThat', 'fail', 'assertThrows',
            # JUnit 5 (Jupiter)
            'BeforeEach', 'AfterEach', 'BeforeAll', 'AfterAll', 'DisplayName',
            'Nested', 'Tag', 'Tags', 'Disabled', 'RepeatedTest', 'ParameterizedTest',
            'ValueSource', 'CsvSource', 'MethodSource', 'ArgumentsSource', 'ExtendWith',
            'Assertions', 'assertAll', 'assertDoesNotThrow', 'assertTimeout', 'assertTimeoutPreemptively',
            'Assumptions', 'assumeTrue', 'assumeFalse', 'assumingThat',
            # TestNG
            'org.testng', 'testng', 'DataProvider', 'Factory', 'Listeners', 'Parameters',
            'ITestContext', 'ITestResult', 'TestNGException', 'SkipException',
            'SuiteRunner', 'TestRunner', 'XmlSuite', 'XmlTest',
            # Mockito
            'Mock', 'Spy', 'InjectMocks', 'Captor', 'MockitoAnnotations',
            'Mockito', 'when', 'verify', 'times', 'never', 'any', 'anyString',
            'anyInt', 'anyLong', 'anyDouble', 'anyFloat', 'anyBoolean', 'anyByte',
            'anyChar', 'anyShort', 'anyList', 'anySet', 'anyMap', 'anyCollection',
            'anyIterable', 'anyObject', 'eq', 'isA', 'isNull', 'isNotNull',
            'doReturn', 'doThrow', 'doNothing', 'doAnswer', 'doCallRealMethod',
            'thenReturn', 'thenThrow', 'thenAnswer', 'thenCallRealMethod',
            'mock', 'spy', 'reset', 'verifyNoMoreInteractions', 'verifyZeroInteractions',
            'verifyNoInteractions', 'inOrder', 'timeout', 'atLeast', 'atMost', 'calls',
            'ArgumentCaptor', 'ArgumentMatchers', 'Answer', 'InvocationOnMock',
            # AssertJ
            'assertThat', 'assertThatThrownBy', 'assertThatExceptionOfType',
            'assertThatCode', 'assertThatNoException', 'assertThatNullPointerException',
            'assertThatIllegalArgumentException', 'assertThatIllegalStateException',
            'assertThatIOException', 'Condition', 'SoftAssertions',
            # Hamcrest
            'Matchers', 'CoreMatchers', 'is', 'equalTo', 'hasItem', 'hasItems',
            'contains', 'containsInAnyOrder', 'hasSize', 'hasEntry', 'hasKey',
            'hasValue', 'nullValue', 'notNullValue', 'instanceOf', 'any',
            'anything', 'not', 'either', 'both', 'allOf', 'anyOf',
            # PowerMock
            'PowerMockito', 'PrepareForTest', 'RunWith', 'PowerMockRunner',
            'mockStatic', 'whenNew', 'verifyStatic', 'verifyNew',
            # EasyMock
            'EasyMock', 'expect', 'replay', 'createMock', 'createNiceMock',
            'createStrictMock', 'expectLastCall',
        }
        
        # Check if symbols are from Java standard library
        java_std_symbols = {
            # Collections
            'List', 'ArrayList', 'LinkedList', 'Vector', 'Stack',
            'Set', 'HashSet', 'TreeSet', 'LinkedHashSet', 'EnumSet',
            'Map', 'HashMap', 'TreeMap', 'LinkedHashMap', 'Hashtable', 'WeakHashMap',
            'IdentityHashMap', 'EnumMap', 'ConcurrentHashMap', 'ConcurrentSkipListMap',
            'Collection', 'Collections', 'Queue', 'Deque', 'PriorityQueue',
            'ArrayDeque', 'BlockingQueue', 'ConcurrentLinkedQueue',
            # Streams and Functional
            'Optional', 'OptionalInt', 'OptionalLong', 'OptionalDouble',
            'Stream', 'IntStream', 'LongStream', 'DoubleStream',
            'Collectors', 'Collector', 'Function', 'Predicate', 'Consumer',
            'Supplier', 'BiFunction', 'BiConsumer', 'BiPredicate', 'UnaryOperator',
            'BinaryOperator', 'ToIntFunction', 'ToLongFunction', 'ToDoubleFunction',
            # Arrays and Utilities
            'Arrays', 'Objects', 'Comparator', 'Iterator', 'ListIterator',
            'Enumeration', 'StringTokenizer', 'Scanner', 'Properties',
            # Primitives and Wrappers
            'String', 'StringBuilder', 'StringBuffer', 'CharSequence',
            'Integer', 'Long', 'Double', 'Float', 'Boolean', 'Byte', 'Short',
            'Character', 'Number', 'Void',
            # Math
            'Math', 'BigDecimal', 'BigInteger', 'Random', 'ThreadLocalRandom',
            'AtomicInteger', 'AtomicLong', 'AtomicBoolean', 'AtomicReference',
            # Date and Time
            'Date', 'Calendar', 'GregorianCalendar', 'TimeZone', 'SimpleDateFormat',
            'DateFormat', 'LocalDate', 'LocalTime', 'LocalDateTime', 'ZonedDateTime',
            'OffsetDateTime', 'Instant', 'Duration', 'Period', 'ZoneId', 'ZoneOffset',
            'DateTimeFormatter', 'Temporal', 'TemporalAdjuster', 'ChronoUnit',
            # Exceptions
            'Exception', 'RuntimeException', 'Error', 'Throwable',
            'IllegalArgumentException', 'IllegalStateException',
            'NullPointerException', 'IndexOutOfBoundsException',
            'ArrayIndexOutOfBoundsException', 'ClassCastException',
            'NumberFormatException', 'ArithmeticException',
            'UnsupportedOperationException', 'NoSuchElementException',
            'IOException', 'FileNotFoundException', 'EOFException',
            'InterruptedException', 'ExecutionException', 'TimeoutException',
            'ClassNotFoundException', 'InstantiationException',
            'NoSuchMethodException', 'NoSuchFieldException',
            # I/O
            'File', 'Path', 'Paths', 'Files', 'FileInputStream', 'FileOutputStream',
            'FileReader', 'FileWriter', 'BufferedReader', 'BufferedWriter',
            'InputStreamReader', 'OutputStreamWriter', 'PrintWriter', 'PrintStream',
            'InputStream', 'OutputStream', 'Reader', 'Writer', 'ByteArrayInputStream',
            'ByteArrayOutputStream', 'StringReader', 'StringWriter',
            # Regex
            'Pattern', 'Matcher', 'PatternSyntaxException',
            # Reflection
            'Class', 'Field', 'Method', 'Constructor', 'Modifier', 'Annotation',
            'Type', 'ParameterizedType', 'TypeVariable', 'WildcardType',
            # Concurrency
            'Thread', 'Runnable', 'Callable', 'Future', 'CompletableFuture',
            'ExecutorService', 'Executors', 'ThreadPoolExecutor', 'ScheduledExecutorService',
            'Lock', 'ReentrantLock', 'ReadWriteLock', 'ReentrantReadWriteLock',
            'Semaphore', 'CountDownLatch', 'CyclicBarrier', 'Phaser',
            # Others
            'System', 'Runtime', 'Process', 'ProcessBuilder',
            'URL', 'URI', 'URLEncoder', 'URLDecoder', 'Charset', 'StandardCharsets',
            'UUID', 'Locale', 'ResourceBundle', 'Logger', 'Level',
        }
        
        # Classify each missing symbol
        common_lib_count = 0
        std_lib_count = 0
        unknown_count = 0
        
        for symbol in missing_symbols:
            if symbol in common_test_symbols:
                common_lib_count += 1
            elif symbol in java_std_symbols:
                std_lib_count += 1
            else:
                unknown_count += 1
        
        total = len(missing_symbols)
        
        # Decision logic
        if total == 0:
            return {
                'type': 'other',
                'confidence': 0.5,
                'missing_symbols': [],
                'known_symbols': [],
                'unknown_symbols': [],
                'error_locations': compilation_error_locations if compilation_error_locations else [],
                'reasoning': 'No missing symbols identified'
            }
        
        # Separate known and unknown symbols
        known_symbols = []
        unknown_symbols = []
        
        for symbol in missing_symbols:
            if symbol in common_test_symbols or symbol in java_std_symbols:
                known_symbols.append(symbol)
            else:
                unknown_symbols.append(symbol)
        
        # Key insight: If there are ANY unknown symbols, we need retrieval for those
        # Known symbols can be handled by imports alone
        
        if unknown_count > 0:
            # Has unknown symbols - need retrieval for the unknown ones
            return {
                'type': 'project_symbol_missing',
                'confidence': unknown_count / total,
                'missing_symbols': missing_symbols,
                'known_symbols': known_symbols,
                'unknown_symbols': unknown_symbols,
                'error_locations': compilation_error_locations,  # Include compilation locations
                'reasoning': f'Found {unknown_count} unknown symbol(s) that need retrieval: {", ".join(unknown_symbols)}. ' +
                            (f'Also found {len(known_symbols)} known symbol(s) that only need imports: {", ".join(known_symbols)}' if known_symbols else '')
            }
        
        # All symbols are known (from common libraries or standard library)
        # No retrieval needed, just imports
        if common_lib_count > std_lib_count:
            return {
                'type': 'common_library_missing',
                'confidence': 1.0,
                'missing_symbols': missing_symbols,
                'known_symbols': known_symbols,
                'unknown_symbols': [],
                'error_locations': compilation_error_locations,  # Include compilation locations
                'reasoning': f'All symbols are from common test libraries: {", ".join(missing_symbols)}'
            }
        else:
            return {
                'type': 'import_missing',
                'confidence': 1.0,
                'missing_symbols': missing_symbols,
                'known_symbols': known_symbols,
                'unknown_symbols': [],
                'error_locations': compilation_error_locations,  # Include compilation locations
                'reasoning': f'All symbols are from Java standard library: {", ".join(missing_symbols)}'
            }
    
    def _classify_error_type_with_original_filter(self, error_message: str, raw_error_output: str,
                                                   test_result: 'TestResultInfo', 
                                                   original_test_code: str) -> Dict[str, Any]:
        """
        Classify error type with original code filtering.
        
        Filters out symbols that appear in the original test code (bCommit version),
        as they are already known and don't need retrieval.
        
        Args:
            error_message: Error message
            raw_error_output: Raw Maven output
            test_result: TestResultInfo
            original_test_code: Original test code from bCommit (before changes)
            
        Returns:
            Classification result with filtered symbols
        """
        # First, do standard classification
        classification = self._classify_error_type(error_message, raw_error_output, test_result)
        
        # Filter missing_symbols: remove those appearing in original test code
        missing_symbols = classification.get('missing_symbols', [])
        filtered_symbols = []
        
        for symbol in missing_symbols:
            if original_test_code and symbol in original_test_code:
                # Symbol appears in original test → already known, skip
                self.log_info(f"Skipping symbol '{symbol}' (found in original test code)")
            else:
                # Symbol doesn't appear in original → newly introduced
                filtered_symbols.append(symbol)
        
        # Re-classify filtered symbols into known/unknown
        known_symbols = []
        unknown_symbols = []
        
        common_test_symbols = self._get_common_test_symbols()
        java_std_symbols = self._get_java_std_symbols()
        
        for symbol in filtered_symbols:
            if symbol in common_test_symbols or symbol in java_std_symbols:
                known_symbols.append(symbol)
            else:
                unknown_symbols.append(symbol)
        
        # Update classification
        classification['missing_symbols'] = filtered_symbols
        classification['known_symbols'] = known_symbols
        classification['unknown_symbols'] = unknown_symbols
        
        # Ensure error_locations key exists (in case it was missing from _classify_error_type)
        if 'error_locations' not in classification:
            classification['error_locations'] = []
        
        self.log_info(f"Symbol filtering: {len(missing_symbols)} → {len(filtered_symbols)} "
                     f"(filtered {len(missing_symbols) - len(filtered_symbols)} from original code)")
        
        return classification
    
    def _get_common_test_symbols(self) -> set:
        """Get set of common test framework symbols"""
        return {
            # JUnit 4
            'Test', 'Before', 'After', 'BeforeClass', 'AfterClass', 'Ignore', 'Rule',
            'Assert', 'assertEquals', 'assertTrue', 'assertFalse', 'assertNull',
            'assertNotNull', 'assertSame', 'assertNotSame', 'assertArrayEquals',
            'assertThat', 'fail', 'assertThrows',
            # JUnit 5
            'BeforeEach', 'AfterEach', 'BeforeAll', 'AfterAll', 'DisplayName',
            'Nested', 'Tag', 'Tags', 'Disabled', 'RepeatedTest', 'ParameterizedTest',
            'Assertions', 'assertAll', 'assertDoesNotThrow', 'assertTimeout',
            # Mockito
            'Mock', 'Spy', 'InjectMocks', 'Mockito', 'when', 'verify', 'times',
            'any', 'anyString', 'anyInt', 'eq', 'doReturn', 'doThrow',
            # AssertJ
            'assertThat', 'assertThatThrownBy',
        }
    
    def _get_java_std_symbols(self) -> set:
        """Get set of Java standard library symbols"""
        return {
            # Collections
            'List', 'ArrayList', 'LinkedList', 'Set', 'HashSet', 'TreeSet',
            'Map', 'HashMap', 'TreeMap', 'LinkedHashMap', 'Collection', 'Collections',
            # Streams
            'Optional', 'Stream', 'Collectors',
            # Others
            'String', 'StringBuilder', 'Integer', 'Long', 'Double', 'Boolean',
            'Arrays', 'Objects', 'Math', 'System',
        }
    
    def _extract_identifier_at_position(self, code: str, col_idx: int) -> Optional[str]:
        """
        Extract the identifier (symbol name) at a specific column position in code.
        
        Handles cases where the column points to whitespace or punctuation by finding
        the nearest identifier.
        
        Args:
            code: The code line
            col_idx: 0-indexed column position
            
        Returns:
            The identifier at that position, or None if not found
        """
        import re
        
        if col_idx < 0 or col_idx >= len(code):
            return None
        
        # If not at an identifier character, scan forward to find the next identifier
        search_idx = col_idx
        while search_idx < len(code) and not (code[search_idx].isalpha() or code[search_idx] == '_'):
            search_idx += 1
        
        # If we reached the end without finding an identifier start, return None
        if search_idx >= len(code):
            return None
        
        # Now we're at the start of an identifier
        # Find the start of the identifier (scan backwards from current position)
        start = search_idx
        while start > 0 and (code[start - 1].isalnum() or code[start - 1] == '_'):
            start -= 1
        
        # Find the end of the identifier (scan forwards)
        end = search_idx
        while end < len(code) and (code[end].isalnum() or code[end] == '_'):
            end += 1
        
        identifier = code[start:end]
        return identifier if identifier else None
    
    def _extract_symbols_from_error_locations(self, 
                                             compilation_error_locations: List[Dict[str, Any]],
                                             test_result: 'TestResultInfo' = None) -> List[str]:
        """
        Extract undefined symbols from compilation error code lines.
        
        This method analyzes the actual code that failed to compile and identifies
        symbols that are likely undefined. It uses multiple strategies:
        1. (PRIORITY) If column number is available, extract symbol at that exact position
        2. Identifies class names (capitalized identifiers)
        3. Identifies static field references (Class.CONSTANT)
        4. Filters out known Java keywords and common symbols
        5. Checks if symbols are already imported in the test file
        
        Args:
            compilation_error_locations: List of compilation error locations with code
            test_result: TestResultInfo for accessing test context
            
        Returns:
            List of symbol names that are likely undefined
        """
        import re
        
        if not compilation_error_locations:
            return []
        
        # Collect test imports to check if symbol is already imported
        test_imports = set()
        if test_result and test_result.test_case.test_imports:
            for imp in test_result.test_case.test_imports:
                # Extract class name from import statement
                # e.g., "import org.example.MyClass;" -> "MyClass"
                match = re.search(r'import\s+(?:static\s+)?[\w.]+\.(\w+)\s*;', imp)
                if match:
                    test_imports.add(match.group(1))
        
        # Java keywords and primitives to exclude
        java_keywords = {
            'abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 
            'char', 'class', 'const', 'continue', 'default', 'do', 'double',
            'else', 'enum', 'extends', 'final', 'finally', 'float', 'for',
            'goto', 'if', 'implements', 'import', 'instanceof', 'int', 'interface',
            'long', 'native', 'new', 'package', 'private', 'protected', 'public',
            'return', 'short', 'static', 'strictfp', 'super', 'switch', 'synchronized',
            'this', 'throw', 'throws', 'transient', 'try', 'void', 'volatile', 'while',
            'true', 'false', 'null',
        }
        
        # Common method names and variables to exclude
        common_identifiers = {
            'get', 'set', 'is', 'has', 'add', 'remove', 'size', 'length',
            'equals', 'hashCode', 'toString', 'compareTo', 'clone',
            'handler', 'result', 'value', 'data', 'item', 'element',
            'log', 'logger', 'message', 'error', 'exception',
        }
        
        extracted_symbols = []
        
        for location in compilation_error_locations:
            code = location.get('code', '').strip()
            if not code:
                continue
            
            # PRIORITY: If column number is available, use it to pinpoint the exact symbol
            column = location.get('column')
            if column is not None:
                # Column is 1-indexed, convert to 0-indexed for Python string
                col_idx = column - 1
                if 0 <= col_idx < len(code):
                    # Extract the identifier at this position
                    symbol = self._extract_identifier_at_position(code, col_idx)
                    if symbol:
                        # Filter out keywords and common identifiers
                        if (symbol not in java_keywords and 
                            symbol not in common_identifiers and
                            symbol not in test_imports):
                            extracted_symbols.append(symbol)
                            self.log_info(f"  Found symbol at column {column}: {symbol}")
                            # Skip other strategies for this location since we have precise info
                            continue
                        else:
                            self.log_info(f"  Symbol at column {column} is '{symbol}' (known/imported, skipping)")
            
            # Strategy 1: Find Class.CONSTANT or Class.method patterns
            # Pattern: CapitalizedWord.CAPITALIZED_WORD or CapitalizedWord.word
            class_member_pattern = r'\b([A-Z][a-zA-Z0-9_]*)\s*\.\s*([A-Z_][A-Z0-9_]*|[a-z][a-zA-Z0-9_]*)\b'
            for match in re.finditer(class_member_pattern, code):
                class_name = match.group(1)
                member_name = match.group(2)
                
                # Check if class is not imported and not a known symbol
                if (class_name not in test_imports and 
                    class_name not in java_keywords and
                    class_name not in common_identifiers):
                    extracted_symbols.append(class_name)
                    self.log_info(f"  Found potential undefined class: {class_name} (in {class_name}.{member_name})")
            
            # Strategy 2: Find standalone capitalized identifiers (potential class names)
            # But be more conservative - only if not followed by a dot (not a chained call)
            standalone_class_pattern = r'\b([A-Z][a-zA-Z0-9_]*)\b(?!\s*\.)'
            for match in re.finditer(standalone_class_pattern, code):
                class_name = match.group(1)
                
                # Filter more strictly for standalone classes
                if (class_name not in test_imports and
                    class_name not in java_keywords and
                    class_name not in common_identifiers and
                    len(class_name) > 2):  # Avoid single/double letter abbreviations
                    extracted_symbols.append(class_name)
                    self.log_info(f"  Found potential undefined class: {class_name}")
            
            # Strategy 3: Check method calls on undefined objects
            # Pattern: undefinedObject.method() where undefinedObject might be the issue
            method_call_pattern = r'\b([a-z][a-zA-Z0-9_]*)\s*\.\s*([a-z][a-zA-Z0-9_]*)\s*\('
            for match in re.finditer(method_call_pattern, code):
                obj_name = match.group(1)
                method_name = match.group(2)
                
                # If it's a common variable pattern, skip
                if obj_name not in common_identifiers and len(obj_name) > 3:
                    # This is less reliable, so mark it for consideration but don't add directly
                    self.log_info(f"  Found potential object reference issue: {obj_name}.{method_name}()")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_symbols = []
        for symbol in extracted_symbols:
            if symbol not in seen:
                seen.add(symbol)
                unique_symbols.append(symbol)
        
        return unique_symbols
    
    def _extract_compilation_error_locations(self, error_message: str,
                                            raw_error_output: str,
                                            test_result: 'TestResultInfo' = None) -> List[Dict[str, Any]]:
        """
        Extract compilation error locations and find the actual failing code.
        
        Args:
            error_message: The error message
            raw_error_output: Raw Maven output
            test_result: TestResultInfo for accessing test context
            
        Returns:
            List of dicts with keys:
                - file_line: Line number in the test file
                - column: Column number (if available, None otherwise)
                - code: The actual code that failed to compile
                - error_message: The specific compilation error
        """
        import re
        from pathlib import Path
        
        locations = []
        
        if not test_result or not raw_error_output:
            return locations
        
        # Pattern for Maven compilation errors with column number:
        # [ERROR] /path/to/File.java:[line,column] error message
        error_pattern_with_column = r'\[ERROR\]\s+(.+\.java):\[(\d+),(\d+)\]\s+(.+)'
        matches = re.findall(error_pattern_with_column, raw_error_output)
        
        # Fallback: try pattern without column if no matches
        if not matches:
            error_pattern_no_column = r'\[ERROR\]\s+(.+\.java):\[(\d+)\]\s+(.+)'
            matches_no_col = re.findall(error_pattern_no_column, raw_error_output)
            # Convert to format: (file, line, None, error_msg)
            matches = [(f, l, None, e) for f, l, e in matches_no_col]
        
        if not matches:
            return locations
        
        # Get test file path
        test_file_path = None
        if test_result.focal_method_info.source_file_path:
            # Derive test file path from source file path
            source_path = test_result.focal_method_info.source_file_path
            test_file_path = source_path.replace('/src/main/java/', '/src/test/java/')
            # Replace source class name with test class name
            test_class = test_result.test_case.name.rsplit('.', 1)[0]
            source_class_file = test_result.focal_method_info.class_name.replace('.', '/') + '.java'
            test_class_file = test_class.replace('.', '/') + '.java'
            test_file_path = test_file_path.replace(source_class_file, test_class_file)
        
        # Process each compilation error
        for file_path, line_num_str, col_num_str, error_msg in matches[:10]:  # Limit to 10 errors
            line_num = int(line_num_str)
            col_num = int(col_num_str) if col_num_str else None
            
            # Read the code line from test_result.test_case.code (not from file!)
            # The file has been restored by JavaTestExecutor.finally, so we must use test code from memory
            code_line = self._extract_line_from_test_code(
                test_result.test_case.code, 
                line_num, 
                test_result.test_method_start_line
            )
            
            if code_line:
                location = {
                    'file_line': line_num,
                    'code': code_line,
                    'error_message': error_msg.strip()
                }
                if col_num is not None:
                    location['column'] = col_num
                locations.append(location)
                
                # Log with column info if available
                if col_num is not None:
                    self.log_info(f"Extracted compilation error: line {line_num}, column {col_num}, code: {code_line[:50]}...")
                else:
                    self.log_info(f"Extracted compilation error: line {line_num}, code: {code_line[:50]}...")
            else:
                # Even if we can't read the code, still record the location
                location = {
                    'file_line': line_num,
                    'code': '',
                    'error_message': error_msg.strip()
                }
                if col_num is not None:
                    location['column'] = col_num
                locations.append(location)
        
        return locations
    
    def _extract_assertion_failure_locations(self, error_message: str, 
                                            raw_error_output: str,
                                            test_result: 'TestResultInfo' = None) -> List[Dict[str, Any]]:
        """
        Extract assertion failure locations from surefire reports and find the actual failing code.
        
        Args:
            error_message: The error message
            raw_error_output: Raw Maven output
            test_result: TestResultInfo for accessing test context
            
        Returns:
            List of dicts with keys:
                - file_line: Line number in the test file
                - code: The actual code that failed
                - error_message: The specific error for this assertion
        """
        import re
        from pathlib import Path
        
        locations = []
        
        if not test_result:
            return locations
        
        test_class = test_result.test_case.name.rsplit('.', 1)[0]  # Get full class name
        test_method = test_result.test_case.name.rsplit('.', 1)[1].replace('()', '')
        
        # Try to find surefire report
        surefire_report_content = None
        
        if self.repo_path:
            repo_path = Path(self.repo_path)
            
            # Try multiple locations for surefire reports
            surefire_paths = []
            
            # 1. Extract from Maven output
            for line in raw_error_output.split('\n'):
                if 'Please refer to' in line and 'surefire-reports' in line:
                    match = re.search(r'(/[^\s]+/surefire-reports)', line)
                    if match:
                        surefire_dir = Path(match.group(1))
                        surefire_paths.append(surefire_dir / f"{test_class}.txt")
            
            # 2. Try standard locations
            # Multi-module: look for module directories
            for module_dir in repo_path.glob("**/target/surefire-reports"):
                surefire_paths.append(module_dir / f"{test_class}.txt")
            
            # Read surefire report
            for surefire_path in surefire_paths:
                if surefire_path.exists():
                    try:
                        with open(surefire_path, 'r', encoding='utf-8') as f:
                            surefire_report_content = f.read()
                        self.log_info(f"Read surefire report from: {surefire_path}")
                        break
                    except Exception as e:
                        self.log_warning(f"Failed to read surefire report {surefire_path}: {e}")
        
        # Parse surefire report or raw output for stack traces
        content_to_parse = surefire_report_content if surefire_report_content else raw_error_output
        
        if not content_to_parse:
            return locations
        
        # Extract all assertion failures with line numbers
        # Pattern: "at package.ClassName.methodName(FileName.java:123)"
        lines = content_to_parse.split('\n')
        
        current_error_message = None
        for i, line in enumerate(lines):
            # Capture error message (e.g., "org.junit.ComparisonFailure: expected:<2> but was:<1>")
            # Also capture Mockito verification failures
            error_keywords = [
                'AssertionError', 'ComparisonFailure', 'AssertionFailedError',
                'ArgumentsAreDifferent', 'WantedButNotInvoked', 'NeverWantedButInvoked',
                'TooManyActualInvocations', 'TooFewActualInvocations', 'NoInteractionsWanted',
                'org.mockito.exceptions.verification'
            ]
            if any(keyword in line for keyword in error_keywords):
                # Extract the message part after the exception type
                if ':' in line:
                    current_error_message = line.split(':', 1)[1].strip()
                else:
                    current_error_message = line.strip()
            
            # Look for stack trace line with test class and method
            if test_class in line and test_method in line:
                # Extract line number from pattern like "ClassName.java:123)"
                match = re.search(r'\((\w+\.java):(\d+)\)', line)
                if match:
                    file_name, line_num = match.groups()
                    line_num = int(line_num)
                    
                    # Now read the actual code from the test file
                    test_file_path = None
                    if test_result.focal_method_info.source_file_path:
                        # Derive test file path from source file path
                        # e.g., "hutool-core/src/main/java/..." -> "hutool-core/src/test/java/..."
                        source_path = test_result.focal_method_info.source_file_path
                        test_file_path = source_path.replace('/src/main/java/', '/src/test/java/')
                        # Replace the source class name with test class name
                        source_class_file = test_result.focal_method_info.class_name.replace('.', '/') + '.java'
                        test_class_file = test_class.replace('.', '/') + '.java'
                        test_file_path = test_file_path.replace(source_class_file, test_class_file)
                    
                    # Read the code line from test_result.test_case.code (not from file!)
                    # The file has been restored by JavaTestExecutor.finally
                    code_line = self._extract_line_from_test_code(
                        test_result.test_case.code,
                        line_num,
                        test_result.test_method_start_line
                    )
                    
                    if code_line:
                        locations.append({
                            'file_line': line_num,
                            'code': code_line,
                            'error_message': current_error_message or error_message
                        })
                        self.log_info(f"Extracted assertion failure: line {line_num}, code: {code_line[:50]}...")
        
        return locations
    
    def _extract_line_from_test_code(self, test_code: str, file_line_number: int, 
                                     test_method_start_line: int) -> str:
        """
        Extract a specific line from test code based on file line number and method start line.
        
        Args:
            test_code: The test method code (what was executed)
            file_line_number: Line number from Maven error (1-indexed, relative to full file)
            test_method_start_line: Line number where the test method starts in the file (1-indexed)
            
        Returns:
            The code line, or empty string if not found
        """
        if not test_code or test_method_start_line == 0:
            return ""
        
        # Calculate relative line number within the test method
        # file_line_number is absolute, test_method_start_line is where @Test starts
        relative_line = file_line_number - test_method_start_line
        
        lines = test_code.strip().split('\n')
        
        # Check if relative_line is within bounds
        if 0 <= relative_line < len(lines):
            return lines[relative_line].strip()
        
        # Fallback: return empty string
        self.log_warning(f"Line number {file_line_number} out of range "
                        f"(method starts at {test_method_start_line}, has {len(lines)} lines)")
        return ""
    
    def _format_test_class_context(self, test_case) -> str:
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
            sections.append(f"""Test Class Fields:
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
            sections.append(f"""Test Class Helper Methods:
```java
{methods_text}
```
These are non-test methods in the same test class (e.g., @Before, @After, helper methods).""")
        
        return "\n\n".join(sections) + "\n"
    
    def _extract_key_error_details(self, raw_error_output: str) -> str:
        """
        Extract key error details from raw Maven output.
        Extracts complete error details including symbol and location information for:
        - Compilation errors
        - Runtime errors (assertion failures, Mockito verification failures, etc.)
        """
        lines = raw_error_output.split('\n')
        error_lines = []
        
        # Strategy 1: Extract compilation errors
        in_compilation_error = False
        for line in lines:
            if '[ERROR] COMPILATION ERROR' in line or 'COMPILATION ERROR' in line:
                in_compilation_error = True
                continue
            
            if any(skip in line for skip in ['BUILD FAILURE', 'Failed to execute goal', '-> [Help', 'To see the full', 'Re-run Maven', 'For more information', 'After correcting']):
                continue
            
            if in_compilation_error:
                if line.strip().startswith('[ERROR]') or (line.startswith('  ') and line.strip()):
                    error_lines.append(line)
                elif line.strip().startswith('[INFO]') and 'error' in line.lower():
                    break
        
        # Strategy 2: If no compilation errors, extract test failure details
        if not error_lines:
            in_test_failure = False
            failure_start_patterns = [
                'FAILURE!', 'FAILURES!', 'ERROR!', 'ERRORS!',
                'AssertionError', 'ComparisonFailure', 'AssertionFailedError',
                'ArgumentsAreDifferent', 'WantedButNotInvoked', 'NeverWantedButInvoked',
                'org.mockito.exceptions', 'org.junit', 'java.lang.AssertionError'
            ]
            
            for i, line in enumerate(lines):
                # Check if this line indicates a test failure
                if any(pattern in line for pattern in failure_start_patterns):
                    in_test_failure = True
                    # Look back to capture test name
                    if i > 0:
                        prev_line = lines[i-1]
                        if 'Time elapsed:' in prev_line or 'Test set:' in prev_line:
                            error_lines.append(prev_line)
                    error_lines.append(line)
                    continue
                
                # If we're in a test failure, keep collecting lines
                if in_test_failure:
                    # Stop conditions
                    if line.strip().startswith('---') and len(line.strip()) > 10:
                        # Next test separator
                        break
                    if 'Results :' in line or 'Tests run:' in line.strip()[:15]:
                        # Summary line - include it and stop
                        error_lines.append(line)
                        break
                    
                    # Include stack traces and error details
                    if line.strip():
                        error_lines.append(line)
        
        if error_lines:
            # Limit to reasonable size (first 50 lines of error detail)
            return '\n'.join(error_lines[:50])
        
        return ""
    
    def _get_error_location_note(self, error_locations: List[Dict[str, Any]]) -> str:
        """
        Get a note about whether errors were located in the test code.
        
        Args:
            error_locations: List of error location dicts
            
        Returns:
            A note to include in the prompt
        """
        if error_locations and len(error_locations) > 0:
            return f"""
NOTE: {len(error_locations)} error location(s) have been marked with ⚠️ in the test code below.
These markers indicate the exact lines where the error occurred. Focus your analysis on these marked lines.
"""
        else:
            return """
⚠️ IMPORTANT NOTE: The error did NOT occur within the test code itself (no specific lines could be pinpointed).
This typically means:
1. The error might be in the test setup, imports, or class-level configuration
2. The error might be in external dependencies or the code being tested
3. The test might need adjustments to work with changes in the focal method

Even though the error is not directly in the test method body, you still need to update the test code to resolve the issue.
Focus on analyzing what changes are needed in the test to make it work correctly.
"""
    
    def _inject_error_comments(self, test_code: str, error_locations: List[Dict[str, Any]], 
                              test_method_start_line: int = 0) -> str:
        """
        Inject error messages as comments into test code at error locations.
        
        Args:
            test_code: Original test code (only the test method)
            error_locations: List of error location dicts with 'file_line' and 'error_message'
            test_method_start_line: Line number where the test method starts in the file (1-indexed)
            
        Returns:
            Test code with error comments injected
        """
        if not error_locations:
            return test_code
        
        lines = test_code.split('\n')
        
        # Create a map of line numbers (relative to test code) to error messages
        error_map = {}
        for loc in error_locations:
            file_line = loc.get('file_line', 0)
            error_msg = loc.get('error_message', '')
            column = loc.get('column')
            
            # Include column info in error message if available
            if column is not None:
                error_msg = f"{error_msg} (at column {column})"
            
            if file_line > 0 and test_method_start_line > 0:
                # Convert file line number to relative line number within test code
                # file_line is 1-indexed absolute line in file
                # test_method_start_line is 1-indexed line where @Test starts
                relative_line = file_line - test_method_start_line
                
                # relative_line is now 0-indexed relative to test_code
                if 0 <= relative_line < len(lines):
                    if relative_line not in error_map:
                        error_map[relative_line] = []
                    error_map[relative_line].append(error_msg)
        
        # Inject error comments
        result_lines = []
        for i, line in enumerate(lines):
            if i in error_map:
                # Extract the indentation of the current line
                indent = len(line) - len(line.lstrip())
                indent_str = ' ' * indent
                
                # Add error comment(s) BEFORE the line with matching indentation
                for error_msg in error_map[i]:
                    # Detect if it's a compilation error or assertion failure
                    if any(keyword in error_msg for keyword in ['expected:', 'but was:', 'ComparisonFailure']):
                        result_lines.append(f"{indent_str}// ⚠️ ASSERTION FAILED: {error_msg}")
                    else:
                        result_lines.append(f"{indent_str}// ⚠️ COMPILATION ERROR: {error_msg}")
            
            # Then add the original line
            result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    def _analyze_compilation_error(self, test_result: TestResultInfo,
                                   filtered_hunks: List[DiffHunk],
                                   error_classification: Dict[str, Any],
                                   retrieval_result: Optional[RetrievalResult]) -> Dict[str, str]:
        """
        Analyze compilation error using LLM.
        
        Args:
            test_result: Test result info
            filtered_hunks: Filtered diff hunks
            error_classification: Classification result
            retrieval_result: Retrieval result (if any)
            
        Returns:
            Dict with 'root_cause' and 'explanation'
        """
        # Inject error comments into test code
        annotated_test_code = self._inject_error_comments(
            test_result.test_case.code,
            error_classification.get('error_locations', []),
            test_result.test_method_start_line
        )
        
        # Get already added imports from previous iterations
        already_added_imports = getattr(test_result.test_case, 'new_imports', []) or []
        already_added_imports_text = ""
        if already_added_imports:
            already_added_imports_text = f"""
IMPORTANT - Previously Added Imports:
The following imports were added in previous iterations:
{chr(10).join(f'  - {imp}' for imp in already_added_imports)}

Instructions for handling these imports:
1. If you believe an import is STILL NEEDED and CORRECT, include it again in your output
2. If you believe an import is INCORRECT or CAUSING ERRORS, DO NOT include it in your output
3. If you believe an import is NO LONGER NEEDED, DO NOT include it in your output
4. In addition to reconsidering the above imports, identify and add any OTHER missing imports that are needed

Your output should contain:
- Previously added imports that you still think are necessary
- Any NEW imports that are needed beyond the ones listed above
"""
        
        # Build prompt based on whether retrieval is available
        if retrieval_result:
            # Normal mode: with retrieval information
            retrieval_section = f"""
Retrieved Information:
{self._format_retrieval_by_symbol(retrieval_result)}
"""
            guidelines = """CRITICAL CONSTRAINTS:
- You can ONLY modify the test code itself
- You can ONLY add import statements
- You CANNOT modify the focal method or any production code (classes being tested)
- You CANNOT change access modifiers of production classes (e.g., changing private to protected)
- You CANNOT refactor production code
- If a symbol is inaccessible (private), use mocking, reflection, or indirect testing instead

Guidelines for Fix Strategies:
1. FOCUS on the error lines marked with ⚠️
2. For known symbols: Suggest specific import statements
3. For unknown symbols with retrieval results: Suggest import or code changes based on retrieved info
4. Provide CONCRETE and DEFINITIVE actions, not conditional suggestions
5. If you're uncertain about a condition, provide a GENERAL/HIGH-LEVEL strategy instead of listing multiple if-else branches
6. AVOID patterns like "If X, do A; If not X, do B; If both fail, do C"
7. Each strategy should be a single actionable step that can be directly applied
8. DO NOT suggest adding imports that are already listed in "Already Added Imports" above"""
        else:
            # Ablation mode: without retrieval
            retrieval_section = """
Note: Retrieval system is disabled (ablation experiment). Analyze based on error message and code changes only.
"""
            guidelines = """CRITICAL CONSTRAINTS:
- You can ONLY modify the test code itself
- You can ONLY add import statements
- You CANNOT modify the focal method or any production code (classes being tested)
- You CANNOT change access modifiers of production classes (e.g., changing private to protected)
- You CANNOT refactor production code
- If a symbol is inaccessible (private), use mocking, reflection, or indirect testing instead

Guidelines for Fix Strategies:
1. FOCUS on the error lines marked with ⚠️
2. For known symbols: Suggest specific import statements
3. For unknown symbols: Suggest possible fixes based on error message and code changes
4. Provide CONCRETE and DEFINITIVE actions, not conditional suggestions
5. If you're uncertain about a condition, provide a GENERAL/HIGH-LEVEL strategy instead of listing multiple if-else branches
6. AVOID patterns like "If X, do A; If not X, do B; If both fail, do C"
7. Each strategy should be a single actionable step that can be directly applied
8. DO NOT suggest adding imports that are already listed in "Already Added Imports" above"""
        
        # Get error location note
        error_location_note = self._get_error_location_note(error_classification.get('error_locations', []))
        
        prompt = f"""
Analyze the following compilation error and provide root cause and possible fix strategies:

Error Type: Compilation Error
Error Message: {test_result.error_message}

Symbol Analysis:
- Known symbols (need imports): {', '.join(error_classification['known_symbols']) if error_classification['known_symbols'] else 'None'}
- Unknown symbols (project-specific): {', '.join(error_classification['unknown_symbols']) if error_classification['unknown_symbols'] else 'None'}

{already_added_imports_text}{retrieval_section}{error_location_note}
Test Code (with error locations marked with ⚠️):
{annotated_test_code}

Code Changes:
{self._format_hunks(filtered_hunks)}

YOUR TASK:
Focus on the lines marked with ⚠️ COMPILATION ERROR. Analyze the root cause and provide actionable fix strategies.

Response format (JSON):
{{
  "root_cause": "Brief description of why compilation fails (focus on error lines)",
  "fix_strategies": [
    "Strategy 1: Specific fix action (e.g., 'Add import: import java.util.List')",
    "Strategy 2: Another possible fix (e.g., 'Rename symbol X to Y and add import')",
    "..."
  ]
}}

{guidelines}
"""
        
        system_prompt = """You are an expert at analyzing Java compilation errors and providing concrete fix strategies.
Focus on the ERROR LINES and give actionable steps, not explanations."""
        
        response_text = self.llm_client.generate(prompt, system_prompt=system_prompt)
        
        # Parse JSON response
        import json
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                response = json.loads(json_match.group())
                # Convert fix_strategies list to explanation string for backward compatibility
                fix_strategies = response.get('fix_strategies', [])
                explanation = '\n'.join(f"- {s}" for s in fix_strategies) if fix_strategies else response.get('explanation', response_text)
                return {
                    'root_cause': response.get('root_cause', 'Compilation error'),
                    'explanation': explanation
                }
            except json.JSONDecodeError:
                pass
        
        # Fallback
        return {
            'root_cause': 'Compilation error',
            'explanation': response_text
        }
    
    def _analyze_assertion_failure(self, test_result: TestResultInfo,
                                   filtered_hunks: List[DiffHunk],
                                   error_classification: Dict[str, Any],
                                   retrieval_result: Optional[RetrievalResult]) -> Dict[str, str]:
        """
        Analyze assertion failure using LLM.
        
        Args:
            test_result: Test result info
            filtered_hunks: Filtered diff hunks
            error_classification: Classification result
            retrieval_result: Retrieval result (if any)
            
        Returns:
            Dict with 'root_cause' and 'explanation'
        """
        # Inject error comments into test code
        annotated_test_code = self._inject_error_comments(
            test_result.test_case.code,
            error_classification.get('error_locations', []),
            test_result.test_method_start_line
        )
        
        # Build retrieval section based on availability
        if retrieval_result:
            retrieval_section = f"""
Retrieved Information (if any):
{self._format_retrieval_by_symbol(retrieval_result)}
"""
        else:
            retrieval_section = """
Note: Retrieval system is disabled (ablation experiment). Analyze based on error message and code changes only.
"""
        
        # Extract key error details from raw output for better context
        detailed_error = ""
        if test_result.raw_error_output:
            key_errors = self._extract_key_error_details(test_result.raw_error_output)
            if key_errors:
                detailed_error = f"\n\nDetailed Error Output:\n{key_errors}\n"
        
        # Get error location note
        error_location_note = self._get_error_location_note(error_classification.get('error_locations', []))
        
        prompt = f"""
Analyze the following assertion failure and provide root cause and possible fix strategies:

Error Type: Assertion Failure
Error Message: {test_result.error_message}{detailed_error}
Symbol Analysis (if any):
- Unknown symbols: {', '.join(error_classification['unknown_symbols']) if error_classification['unknown_symbols'] else 'None'}
{retrieval_section}{error_location_note}
Test Code (with failed assertion locations marked with ⚠️):
{annotated_test_code}

Code Changes:
{self._format_hunks(filtered_hunks)}

YOUR TASK:
Focus on the lines marked with ⚠️ ASSERTION FAILED. Analyze why the assertion failed and provide actionable fix strategies.

Response format (JSON):
{{
  "root_cause": "Brief description of why assertion fails (focus on the failed assertion lines)",
  "fix_strategies": [
    "Strategy 1: Specific fix action (e.g., 'Change expected value from X to Y in line N')",
    "Strategy 2: Another possible fix (e.g., 'Update assertion method from assertEquals to assertThat')",
    "..."
  ]
}}

CRITICAL CONSTRAINTS:
- You can ONLY modify the test code itself (assertions, test logic, setup)
- You can ONLY add import statements if needed
- You CANNOT modify the focal method or any production code (classes being tested)
- You CANNOT change the behavior of production code
- The test must adapt to the NEW behavior of the focal method (based on code changes)

Guidelines for Fix Strategies:
1. FOCUS on the assertion lines marked with ⚠️
2. Analyze the code changes to understand the NEW expected behavior
3. Provide CONCRETE and DEFINITIVE fix actions (e.g., "Change expected value from X to Y in line N")
4. If you're uncertain about a condition, provide a GENERAL/HIGH-LEVEL strategy instead of listing multiple if-else branches
5. AVOID patterns like "If X, do A; If not X, do B; If both fail, do C"
6. Each strategy should be a single actionable step that can be directly applied
7. Do NOT just explain what changed - suggest HOW to fix the test to match new behavior
"""
        
        system_prompt = """You are an expert at analyzing test assertion failures and providing concrete fix strategies.
Focus on the FAILED ASSERTION LINES and give actionable steps to update the test to match the new behavior of the focal method.
Provide definitive strategies, not conditional ones."""
        
        response_text = self.llm_client.generate(prompt, system_prompt=system_prompt)
        
        # Parse JSON response
        import json
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                response = json.loads(json_match.group())
                # Convert fix_strategies list to explanation string for backward compatibility
                fix_strategies = response.get('fix_strategies', [])
                explanation = '\n'.join(f"- {s}" for s in fix_strategies) if fix_strategies else response.get('explanation', response_text)
                return {
                    'root_cause': response.get('root_cause', 'Assertion failure'),
                    'explanation': explanation
                }
            except json.JSONDecodeError:
                pass
        
        # Fallback
        return {
            'root_cause': 'Assertion failure',
            'explanation': response_text
        }
    
    def _analyze_general_error(self, test_result: TestResultInfo,
                               filtered_hunks: List[DiffHunk],
                               error_classification: Dict[str, Any],
                               retrieval_result: Optional[RetrievalResult]) -> Dict[str, str]:
        """
        Analyze general error using LLM.
        
        Fallback for unclassified errors.
        """
        # Inject error comments into test code if available
        annotated_test_code = self._inject_error_comments(
            test_result.test_case.code,
            error_classification.get('error_locations', []),
            test_result.test_method_start_line
        )
        
        # Extract key error details from raw output for better context
        detailed_error = ""
        if test_result.raw_error_output:
            key_errors = self._extract_key_error_details(test_result.raw_error_output)
            if key_errors:
                detailed_error = f"\n\nDetailed Error Output:\n{key_errors}\n"
        
        # Get error location note
        error_location_note = self._get_error_location_note(error_classification.get('error_locations', []))
        
        prompt = f"""
Analyze the following test failure and provide root cause and possible fix strategies:

Error Type: {test_result.status}
Error Message: {test_result.error_message}{detailed_error}{error_location_note}
Test Code (with error locations marked if available):
{annotated_test_code}

Code Changes:
{self._format_hunks(filtered_hunks)}

YOUR TASK:
Analyze the error and provide actionable fix strategies. Focus on any lines marked with ⚠️.

Response format (JSON):
{{
  "root_cause": "Brief description of why the test fails",
  "fix_strategies": [
    "Strategy 1: Specific fix action",
    "Strategy 2: Another possible fix",
    "..."
  ]
}}

CRITICAL CONSTRAINTS:
- You can ONLY modify the test code itself
- You can ONLY add import statements if needed
- You CANNOT modify the focal method or any production code (classes being tested)
- You CANNOT change access modifiers of production classes
- You CANNOT refactor production code

Guidelines for Fix Strategies:
1. Focus on error lines if marked with ⚠️
2. Provide CONCRETE and DEFINITIVE actions, not conditional suggestions
3. If you're uncertain about a condition, provide a GENERAL/HIGH-LEVEL strategy instead of listing multiple if-else branches
4. AVOID patterns like "If X, do A; If not X, do B; If both fail, do C"
5. Each strategy should be a single actionable step that can be directly applied
6. Consider the code changes when suggesting fixes
"""
        
        system_prompt = """You are an expert at analyzing test failures and providing concrete fix strategies.
Give definitive actionable steps that can be directly applied, not conditional explanations."""
        
        response_text = self.llm_client.generate(prompt, system_prompt=system_prompt)
        
        # Parse JSON response
        import json
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            try:
                response = json.loads(json_match.group())
                # Convert fix_strategies list to explanation string for backward compatibility
                fix_strategies = response.get('fix_strategies', [])
                explanation = '\n'.join(f"- {s}" for s in fix_strategies) if fix_strategies else response_text
                return {
                    'root_cause': response.get('root_cause', f'Test failure: {test_result.status}'),
                    'explanation': explanation
                }
            except json.JSONDecodeError:
                pass
        
        # Fallback
        return {
            'root_cause': f"Test failure: {test_result.status}",
            'explanation': f"Error message: {test_result.error_message}\n\nPlease review the error details and code changes to determine the fix."
        }
    
    def _format_error_locations(self, error_locations: List[Dict[str, Any]]) -> str:
        """Format error locations for prompt"""
        if not error_locations:
            return "No specific error locations identified"
        
        formatted = []
        for i, loc in enumerate(error_locations[:5], 1):  # Show first 5
            formatted.append(f"Error {i}:")
            formatted.append(f"  Line {loc.get('file_line', '?')}: {loc.get('code', '')}")
            formatted.append(f"  Message: {loc.get('error_message', '')}")
        
        if len(error_locations) > 5:
            formatted.append(f"... and {len(error_locations) - 5} more error(s)")
        
        return '\n'.join(formatted)
    
    def _format_retrieval_by_symbol(self, retrieval_result: Optional[RetrievalResult]) -> str:
        """Format retrieval results grouped by symbol"""
        if not retrieval_result or not retrieval_result.retrieved_items:
            return "No retrieval performed or no results found"
        
        formatted = []
        for symbol, items in retrieval_result.retrieved_items.items():
            formatted.append(f"\n{symbol}:")
            
            if not items:
                formatted.append("  No matches found")
                continue
            
            # Show top 3 items
            for item in items[:3]:
                if 'method_name' in item:
                    # Method
                    formatted.append(f"  - {item.get('class_name', '?')}.{item.get('method_name', '?')}")
                    formatted.append(f"    Signature: {item.get('signature', 'N/A')}")
                elif 'field_name' in item:
                    # Field
                    formatted.append(f"  - {item.get('class_name', '?')}.{item.get('field_name', '?')}: {item.get('field_type', '?')}")
            
            if len(items) > 3:
                formatted.append(f"  ... and {len(items) - 3} more")
        
        if retrieval_result.failed_symbols:
            formatted.append(f"\nFailed to retrieve: {', '.join(retrieval_result.failed_symbols)}")
        
        return '\n'.join(formatted)
    
    
    def _format_hunks(self, filtered_hunks: List[DiffHunk]) -> str:
        """Format diff hunks"""
        if not filtered_hunks:
            return "No relevant code changes"
        
        formatted = []
        for i, hunk in enumerate(filtered_hunks[:5], 1):  # Show first 5
            hunk_type = hunk.hunk_type or "unknown"
            formatted.append(f"\nHunk {i} [{hunk_type}] - {hunk.file_path}:")
            formatted.append(hunk.context)
        
        if len(filtered_hunks) > 5:
            formatted.append(f"\n... and {len(filtered_hunks) - 5} more hunks")
        
        return '\n'.join(formatted)
