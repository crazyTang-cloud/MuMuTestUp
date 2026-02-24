"""
LLM Agent wrapper for pipeline decision-making.

Handles all LLM interactions including triage, SQL generation,
result evaluation, and RAG query generation.
"""
import logging
import json
from typing import Dict, Any, Optional, List, Tuple
from openai import OpenAI
import time

logger = logging.getLogger(__name__)


class LLMAgent:
    """
    LLM agent for multi-phase decision making in the pipeline.
    
    Uses structured outputs and chain-of-thought prompting for reliable
    decisions at each phase.
    """
    
    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.1,
        max_retries: int = 3,
        timeout: int = 3000
    ):
        """
        Initialize LLM agent.
        
        Args:
            model: Model name (e.g., "gpt-4o-2024-11-20")
            api_key: OpenAI API key
            base_url: API base URL
            temperature: Sampling temperature (lower = more deterministic)
            max_retries: Maximum retry attempts for API calls
            timeout: Request timeout in seconds
        """
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout
        )
        
        logger.info(f"LLM Agent initialized with model: {model}, timeout: {timeout}s")
    
    def ask(self, prompt: str, response_format: Optional[Dict[str, Any]] = None) -> str:
        """
        Simple interface for asking LLM a question.
        
        Args:
            prompt: The prompt/question to ask
            response_format: Optional JSON schema for structured output
        
        Returns:
            The LLM's response as a string
        
        Raises:
            Exception: If LLM call fails after all retries
        """
        # Use a generic system prompt for general queries
        system_prompt = "You are a helpful AI assistant specialized in software engineering and code analysis."
        
        success, text, json_data = self._call_llm(system_prompt, prompt, response_format)
        
        if not success or text is None:
            raise Exception("LLM call failed after all retries")
        
        return text
    
    def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, Optional[str], Optional[Dict[str, Any]]]:
        """
        Make LLM API call with retry logic.
        
        Args:
            system_prompt: System role prompt
            user_prompt: User message
            response_format: Optional JSON schema for structured output
        
        Returns:
            Tuple of (success, text_response, json_response)
        """
        for attempt in range(self.max_retries):
            try:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature
                }
                
                # Add response format if specified
                if response_format:
                    kwargs["response_format"] = response_format
                
                response = self.client.chat.completions.create(**kwargs)
                
                content = response.choices[0].message.content
                
                # Try to parse as JSON if response_format was specified
                json_data = None
                if response_format:
                    try:
                        json_data = json.loads(content)
                    except json.JSONDecodeError:
                        # Try to extract JSON from text (in case LLM added extra text)
                        import re
                        json_match = re.search(r'\{.*\}', content, re.DOTALL)
                        if json_match:
                            try:
                                json_data = json.loads(json_match.group())
                                logger.debug("Extracted JSON from response text")
                            except json.JSONDecodeError:
                                logger.warning("Failed to parse LLM response as JSON")
                        else:
                            logger.warning("Failed to parse LLM response as JSON")
                
                return True, content, json_data
                
            except Exception as e:
                logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"LLM call failed after {self.max_retries} attempts")
                    return False, None, None
        
        return False, None, None
    
    def triage_error(
        self,
        error_test_code: str,
        error_message: str,
        error_log: str
    ) -> Tuple[bool, str, str]:
        """
        Phase 1: Analyze if error is simple enough to fix without repo context.
        
        Args:
            error_test_code: The failing test code
            error_message: Short error message
            error_log: Full error log
        
        Returns:
            Tuple of (can_skip, reasoning, decision)
            - can_skip: True if error is simple enough to skip retrieval
            - reasoning: Explanation of the decision
            - decision: "SKIP" or "PROCEED"
        """
        logger.info("Phase 1: Triaging error complexity...")
        
        system_prompt = """You are a senior software engineer analyzing test failures.
Your task is to determine if a test failure is simple enough to fix WITHOUT needing to examine the production code repository.

Simple errors that DON'T need repository context:
- Obvious syntax errors in the test itself
- Simple assertion logic errors
- Incorrect expected values that are self-evident
- Missing imports that are clear from the error
- Simple null pointer issues with obvious fixes

Complex errors that NEED repository context:
- Type mismatches requiring understanding of production code
- Unexpected behavior from production methods
- Integration issues between components
- Errors requiring knowledge of class structure or method implementations
- Any error where the fix depends on understanding production code logic

Respond in JSON format with:
{
  "decision": "SKIP" or "PROCEED",
  "reasoning": "Brief explanation of your decision",
  "confidence": "high" or "medium" or "low"
}

Be conservative: when in doubt, choose "PROCEED" to gather more context."""
        
        user_prompt = f"""Analyze this test failure:

**Test Code:**
```java
{error_test_code}
```

**Error Message:**
{error_message}

**Error Log (relevant excerpt):**
{error_log[:2000]}

Should we skip further repository analysis (SKIP) or proceed to gather context (PROCEED)?"""
        
        response_format = {
            "type": "json_object"
        }
        
        success, text, json_data = self._call_llm(system_prompt, user_prompt, response_format)
        
        if not success or not json_data:
            # Default to PROCEED on error
            return False, "LLM call failed, proceeding to be safe", "PROCEED"
        
        decision = json_data.get("decision", "PROCEED")
        reasoning = json_data.get("reasoning", "No reasoning provided")
        
        can_skip = (decision == "SKIP")
        
        logger.info(f"Triage decision: {decision} - {reasoning}")
        
        return can_skip, reasoning, decision
    
    def generate_sql_query(
        self,
        error_context: str,
        previous_queries: List[str] = None,
        previous_results: List[str] = None,
        round_num: int = 1
    ) -> Tuple[bool, str, str]:
        """
        Phase 2: Generate SQL query to search metadata.
        
        Args:
            error_context: Error information
            previous_queries: Previous SQL queries attempted
            previous_results: Results from previous queries
            round_num: Current round number
        
        Returns:
            Tuple of (success, sql_query, reasoning)
        """
        logger.info(f"Phase 2: Generating SQL query (round {round_num})...")
        
        system_prompt = """You are a database query expert helping to find relevant code metadata.

The SQLite database has two tables:

1. 'methods' table - Available columns:
   - id TEXT PRIMARY KEY
   - module_name TEXT
   - file_path TEXT
   - class_name TEXT (fully qualified)
   - method_name TEXT
   - signature TEXT (full method signature, includes modifiers like public/private/static)
   - javadoc TEXT
   - body TEXT (full method source code)
   - start_line INTEGER
   - end_line INTEGER
   - is_test BOOLEAN
   - imports TEXT (JSON array)

   ⚠️ CRITICAL: When querying 'methods' table, do NOT use these columns (they don't exist):
   - field_name, field_type, modifiers, initializer

2. 'fields' table - Available columns:
   - id TEXT PRIMARY KEY
   - module_name TEXT
   - file_path TEXT
   - class_name TEXT (fully qualified)
   - field_name TEXT
   - field_type TEXT
   - modifiers TEXT (e.g., "private static final")
   - initializer TEXT (field initializer/value if available)
   - javadoc TEXT
   - start_line INTEGER
   - end_line INTEGER
   - is_test BOOLEAN
   - imports TEXT (JSON array)

   ⚠️ CRITICAL: When querying 'fields' table, do NOT use these columns (they don't exist):
   - method_name, signature, body

CRITICAL CONSTRAINTS:
1. MUST include: WHERE is_test = 0 (or = false) - NEVER search test code
2. MUST include: LIMIT clause (recommend 3-5 for deep analysis with body, 10+ for metadata only)
3. ONLY SELECT queries allowed

QUERY STRATEGY:
- For understanding method implementations: SELECT body, signature, javadoc, class_name, method_name, file_path FROM methods (limit 3-5)
- For browsing available methods: SELECT class_name, method_name, signature FROM methods (limit 10+)
- For finding field/constant values: SELECT class_name, field_name, field_type, initializer, javadoc FROM fields (limit 5-10)
- **Including body field allows deeper analysis but uses more tokens**
- **IMPORTANT: Always include method_name/field_name and class_name in SELECT to identify the item**

TIP: If previous queries found metadata but were insufficient, try INCLUDING the body field 
to see actual implementations and understand root causes.

Respond in JSON format:
{
  "sql": "Your SQL query here",
  "reasoning": "Why this query will help",
  "expected_findings": "What you expect to find"
}"""
        
        user_prompt = f"""Generate a SQL query to find relevant production code for this error:

{error_context}
"""
        
        if previous_queries:
            user_prompt += f"\n**Previous queries tried:**\n"
            for i, (q, r) in enumerate(zip(previous_queries, previous_results or [])):
                user_prompt += f"\nRound {i+1} Query:\n{q}\n"
                if r:
                    user_prompt += f"Result: {r[:500]}...\n"
        
        user_prompt += "\nGenerate a SQL query that will help locate the relevant code. Be specific and targeted."
        
        response_format = {"type": "json_object"}
        
        success, text, json_data = self._call_llm(system_prompt, user_prompt, response_format)
        
        if not success or not json_data:
            return False, "", "Failed to generate SQL query"
        
        sql = json_data.get("sql", "")
        reasoning = json_data.get("reasoning", "")
        
        logger.debug(f"Generated SQL: {sql}")
        
        return True, sql, reasoning
    
    def validate_sql(self, sql: str) -> Tuple[bool, str]:
        """
        Validate SQL query for safety constraints.
        
        Args:
            sql: SQL query to validate
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        sql_lower = sql.lower().strip()
        
        # Must be SELECT only
        if not sql_lower.startswith('select'):
            return False, "Only SELECT queries are allowed"
        
        # Must have WHERE is_test = 0
        if 'is_test' not in sql_lower:
            return False, "Query must filter by is_test column"
        
        # Check for is_test = 0 or is_test = false
        has_test_filter = ('is_test = 0' in sql_lower or 
                          'is_test=0' in sql_lower or
                          'is_test = false' in sql_lower or
                          'is_test=false' in sql_lower)
        
        if not has_test_filter:
            return False, "Query must include 'WHERE is_test = 0' to exclude test code"
        
        # Must have LIMIT clause
        if 'limit' not in sql_lower:
            return False, "Query must include LIMIT clause for cost control"
        
        # No dangerous keywords
        dangerous = ['drop', 'delete', 'update', 'insert', 'alter', 'create', 'truncate']
        for keyword in dangerous:
            if keyword in sql_lower:
                return False, f"Dangerous keyword '{keyword}' not allowed"
        
        return True, "SQL query is valid"
    
    def evaluate_sql_results(
        self,
        error_context: str,
        sql_query: str,
        results: List[Dict[str, Any]],
        round_num: int,
        max_rounds: int
    ) -> Tuple[str, str, Optional[str]]:
        """
        Evaluate if SQL query results provide sufficient information.
        
        Args:
            error_context: Error information
            sql_query: The SQL query that was executed
            results: Query results
            round_num: Current round number
            max_rounds: Maximum allowed rounds
        
        Returns:
            Tuple of (decision, reasoning, extracted_context)
            - decision: "SUFFICIENT", "RETRY", or "PROCEED_TO_RAG"
            - reasoning: Explanation
            - extracted_context: Useful info extracted (if sufficient)
        """
        logger.info(f"Phase 2: Evaluating SQL results (round {round_num}/{max_rounds})...")
        
        if not results:
            if round_num >= max_rounds:
                return "PROCEED_TO_RAG", "No results found after max rounds", None
            else:
                return "RETRY", "No results, will refine query", None
        
        system_prompt = """You are evaluating whether SQL query results provide enough information to fix a test failure.

The results contain metadata about production code methods, including:
- Signatures, javadoc documentation
- Full method bodies (source code) if included in the query
- Class names, file paths, imports

Evaluate if these results are SUFFICIENT to understand and fix the error, or if we need:
- RETRY: Refine the SQL query for better results (e.g., query different methods, include body field)
- SUFFICIENT: We have enough information to proceed with fixing
- PROCEED_TO_RAG: Need semantic search to find related code

IMPORTANT: If you see method bodies (body field), carefully analyze them to determine if they contain the root cause or fix hints.

Respond in JSON:
{
  "decision": "SUFFICIENT" or "RETRY" or "PROCEED_TO_RAG",
  "reasoning": "Explanation",
  "extracted_context": "If SUFFICIENT, extract ALL key relevant information including method bodies, signatures, and javadocs that will help fix the error"
}"""
        
        # Format results for LLM - provide more complete information
        # Take up to 5 results and allow more tokens for body content
        results_to_show = results[:5]
        
        # Format each result clearly, highlighting important fields
        formatted_results = []
        for i, result in enumerate(results_to_show):
            formatted = f"\n--- Result {i+1} ---"
            
            # Always show metadata
            if 'class_name' in result:
                formatted += f"\nClass: {result['class_name']}"
            if 'method_name' in result:
                formatted += f"\nMethod: {result['method_name']}"
            if 'signature' in result:
                formatted += f"\nSignature: {result['signature']}"
            if 'javadoc' in result and result['javadoc']:
                formatted += f"\nJavadoc: {result['javadoc'][:500]}"  # Limit javadoc
            
            # Show full method body if available (this is the key improvement!)
            if 'body' in result and result['body']:
                formatted += f"\n\nMethod Body:\n{result['body']}"
            
            # Show other useful fields
            if 'file_path' in result:
                formatted += f"\nFile: {result['file_path']}"
            if 'module_name' in result:
                formatted += f"\nModule: {result['module_name']}"
            
            formatted_results.append(formatted)
        
        results_str = "\n".join(formatted_results)
        
        # Limit total length but be more generous (10K tokens instead of 3K chars)
        max_length = 10000
        if len(results_str) > max_length:
            results_str = results_str[:max_length] + f"\n\n... (truncated, showing {len(results_to_show)}/{len(results)} results)"
        
        user_prompt = f"""Error context:
{error_context}

SQL Query executed:
{sql_query}

Query returned {len(results)} total rows. Showing first {len(results_to_show)} with full content:
{results_str}

Round {round_num} of {max_rounds}.

Carefully analyze the results above, especially any method bodies shown. 
Evaluate: Do these results provide enough information to fix the error?"""
        
        response_format = {"type": "json_object"}
        
        success, text, json_data = self._call_llm(system_prompt, user_prompt, response_format)
        
        if not success or not json_data:
            # If max rounds reached, proceed to RAG
            if round_num >= max_rounds:
                return "PROCEED_TO_RAG", "Evaluation failed, max rounds reached", None
            return "RETRY", "Evaluation failed, will retry", None
        
        decision = json_data.get("decision", "PROCEED_TO_RAG")
        reasoning = json_data.get("reasoning", "")
        extracted = json_data.get("extracted_context")
        
        # Convert extracted context to string if it's a dict/object
        if extracted and isinstance(extracted, dict):
            # Format as pretty JSON string
            extracted = json.dumps(extracted, indent=2, ensure_ascii=False)
        elif extracted and not isinstance(extracted, str):
            # Convert any other type to string
            extracted = str(extracted)
        
        logger.info(f"SQL evaluation: {decision} - {reasoning}")
        
        return decision, reasoning, extracted
    
    def generate_rag_query(
        self,
        error_context: str,
        previous_attempts: List[Dict[str, Any]] = None,
        round_num: int = 1
    ) -> Tuple[bool, str, str]:
        """
        Phase 3: Generate semantic search query for RAG.
        
        Args:
            error_context: Error information
            previous_attempts: Previous search attempts with results
            round_num: Current round number
        
        Returns:
            Tuple of (success, query_text, reasoning)
        """
        logger.info(f"Phase 3: Generating RAG query (round {round_num})...")
        
        system_prompt = """You are generating semantic search queries to find relevant production code.

**CRITICAL: Query Format Must Match Embedding Format**

The vector database contains embeddings in this EXACT format:
```
Class: full.package.ClassName
Method: public ReturnType methodName(ParamType param)
Doc: /** Javadoc content here */
```

Your query MUST follow a similar structure to maximize retrieval accuracy:
```
Class: [package/class hints]
Method: [method purpose and signature hints]
Doc: [what the method should do]
```

Example good queries:
- "Class: condition parsing\nMethod: parse expression to value\nDoc: handles string expressions with operators"
- "Class: database condition\nMethod: static method for parsing field conditions\nDoc: validates and converts expression strings"

BAD queries (too vague, won't match well):
- "Methods for parsing expressions"
- "Find code that handles BigDecimal"

Tips:
- Use the "Class: ... Method: ... Doc: ..." structure
- Be specific about method names, types, and purposes
- Include relevant keywords from the error

Respond in JSON:
{
  "query": "Your structured query following the Class/Method/Doc format",
  "reasoning": "Why this query will find relevant code",
  "target": "methods" or "fields" or "both"
}"""
        
        user_prompt = f"""Generate a semantic search query for this error:

{error_context}
"""
        
        if previous_attempts:
            user_prompt += "\n**Previous search attempts:**\n"
            for i, attempt in enumerate(previous_attempts):
                user_prompt += f"\nRound {i+1}:\n"
                user_prompt += f"Query: {attempt.get('query', '')}\n"
                user_prompt += f"Results: {attempt.get('summary', 'No results')}\n"
        
        user_prompt += "\nGenerate a query that will find the relevant production code."
        
        response_format = {"type": "json_object"}
        
        success, text, json_data = self._call_llm(system_prompt, user_prompt, response_format)
        
        if not success or not json_data:
            # Generate a fallback query
            return True, error_context[:200], "Fallback query from error context"
        
        query = json_data.get("query", error_context[:200])
        reasoning = json_data.get("reasoning", "")
        target = json_data.get("target", "methods")
        
        logger.info(f"Generated RAG query (target={target}): {query[:100]}...")
        
        return True, query, reasoning
    
    def decide_search_target(
        self,
        error_context: str,
        query: str
    ) -> str:
        """
        Decide whether to search methods, fields, or both.
        
        Args:
            error_context: Error information
            query: The search query
        
        Returns:
            "methods", "fields", or "both"
        """
        # Simple heuristic: check if error involves field access
        error_lower = error_context.lower()
        
        field_indicators = ['field', 'variable', 'attribute', 'member', 'property']
        has_field_indicator = any(ind in error_lower for ind in field_indicators)
        
        if has_field_indicator:
            return "both"
        else:
            return "methods"  # Default to methods for most test failures
    
    def decide_rag_action(
        self,
        error_context: str,
        available_modules: List[str],
        embedded_modules: List[str],
        round_num: int,
        previous_results: Optional[str] = None
    ) -> Tuple[str, Optional[str], str]:
        """
        Decide next action in RAG search: embed new module or just refine query.
        
        Args:
            error_context: Error information
            available_modules: List of all available modules in repo
            embedded_modules: List of modules already embedded
            round_num: Current round number
            previous_results: Summary of previous search results (if any)
        
        Returns:
            Tuple of (action, module_name, reasoning)
            - action: "EMBED_MODULE" or "REFINE_QUERY"
            - module_name: Module to embed (if action is EMBED_MODULE), None otherwise
            - reasoning: Explanation of the decision
        """
        logger.info(f"Deciding RAG action for round {round_num}...")
        
        # Calculate remaining modules
        remaining_modules = [m for m in available_modules if m not in embedded_modules]
        
        if not remaining_modules:
            # No more modules to embed, can only refine query
            return "REFINE_QUERY", None, "All modules already embedded, will refine query"
        
        system_prompt = """You are an expert at analyzing test errors and deciding search strategies.

You need to decide the next action in a RAG (Retrieval-Augmented Generation) search for fixing a test error.

Available actions:
1. EMBED_MODULE: Embed a new module's code into the vector database and then search it
2. REFINE_QUERY: Don't embed new code, just refine the search query on already-embedded modules

Cost consideration:
- EMBED_MODULE is expensive (embedding costs money and time)
- REFINE_QUERY is cheap (just changes the query)

Guidelines:
- Round 1: Usually EMBED_MODULE (need to start somewhere)
- Later rounds: 
  * If previous results were close but not quite right → REFINE_QUERY
  * If previous results were totally off-topic → EMBED_MODULE (wrong module)
  * If no results found → EMBED_MODULE (need more code)

Respond in JSON:
{
  "action": "EMBED_MODULE" or "REFINE_QUERY",
  "module_name": "module-name" (only if EMBED_MODULE, otherwise null),
  "reasoning": "Brief explanation of your decision"
}"""
        
        # Build context about current state
        user_prompt = f"""Error context:
{error_context[:1500]}

**Available modules in repository:** {', '.join(available_modules)}

**Already embedded modules:** {', '.join(embedded_modules) if embedded_modules else 'None yet'}

**Remaining modules:** {', '.join(remaining_modules)}

**Current round:** {round_num}
"""
        
        if previous_results:
            user_prompt += f"\n**Previous search results summary:**\n{previous_results}\n"
        
        user_prompt += "\nDecide: Should we EMBED_MODULE (which one?) or REFINE_QUERY?"
        
        response_format = {"type": "json_object"}
        
        success, text, json_data = self._call_llm(system_prompt, user_prompt, response_format)
        
        if not success or not json_data:
            # Default: if round 1, embed first reasonable module; otherwise refine
            if round_num == 1 and remaining_modules:
                # Pick first non-root module if exists, otherwise root
                target = next((m for m in remaining_modules if m != "root"), remaining_modules[0])
                return "EMBED_MODULE", target, "LLM failed, defaulting to embed first module"
            return "REFINE_QUERY", None, "LLM failed, defaulting to refine query"
        
        action = json_data.get("action", "REFINE_QUERY")
        module_name = json_data.get("module_name")
        reasoning = json_data.get("reasoning", "")
        
        # Validate module_name if action is EMBED_MODULE
        if action == "EMBED_MODULE":
            if not module_name or module_name not in remaining_modules:
                # Invalid module, pick a fallback
                logger.warning(f"LLM suggested invalid module '{module_name}', picking fallback")
                module_name = remaining_modules[0]
                reasoning += f" (Corrected to {module_name})"
        
        logger.info(f"RAG action decision: {action}" + (f" - {module_name}" if module_name else ""))
        
        return action, module_name, reasoning
    
    def evaluate_rag_results(
        self,
        error_context: str,
        query: str,
        results: List[Dict[str, Any]],
        round_num: int,
        max_rounds: int
    ) -> Tuple[str, str, Optional[str]]:
        """
        Evaluate RAG search results for usefulness.
        
        Args:
            error_context: Error information
            query: Search query used
            results: RAG search results
            round_num: Current round number
            max_rounds: Maximum rounds
        
        Returns:
            Tuple of (decision, reasoning, extracted_context)
            - decision: "USEFUL", "RETRY", or "NOT_USEFUL"
            - reasoning: Explanation
            - extracted_context: Filtered useful code (if useful)
        """
        logger.info(f"Phase 3: Evaluating RAG results (round {round_num}/{max_rounds})...")
        
        if not results:
            if round_num >= max_rounds:
                return "NOT_USEFUL", "No results after max rounds", None
            return "RETRY", "No results, will refine query", None
        
        system_prompt = """You are evaluating RAG search results to determine if they're useful for fixing a test error.

The results contain actual production code (method bodies, signatures, etc.).

Evaluate:
- Are these code snippets relevant to the error?
- Do they help understand the production code behavior?
- Would they assist in fixing the test?

Respond in JSON:
{
  "decision": "USEFUL" or "RETRY" or "NOT_USEFUL",
  "reasoning": "Explanation",
  "extracted_context": "If USEFUL, extract and explain the relevant code snippets that will help"
}

Be honest: if results are not helpful, say so. This pipeline values precision over false positives."""
        
        # Format top results
        results_summary = []
        for i, r in enumerate(results[:5]):
            results_summary.append({
                'rank': i + 1,
                'class': r.get('class_name', ''),
                'method': r.get('method_name', ''),
                'score': r.get('score', 0),
                'signature': r.get('content', '')[:300]
            })
        
        results_str = json.dumps(results_summary, indent=2)
        
        user_prompt = f"""Error context:
{error_context}

Search query: {query}

Results ({len(results)} found):
{results_str}

Round {round_num} of {max_rounds}.

Are these results useful for fixing the error?"""
        
        response_format = {"type": "json_object"}
        
        success, text, json_data = self._call_llm(system_prompt, user_prompt, response_format)
        
        if not success or not json_data:
            if round_num >= max_rounds:
                return "NOT_USEFUL", "Evaluation failed at max rounds", None
            return "RETRY", "Evaluation failed, will retry", None
        
        decision = json_data.get("decision", "NOT_USEFUL")
        reasoning = json_data.get("reasoning", "")
        extracted = json_data.get("extracted_context")
        
        logger.info(f"RAG evaluation: {decision} - {reasoning}")
        
        return decision, reasoning, extracted

