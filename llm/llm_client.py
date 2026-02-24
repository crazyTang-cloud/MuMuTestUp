import requests
import json
import time
from typing import Optional, Dict, Any, Callable
from utils import logger, get_sample_logger
from config import config

class LLMClient:
    """Client for communicating with LLM (supports Ollama and OpenAI-compatible APIs)"""
    
    def __init__(self, model: Optional[str] = None, agent_name: str = "Unknown", 
                 max_retries: Optional[int] = None, retry_delay: Optional[float] = None):
        """
        Initialize LLM client
        
        Args:
            model: Model name to use (overrides config)
            agent_name: Name of the agent using this client
            max_retries: Maximum number of retries (None uses config.llm.max_retries)
            retry_delay: Initial retry delay in seconds (None uses config.llm.retry_delay)
        """
        # Use the new unified LLM config
        self.llm_config = config.llm
        self.model = model or self.llm_config.model
        self.provider = self.llm_config.provider
        self.base_url = self.llm_config.get_base_url()
        self.api_key = self.llm_config.api_key
        self.temperature = self.llm_config.temperature
        self.logger = logger
        self.agent_name = agent_name  # Name of the agent using this client
        
        # Retry configuration - use LLM config values as defaults
        self.max_retries = max_retries if max_retries is not None else self.llm_config.max_retries
        self.retry_delay = retry_delay if retry_delay is not None else self.llm_config.retry_delay
        
    def _retry_with_backoff(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with exponential backoff retry strategy
        
        Args:
            func: Function to execute
            *args, **kwargs: Arguments to pass to the function
            
        Returns:
            Result from the function
            
        Raises:
            Exception: If all retries fail
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except requests.exceptions.Timeout as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    self.logger.warning(
                        f"Request timeout (attempt {attempt + 1}/{self.max_retries + 1}), "
                        f"retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(f"Request failed after {self.max_retries + 1} attempts (timeout)")
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Connection error (attempt {attempt + 1}/{self.max_retries + 1}), "
                        f"retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(f"Request failed after {self.max_retries + 1} attempts (connection error)")
            except requests.exceptions.HTTPError as e:
                last_exception = e
                # Check if it's a rate limit error (429) or server error (5xx)
                if hasattr(e, 'response') and e.response is not None:
                    status_code = e.response.status_code
                    
                    # Retry on rate limit (429) or server errors (500-599)
                    if status_code == 429 or (500 <= status_code < 600):
                        if attempt < self.max_retries:
                            # For rate limits, use longer delay
                            delay = self.retry_delay * (2 ** attempt)
                            if status_code == 429:
                                delay *= 2  # Double the delay for rate limits
                            
                            self.logger.warning(
                                f"HTTP {status_code} error (attempt {attempt + 1}/{self.max_retries + 1}), "
                                f"retrying in {delay:.1f}s..."
                            )
                            time.sleep(delay)
                        else:
                            self.logger.error(
                                f"Request failed after {self.max_retries + 1} attempts (HTTP {status_code})"
                            )
                    else:
                        # Don't retry on client errors (4xx except 429)
                        self.logger.error(f"HTTP {status_code} error, not retrying")
                        raise
                else:
                    raise
            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.max_retries:
                    delay = self.retry_delay * (2 ** attempt)
                    self.logger.warning(
                        f"Request error (attempt {attempt + 1}/{self.max_retries + 1}), "
                        f"retrying in {delay:.1f}s..."
                    )
                    time.sleep(delay)
                else:
                    self.logger.error(f"Request failed after {self.max_retries + 1} attempts")
        
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise Exception("Request failed for unknown reason")
    
    def _make_request_ollama(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to Ollama"""
        url = f"{self.base_url}{endpoint}"
        
        def _request():
            response = requests.post(
                url, 
                json=payload, 
                timeout=self.llm_config.timeout
            )
            response.raise_for_status()
            return response.json()
        
        try:
            return self._retry_with_backoff(_request)
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ollama request failed: {e}")
            raise
    
    def _make_request_openai(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make HTTP request to OpenAI-compatible API"""
        # Use custom API URL if provided, otherwise construct from base_url
        url = self.llm_config.api_url or f"{self.base_url}/v1/chat/completions"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add API key if provided
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        def _request():
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=self.llm_config.timeout
            )
            response.raise_for_status()
            return response.json()
        
        try:
            return self._retry_with_backoff(_request)
        except requests.exceptions.RequestException as e:
            self.logger.error(f"OpenAI API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    self.logger.error(f"Response: {e.response.text}")
                except:
                    pass
            raise
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None,
                 prompt_type: str = "general") -> str:
        """
        Generate response from LLM
        
        Args:
            prompt: User prompt
            system_prompt: System prompt for context
            prompt_type: Type of prompt for logging purposes
            
        Returns:
            Generated response
        """
        messages = []
        
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Log prompt to sample logger
        sample_logger = get_sample_logger()
        if sample_logger.log_file:
            sample_logger.log_prompt(
                agent_name=self.agent_name,
                prompt_type=prompt_type,
                system_prompt=system_prompt or "",
                user_prompt=prompt
            )
        
        try:
            if self.provider == "ollama":
                # Ollama API format
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "temperature": self.temperature
                }
                response = self._make_request_ollama("/api/chat", payload)
                response_text = response.get("message", {}).get("content", "")
            
            elif self.provider == "openai":
                # OpenAI API format
                payload = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature
                }
                response = self._make_request_openai(payload)
                response_text = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
            
            # Log response to sample logger
            if sample_logger.log_file:
                sample_logger.log_response(
                    agent_name=self.agent_name,
                    response=response_text
                )
            
            return response_text
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            
            # Log error to sample logger
            if sample_logger.log_file:
                sample_logger.log_error(f"LLM generation failed: {e}")
            
            return ""
    
    def generate_json(self, prompt: str, system_prompt: Optional[str] = None,
                      prompt_type: str = "json") -> Dict[str, Any]:
        """
        Generate JSON response from LLM
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            prompt_type: Type of prompt for logging purposes
            
        Returns:
            Parsed JSON response
        """
        response_text = self.generate(prompt, system_prompt, prompt_type)
        
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                
                # Log parsed result
                sample_logger = get_sample_logger()
                if sample_logger.log_file:
                    sample_logger.log_custom(
                        category="parsed_json",
                        data=parsed,
                        description="Parsed JSON from LLM response"
                    )
                
                return parsed
            else:
                self.logger.warning(f"No JSON found in response: {response_text}")
                return {}
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {e}")
            return {}
    
    def change_model(self, model: str):
        """Change the model to use"""
        self.model = model
        self.logger.info(f"Switched to model: {model}")
    
    def set_agent_name(self, agent_name: str):
        """Set the agent name for logging purposes"""
        self.agent_name = agent_name
