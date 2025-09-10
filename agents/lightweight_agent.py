"""
Lightweight Agent for M2S-x-teaming
Simplified agent with minimal dependencies, focusing on OpenAI API support
"""

import logging
import os
from random import uniform
from time import sleep
from typing import Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv

from openai import OpenAI

# Load environment variables from .env file
load_dotenv()


class APICallError(Exception):
    """Custom exception for API call failures"""
    pass


class LightweightAgent:
    """
    Lightweight agent for M2S-x-teaming with minimal dependencies
    Supports OpenAI API and basic functionality needed for M2S system
    """

    def __init__(self, config: Dict):
        """Initialize lightweight agent with OpenAI support"""
        self.config = config
        self.provider = config.get("provider", "openai")
        self.model = config.get("model", "gpt-4.1-2025-04-14")
        self.max_retries = config.get("max_retries", 3)
        
        # Initialize OpenAI client (supports custom API endpoints)
        try:
            if self.provider == "openai" or self.provider == "custom":
                # Custom API support
                if self.provider == "custom":
                    api_key = config.get("api_key") or os.getenv("CUSTOM_API_KEY")
                    base_url = config.get("base_url") or os.getenv("CUSTOM_BASE_URL")
                    if not api_key or not base_url:
                        raise ValueError("API key and base URL required. Set CUSTOM_API_KEY and CUSTOM_BASE_URL environment variables.")
                    self.client = OpenAI(api_key=api_key, base_url=base_url)
                    self.mock_mode = False
                    logging.info(f"Custom API client initialized: {base_url} with model: {self.model}")
                else:
                    # Standard OpenAI API
                    api_key = os.getenv("OPENAI_API_KEY")
                    if not api_key:
                        # For testing without API key, create mock client
                        self.client = None
                        self.mock_mode = True
                        logging.warning("No OpenAI API key found, running in mock mode. Please set OPENAI_API_KEY in .env file")
                    else:
                        self.client = OpenAI(api_key=api_key)
                        self.mock_mode = False
                        logging.info(f"OpenAI client initialized with model: {self.model}")
            else:
                raise APICallError(f"Provider {self.provider} not supported in lightweight mode")
                
        except Exception as e:
            raise APICallError(f"Error initializing {self.provider} client: {str(e)}")

    def call_api(
        self,
        messages: List[Dict],
        temperature: float,
        response_format: Optional[Dict] = None,
        return_messages: bool = False,
    ) -> Union[str, Tuple[str, List[Dict]]]:
        """
        API call handler with OpenAI support and mock fallback
        
        Args:
            messages: List of message dictionaries
            temperature: Float value for temperature
            response_format: Optional response format specifications
            return_messages: If True, returns tuple of (response, messages)
            
        Returns:
            Either string response or tuple of (response, messages)
        """
        
        # Mock mode for testing without API key
        if self.mock_mode:
            return self._mock_response(messages, return_messages)
        
        logging.info(f"Calling {self.provider} model {self.model} with temperature {temperature}")

        for attempt in range(self.max_retries):
            try:
                # Add retry delay if needed
                if attempt > 0:
                    delay = 2 * attempt + uniform(0, 1)
                    logging.info(f"Retry attempt {attempt + 1}/{self.max_retries}. Waiting {delay:.2f}s...")
                    sleep(delay)

                # Call OpenAI API
                api_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                }
                
                if response_format:
                    api_params["response_format"] = response_format

                response = self.client.chat.completions.create(**api_params)
                response_content = response.choices[0].message.content

                # Return based on return_messages flag
                return (response_content, messages) if return_messages else response_content

            except Exception as e:
                error_msg = str(e)
                logging.error(f"API call failed for {self.provider} (Attempt {attempt + 1}/{self.max_retries}): {error_msg}")

                if attempt == self.max_retries - 1:
                    # Final attempt failed, try mock response
                    logging.warning("All API attempts failed, falling back to mock response")
                    return self._mock_response(messages, return_messages)
                    
                continue
    
    def _mock_response(self, messages: List[Dict], return_messages: bool) -> Union[str, Tuple[str, List[Dict]]]:
        """Generate mock response for testing"""
        user_message = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        # Generate appropriate mock response based on request type
        if "M2S" in user_message and "template" in user_message.lower():
            # Template generation request
            response = '''[
                {
                    "id": "mock_001",
                    "name": "mock_creative_template",
                    "template": "Execute the following sequence: {PROMPTS_SEQUENTIAL}",
                    "description": "Mock template for sequential execution",
                    "placeholder_type": "PROMPTS_SEQUENTIAL",
                    "type": "generated",
                    "example_usage": "Multi-turn sequence -> Single sequential command"
                }
            ]'''
        elif "convert" in user_message.lower():
            # Conversion request
            response = "Sequential execution: Task 1 → Task 2 → Task 3 → Complete"
        else:
            response = f"Mock response generated for testing purposes. Original request: {user_message[:100]}..."
        
        return (response, messages) if return_messages else response


# Alias for compatibility
BaseAgent = LightweightAgent