"""
AI Services Module for Text Correction in MinerU2PPT

Supports multiple AI providers:
- OpenAI (GPT models)
- Google Gemini
- Anthropic Claude
- Groq

Primary communication through JSON for structured text correction.
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import asyncio
import concurrent.futures
from dataclasses import dataclass
# Set up logging
logger = logging.getLogger(__name__)

# Configure file logging for debugging in frozen app
try:
    import sys
    import os
    if getattr(sys, 'frozen', False):
        log_dir = os.path.dirname(sys.executable)
    else:
        log_dir = os.path.dirname(os.path.abspath(__file__))
        
    log_file = os.path.join(log_dir, 'ai_debug.log')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.DEBUG)
    logger.info("AI Services logging initialized")
except Exception as e:
    print(f"Failed to setup file logging: {e}")

# AI Client imports (will be installed via requirements.txt)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.genai as genai
    GOOGLE_AVAILABLE = True
    GOOGLE_PACKAGE = "google.genai"
except ImportError:
    GOOGLE_AVAILABLE = False
    GOOGLE_PACKAGE = None
    logger.error("Google GenAI package not available. Install with: pip install google-genai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TextCorrectionRequest:
    """Request structure for text correction"""
    text: str
    context: Optional[str] = None
    language: str = "auto"
    correction_type: str = "grammar_spelling"  # grammar_spelling, formatting, both


@dataclass
class BatchTextCorrectionRequest:
    """Request structure for batch text correction"""
    texts: List[str]
    context: Optional[str] = None
    language: str = "auto"
    correction_type: str = "grammar_spelling"
    page_number: Optional[int] = None
    section_name: Optional[str] = None


@dataclass
class TextCorrectionResponse:
    """Response structure for text correction"""
    original_text: str
    corrected_text: str
    changes: List[Dict[str, Any]]
    confidence: float
    provider: str
    model: str


@dataclass
class BatchTextCorrectionResponse:
    """Response structure for batch text correction"""
    original_texts: List[str]
    corrected_texts: List[str]
    changes: List[List[Dict[str, Any]]]  # Changes for each text
    confidence: float
    provider: str
    model: str
    page_number: Optional[int] = None
    section_name: Optional[str] = None


class AIServiceBase(ABC):
    """Abstract base class for AI text correction services"""
    
    def __init__(self, api_key: str, model_name: str = None):
        self.api_key = api_key
        self.model_name = model_name
        self.is_authenticated = False
        self.available_models = []  # Will be populated after authentication
    
    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the AI service"""
        pass
    
    @abstractmethod
    async def correct_text(self, request: TextCorrectionRequest) -> TextCorrectionResponse:
        """Correct text using the AI service"""
        pass
    
    @abstractmethod
    async def correct_text_batch(self, request: BatchTextCorrectionRequest) -> BatchTextCorrectionResponse:
        """Correct multiple texts in a single request"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models for this service"""
        pass
    
    @abstractmethod
    async def fetch_available_models(self) -> List[str]:
        """Fetch available models from the API after authentication"""
        pass
    
    def get_default_models(self) -> List[str]:
        """Get default fallback models if API fetch fails"""
        return self.get_available_models()
    
    @abstractmethod
    def get_auth_url(self) -> str:
        """Get authentication URL for getting API keys/tokens"""
        pass
    
    @abstractmethod
    def get_auth_instructions(self) -> str:
        """Get detailed authentication instructions"""
        pass
    
    def get_provider_name(self) -> str:
        """Get human-readable provider name"""
        return self.__class__.__name__.replace("Service", "")
    
    def calculate_optimal_batch_size(self, texts: List[str]) -> int:
        """Calculate optimal batch size based on text amount"""
        total_chars = sum(len(text) for text in texts)
        
        # Optimal batch sizes based on total character count
        if total_chars < 1000:  # Small amount - process all at once
            return len(texts)
        elif total_chars < 5000:  # Medium amount - batch by 5-10 texts
            return min(10, len(texts))
        elif total_chars < 15000:  # Large amount - batch by 3-5 texts
            return min(5, len(texts))
        else:  # Very large amount - batch by 2-3 texts
            return min(3, len(texts))
    
    def create_batches(self, texts: List[str], context: str = None, correction_type: str = "grammar_spelling", 
                      page_number: int = None, section_name: str = None) -> List[BatchTextCorrectionRequest]:
        """Create optimal batches from a list of texts"""
        if not texts:
            return []
        
        batch_size = self.calculate_optimal_batch_size(texts)
        batches = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batches.append(BatchTextCorrectionRequest(
                texts=batch_texts,
                context=context,
                correction_type=correction_type,
                page_number=page_number,
                section_name=section_name
            ))
        
        return batches
    
    def get_json_system_prompt(self, correction_type: str, batch_mode: bool = False) -> str:
        """Get JSON-formatted system prompt for text correction"""
        if batch_mode:
            return self._get_batch_json_prompt(correction_type)
        else:
            return self._get_single_json_prompt(correction_type)
    
    def _get_single_json_prompt(self, correction_type: str) -> str:
        """Get JSON system prompt for single text correction"""
        prompts = {
            "grammar_spelling": """You are a text correction assistant. Correct grammar and spelling errors while preserving meaning and structure.

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{
    "corrected_text": "The corrected text here",
    "changes": [
        {
            "type": "spelling|grammar|punctuation",
            "original": "original text",
            "corrected": "corrected text",
            "position": {"start": 0, "end": 5},
            "reason": "explanation for the change"
        }
    ],
    "confidence": 0.95
}

Only correct actual errors. Do not change style, formatting, or make unnecessary modifications.""",
            
            "formatting": """You are a text formatting assistant. Improve text formatting while preserving content and meaning.

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{
    "corrected_text": "The formatted text here",
    "changes": [
        {
            "type": "formatting|structure|spacing",
            "original": "original text",
            "corrected": "formatted text",
            "position": {"start": 0, "end": 5},
            "reason": "explanation for the change"
        }
    ],
    "confidence": 0.95
}

Focus on improving readability and consistency.""",
            
            "both": """You are a comprehensive text correction assistant. Correct grammar, spelling, and improve formatting while preserving meaning.

IMPORTANT: Respond ONLY with valid JSON in this exact format:
{
    "corrected_text": "The corrected and formatted text here",
    "changes": [
        {
            "type": "spelling|grammar|punctuation|formatting|structure",
            "original": "original text",
            "corrected": "corrected text",
            "position": {"start": 0, "end": 5},
            "reason": "explanation for the change"
        }
    ],
    "confidence": 0.95
}

Make necessary corrections and formatting improvements."""
        }
        return prompts.get(correction_type, prompts["grammar_spelling"])
    
    def _get_batch_json_prompt(self, correction_type: str) -> str:
        """Get JSON system prompt for batch text correction"""
        prompts = {
            "grammar_spelling": """You are a text correction assistant. You will receive a JSON array of texts and must return corrections in JSON format.

INPUT FORMAT: You will receive a JSON object with an array of texts:
{
    "texts": ["text1", "text2", "text3", ...]
}

OUTPUT FORMAT: Respond ONLY with valid JSON in this EXACT format:
{
    "corrected_texts": [
        "First corrected text here",
        "Second corrected text here", 
        "Third corrected text here"
    ],
    "changes": [
        [
            {
                "type": "spelling|grammar|punctuation",
                "original": "original text",
                "corrected": "corrected text",
                "position": {"start": 0, "end": 5},
                "reason": "explanation for the change"
            }
        ],
        [],
        [
            {
                "type": "grammar",
                "original": "original",
                "corrected": "corrected",
                "position": {"start": 0, "end": 8},
                "reason": "grammar fix"
            }
        ]
    ],
    "confidence": 0.95
}

CRITICAL RULES:
- corrected_texts array MUST have the SAME number of elements as the input texts array
- changes array MUST have the SAME number of elements as the input texts array  
- If a text has no changes, include an empty array [] in the changes
- Only correct actual grammar and spelling errors, preserve meaning and structure""",
            
            "formatting": """You are a text formatting assistant. You will receive a JSON array of texts and must return formatted versions in JSON format.

INPUT FORMAT: You will receive a JSON object with an array of texts:
{
    "texts": ["text1", "text2", "text3", ...]
}

OUTPUT FORMAT: Respond ONLY with valid JSON in this EXACT format:
{
    "corrected_texts": [
        "First formatted text here",
        "Second formatted text here",
        "Third formatted text here"
    ],
    "changes": [
        [
            {
                "type": "formatting|structure|spacing",
                "original": "original text",
                "corrected": "formatted text",
                "position": {"start": 0, "end": 5},
                "reason": "explanation for the change"
            }
        ],
        [],
        [
            {
                "type": "formatting", 
                "original": "original",
                "corrected": "formatted",
                "position": {"start": 0, "end": 8},
                "reason": "formatting improvement"
            }
        ]
    ],
    "confidence": 0.95
}

CRITICAL RULES:
- corrected_texts array MUST have the SAME number of elements as the input texts array
- changes array MUST have the SAME number of elements as the input texts array
- If a text has no changes, include an empty array [] in the changes
- Focus on improving readability and consistency while preserving content""",
            
            "both": """You are a comprehensive text correction assistant. You will receive a JSON array of texts and must return corrected/formatted versions in JSON format.

INPUT FORMAT: You will receive a JSON object with an array of texts:
{
    "texts": ["text1", "text2", "text3", ...]
}

OUTPUT FORMAT: Respond ONLY with valid JSON in this EXACT format:
{
    "corrected_texts": [
        "First corrected and formatted text here",
        "Second corrected and formatted text here",
        "Third corrected and formatted text here"
    ],
    "changes": [
        [
            {
                "type": "spelling|grammar|punctuation|formatting|structure",
                "original": "original text",
                "corrected": "corrected text",
                "position": {"start": 0, "end": 5},
                "reason": "explanation for the change"
            }
        ],
        [],
        [
            {
                "type": "grammar",
                "original": "original",
                "corrected": "corrected", 
                "position": {"start": 0, "end": 8},
                "reason": "grammar and formatting fix"
            }
        ]
    ],
    "confidence": 0.95
}

CRITICAL RULES:
- corrected_texts array MUST have the SAME number of elements as the input texts array
- changes array MUST have the SAME number of elements as the input texts array
- If a text has no changes, include an empty array [] in the changes
- Correct grammar, spelling and improve formatting while preserving meaning"""
        }
        return prompts.get(correction_type, prompts["grammar_spelling"])
    
    def parse_json_response(self, content: str, is_batch: bool = False, expected_count: int = 1) -> Dict[str, Any]:
        """Parse JSON response from AI provider with improved truncation handling"""
        # Ensure content is properly decoded (handle encoding issues)
        if isinstance(content, bytes):
            try:
                content = content.decode('utf-8')
            except UnicodeDecodeError:
                content = content.decode('utf-8', errors='replace')
        
        content = content.strip()
        
        try:
            # Try to parse as JSON first
            parsed = json.loads(content)
            
            # Validate batch response structure
            if is_batch and "corrected_texts" in parsed:
                if not isinstance(parsed["corrected_texts"], list):
                    logger.error(f"corrected_texts is not a list: {type(parsed['corrected_texts'])}")
                    raise json.JSONDecodeError("Invalid corrected_texts format", content, 0)
                    
                # Check if we have the expected number of texts
                if len(parsed["corrected_texts"]) != expected_count:
                    logger.error(f"Expected {expected_count} corrected texts, got {len(parsed['corrected_texts'])}")
                    # Don't raise error here, let the caller handle it
                    
            return parsed
            
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parsing failed at position {e.pos}: {e.msg}")
            logger.debug(f"Content preview: {content[:300]}...")
            
            # Try to repair truncated JSON
            repaired_content = self._repair_truncated_json(content)
            if repaired_content != content:
                try:
                    parsed = json.loads(repaired_content)
                    logger.info("Successfully repaired truncated JSON")
                    return parsed
                except json.JSONDecodeError:
                    logger.debug("JSON repair failed, trying extraction")
            
            # Extract partial data from broken JSON
            return self._extract_partial_json_data(content, is_batch, expected_count)
    
    def _repair_truncated_json(self, content: str) -> str:
        """Attempt to repair truncated JSON by adding missing closing elements"""
        # Count unclosed braces and brackets
        open_braces = content.count('{')
        close_braces = content.count('}')
        open_brackets = content.count('[')
        close_brackets = content.count(']')
        
        # Count quotes to detect unterminated strings
        quote_count = content.count('"')
        
        repaired = content
        
        # If we're inside a string (odd number of quotes), close it
        if quote_count % 2 != 0:
            repaired += '"'
            logger.debug("Added missing closing quote")
        
        # Close missing brackets first (arrays before objects)
        missing_brackets = open_brackets - close_brackets
        for _ in range(missing_brackets):
            repaired += ']'
            logger.debug("Added missing closing bracket")
        
        # Close missing braces
        missing_braces = open_braces - close_braces
        for _ in range(missing_braces):
            repaired += '}'
            logger.debug("Added missing closing brace")
        
        return repaired
    
    def _extract_partial_json_data(self, content: str, is_batch: bool = False, expected_count: int = 1) -> Dict[str, Any]:
        """Extract partial data from broken JSON using regex patterns"""
        import re
        
        logger.info("Extracting partial data from broken JSON")
        
        if is_batch:
            # For batch responses, try to extract corrected_texts array
            result = {
                "corrected_texts": [],
                "changes": [],
                "confidence": 0.5
            }
            
            # Try to extract texts from array pattern
            array_match = re.search(r'"corrected_texts"\s*:\s*\[(.*?)\]', content, re.DOTALL)
            if array_match:
                array_content = array_match.group(1)
                # Extract individual quoted strings
                texts = re.findall(r'"([^"]*)"', array_content)
                result["corrected_texts"] = texts[:expected_count]
            
            # If we didn't get enough texts, fill with fallbacks
            while len(result["corrected_texts"]) < expected_count:
                result["corrected_texts"].append("Text extraction failed")
            
            # Create empty changes arrays
            result["changes"] = [[] for _ in result["corrected_texts"]]
            
            return result
        
        else:
            # For single responses, extract individual fields
            result = {
                "corrected_text": "",
                "changes": [],
                "confidence": 0.5
            }
            
            # Extract corrected_text
            text_patterns = [
                r'"corrected_text"\s*:\s*"([^"]*)"',  # Standard quoted text
                r'"corrected_text"\s*:\s*"([^"]*)',   # Unclosed quoted text
            ]
            
            for pattern in text_patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    result["corrected_text"] = match.group(1)
                    logger.debug(f"Extracted corrected_text: {result['corrected_text'][:50]}...")
                    break
            
            # Extract confidence
            conf_match = re.search(r'"confidence"\s*:\s*([\d.]+)', content)
            if conf_match:
                try:
                    result["confidence"] = float(conf_match.group(1))
                except ValueError:
                    pass
            
            # Extract changes array (simplified)
            changes_match = re.search(r'"changes"\s*:\s*\[(.*)', content, re.DOTALL)
            if changes_match:
                changes_content = changes_match.group(1)
                
                # Try to extract at least one change object
                change_patterns = [
                    r'"type"\s*:\s*"([^"]*)"',
                    r'"original"\s*:\s*"([^"]*)"', 
                    r'"corrected"\s*:\s*"([^"]*)"',
                    r'"reason"\s*:\s*"([^"]*)'  # Note: may be unclosed
                ]
                
                change_data = {}
                for i, pattern in enumerate(change_patterns):
                    match = re.search(pattern, changes_content)
                    if match:
                        field_names = ["type", "original", "corrected", "reason"]
                        change_data[field_names[i]] = match.group(1)
                
                if len(change_data) >= 2:  # At least type and one other field
                    change_obj = {
                        "type": change_data.get("type", "unknown"),
                        "original": change_data.get("original", ""),
                        "corrected": change_data.get("corrected", ""),
                        "position": {"start": 0, "end": 0},
                        "reason": change_data.get("reason", "Partial extraction from truncated JSON")
                    }
                    result["changes"] = [change_obj]
                    logger.debug(f"Extracted change: {change_obj['original']} -> {change_obj['corrected']}")
            
            # If we couldn't extract corrected_text, use fallback
            if not result["corrected_text"]:
                # Try to find any quoted text that might be the corrected version
                all_quotes = re.findall(r'"([^"]*)"', content)
                if all_quotes:
                    # Use the longest quoted text as likely corrected text
                    result["corrected_text"] = max(all_quotes, key=len)
                    logger.debug(f"Using longest quoted text as fallback: {result['corrected_text'][:50]}...")
                else:
                    result["corrected_text"] = "JSON extraction failed"
            
            logger.info(f"Partial extraction complete: {len(result['changes'])} changes, confidence {result['confidence']}")
            return result
    
    async def correct_texts_by_page(self, texts: List[str], page_number: int, 
                                   context: str = None, correction_type: str = "grammar_spelling") -> List[str]:
        """Correct all texts for a specific page efficiently"""
        if not texts:
            return []
        
        # Create batches optimized for the amount of text
        batches = self.create_batches(texts, context, correction_type, page_number)
        corrected_texts = []
        
        for batch in batches:
            try:
                batch_response = await self.correct_text_batch(batch)
                corrected_texts.extend(batch_response.corrected_texts)
            except Exception as e:
                logger.error(f"Batch correction failed: {e}")
                # Fallback to individual correction
                for text in batch.texts:
                    try:
                        individual_request = TextCorrectionRequest(
                            text=text,
                            context=context,
                            correction_type=correction_type
                        )
                        individual_response = await self.correct_text(individual_request)
                        corrected_texts.append(individual_response.corrected_text)
                    except Exception as individual_error:
                        logger.error(f"Individual correction failed: {individual_error}")
                        corrected_texts.append(text)  # Return original text
        
        return corrected_texts
    


class OpenAIService(AIServiceBase):
    """OpenAI GPT-based text correction service"""
    
    def __init__(self, api_key: str, model_name: str = None):
        super().__init__(api_key, model_name)
        self.client = None
    
    async def authenticate(self) -> bool:
        """Authenticate with OpenAI API"""
        try:
            if not OPENAI_AVAILABLE:
                logger.error("OpenAI library not available")
                return False
            
            self.client = openai.OpenAI(api_key=self.api_key)
            # Test authentication by listing models
            models_response = await asyncio.get_event_loop().run_in_executor(
                None, 
                lambda: self.client.models.list()
            )
            
            # Fetch available models during authentication
            self.available_models = await self.fetch_available_models_internal(models_response)
            
            # Set default model if none specified
            if not self.model_name and self.available_models:
                self.model_name = self.available_models[0]
            
            self.is_authenticated = True
            logger.info(f"OpenAI authentication successful. Found {len(self.available_models)} models")
            return True
        except Exception as e:
            logger.error(f"OpenAI authentication failed: {e}")
            return False
    
    async def correct_text(self, request: TextCorrectionRequest) -> TextCorrectionResponse:
        """Correct text using OpenAI"""
        if not self.is_authenticated:
            if not await self.authenticate():
                raise Exception("OpenAI authentication failed")
        
        try:
            system_prompt = self.get_json_system_prompt(request.correction_type)
            user_message = f"Please correct the following text:\n\n{request.text}"
            
            if request.context:
                user_message += f"\n\nContext: {request.context}"
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.1,
                    max_tokens=4000,
                    response_format={"type": "json_object"}  # Force JSON response
                )
            )
            
            content = response.choices[0].message.content.strip()
            result_json = self.parse_json_response(content, is_batch=False)
            
            return TextCorrectionResponse(
                original_text=request.text,
                corrected_text=result_json.get("corrected_text", request.text),
                changes=result_json.get("changes", []),
                confidence=result_json.get("confidence", 0.8),
                provider="OpenAI",
                model=self.model_name
            )
            
        except Exception as e:
            logger.error(f"OpenAI text correction failed: {e}")
            raise
    
    async def correct_text_batch(self, request: BatchTextCorrectionRequest) -> BatchTextCorrectionResponse:
        """Correct multiple texts using OpenAI in a single request"""
        if not self.is_authenticated:
            if not await self.authenticate():
                raise Exception("OpenAI authentication failed")
        
        try:
            system_prompt = self.get_json_system_prompt(request.correction_type, batch_mode=True)
            
            # Create JSON input for the AI
            input_json = {
                "texts": request.texts
            }
            
            # Add context information if provided
            if request.context:
                input_json["context"] = request.context
            if request.page_number:
                input_json["page_number"] = request.page_number
            if request.section_name:
                input_json["section_name"] = request.section_name
            
            user_message = f"Please process this JSON input and return the corrected texts:\n\n{json.dumps(input_json, indent=2, ensure_ascii=False)}"
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=0.1,
                    max_tokens=8000,
                    response_format={"type": "json_object"}  # Force JSON response
                )
            )
            
            content = response.choices[0].message.content.strip()
            result_json = self.parse_json_response(content, is_batch=True, expected_count=len(request.texts))
            
            # Strict validation of response format
            corrected_texts = result_json.get("corrected_texts", [])
            if not isinstance(corrected_texts, list):
                logger.error(f"OpenAI batch correction: corrected_texts is not a list, got {type(corrected_texts)}")
                corrected_texts = request.texts  # Fallback to original texts
            elif len(corrected_texts) != len(request.texts):
                logger.error(f"OpenAI batch correction returned {len(corrected_texts)} texts, expected {len(request.texts)}")
                logger.error(f"Input texts ({len(request.texts)}): {[text[:50] + '...' if len(text) > 50 else text for text in request.texts]}")
                logger.error(f"Output texts ({len(corrected_texts)}): {[text[:50] + '...' if len(text) > 50 else text for text in corrected_texts]}")
                logger.error(f"AI Response content (first 500 chars): {content[:500]}...")
                
                # Strict padding/trimming with original texts as fallback
                if len(corrected_texts) < len(request.texts):
                    # Pad missing texts with originals
                    for i in range(len(corrected_texts), len(request.texts)):
                        corrected_texts.append(request.texts[i])
                        logger.warning(f"Using original text for missing output {i+1}: '{request.texts[i][:50]}...'")
                elif len(corrected_texts) > len(request.texts):
                    # Trim excess texts
                    logger.warning(f"Trimming {len(corrected_texts) - len(request.texts)} excess texts from output")
                    corrected_texts = corrected_texts[:len(request.texts)]
            
            changes = result_json.get("changes", [])
            if not isinstance(changes, list) or len(changes) != len(request.texts):
                logger.warning(f"OpenAI batch correction: invalid changes format, creating empty changes list")
                changes = [[] for _ in request.texts]
            
            return BatchTextCorrectionResponse(
                original_texts=request.texts,
                corrected_texts=corrected_texts,
                changes=changes,
                confidence=result_json.get("confidence", 0.8),
                provider="OpenAI",
                model=self.model_name,
                page_number=request.page_number,
                section_name=request.section_name
            )
            
        except Exception as e:
            logger.error(f"OpenAI batch text correction failed: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get available OpenAI models"""
        return self.available_models if self.available_models else ["gpt-3.5-turbo"]
    
    async def fetch_available_models_internal(self, models_response) -> List[str]:
        """Internal method to process models response"""
        try:
            # Filter for text generation models that are suitable for our use case
            suitable_models = []
            for model in models_response.data:
                model_id = model.id
                # Include GPT models and other text generation models
                if any(prefix in model_id for prefix in ['gpt-', 'text-', 'o1-']):
                    # Prioritize newer/better models
                    priority_models = ['gpt-4o', 'gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo']
                    if any(good_model in model_id for good_model in priority_models):
                        # Insert based on priority
                        for i, priority in enumerate(priority_models):
                            if priority in model_id:
                                suitable_models.insert(i, model_id)
                                break
                        else:
                            suitable_models.append(model_id)
                    else:
                        suitable_models.append(model_id)
            
            return suitable_models if suitable_models else ["gpt-3.5-turbo"]  # Fallback
                
        except Exception as e:
            logger.error(f"Failed to process OpenAI models: {e}")
            return ["gpt-3.5-turbo"]  # Fallback

    async def fetch_available_models(self) -> List[str]:
        """Fetch available models from OpenAI API"""
        if not self.is_authenticated:
            logger.warning("Not authenticated, cannot fetch models")
            return []
        
        if self.available_models:
            return self.available_models
        
        try:
            # Fetch models from OpenAI API
            models_response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.models.list()
            )
            
            self.available_models = await self.fetch_available_models_internal(models_response)
            return self.available_models
                
        except Exception as e:
            logger.error(f"Failed to fetch OpenAI models: {e}")
            return []
    
    def get_auth_url(self) -> str:
        """Get OpenAI API key signup URL"""
        return "https://platform.openai.com/api-keys"
    
    def get_auth_instructions(self) -> str:
        """Get OpenAI authentication instructions"""
        return """To use OpenAI services:

1. Visit: https://platform.openai.com/api-keys
2. Sign in to your OpenAI account (or create one)
3. Click "Create new secret key"
4. Copy the generated API key (starts with 'sk-...')
5. Paste the API key in the configuration

Note: You need to have credits in your OpenAI account to use the API.
Free tier users get limited credits. Check your usage at:
https://platform.openai.com/usage"""


class GoogleGeminiService(AIServiceBase):
    """Google Gemini text correction service"""
    
    def __init__(self, api_key: str, model_name: str = None):
        super().__init__(api_key, model_name)
        self.client = None
    
    async def authenticate(self) -> bool:
        """Authenticate with Google Gemini using the new google.genai package.

        IMPORTANT:
        - In early versions we tried to *discover* models via a direct REST
          call (using requests to hit the public models endpoint).
        - That REST path turned out to be brittle in frozen executables
          (PyInstaller) even when it worked fine in a normal venv, causing
          "Model fetch failed" in the GUI.
        - We now avoid the REST discovery entirely and rely on a curated list
          of known-good text models. This makes behaviour consistent between
          source and standalone EXE while still testing the real API key via a
          generate_content call.
        """
        try:
            if not GOOGLE_AVAILABLE:
                logger.error("Google GenAI library not available. Install with: pip install google-genai")
                return False
            
            # Create the new Google GenAI client
            self.client = genai.Client(api_key=self.api_key)
            
            # Static list of recommended *text* models (kept in priority order).
            # We prefer the newer 3.x / 2.5 models you actually have access to,
            # and fall back to the stable 2.0/1.5 families if needed.
            # NOTE: we intentionally skip image/tts/robotics/specialized variants.
            self.available_models = [
                # Latest general-purpose previews
                "gemini-3-pro-preview",
                "gemini-3-flash-preview",
                # Current 2.5 production models
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                # Stable aliases
                "gemini-pro-latest",
                "gemini-flash-latest",
                # Older but widely available fallbacks
                "gemini-2.0-flash",
                "gemini-2.0-pro",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
            ]
            logger.info(f"Using static Gemini model list: {self.available_models}")
            
            # Set model name: use configured model if it's in available list, otherwise use first available
            if self.model_name and self.model_name in self.available_models:
                logger.info(f"Using configured model: {self.model_name}")
            else:
                if self.model_name:
                    logger.warning(f"Configured model '{self.model_name}' not available, switching to: {self.available_models[0]}")
                else:
                    logger.info(f"No model configured, using first available: {self.available_models[0]}")
                self.model_name = self.available_models[0]
            
            # Test authentication with the selected model
            logger.info(f"Testing authentication with model: {self.model_name}")
            try:
                await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.client.models.generate_content(
                        model=self.model_name,
                        contents="Hi"
                    ).text
                )
                logger.info(f"Authentication successful with model: {self.model_name}")
            except Exception as e:
                # If the first model fails, try others from the available list
                logger.warning(f"Authentication failed with {self.model_name}: {e}")
                
                for alternative_model in self.available_models[1:]:  # Skip the first one we already tried
                    try:
                        logger.info(f"Trying alternative model: {alternative_model}")
                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda m=alternative_model: self.client.models.generate_content(
                                model=m,
                                contents="Hi"
                            ).text
                        )
                        self.model_name = alternative_model
                        logger.info(f"Authentication successful with alternative model: {alternative_model}")
                        break
                    except Exception as alt_e:
                        logger.warning(f"Alternative model {alternative_model} also failed: {alt_e}")
                        continue
                else:
                    raise Exception(f"Authentication failed with all discovered models. Last error: {e}")
            
            self.is_authenticated = True
            logger.info(f"Google Gemini authentication successful. Found {len(self.available_models)} models")
            return True
        except Exception as e:
            logger.error(f"Google Gemini authentication failed: {e}")
            return False
    
    async def correct_text(self, request: TextCorrectionRequest) -> TextCorrectionResponse:
        """Correct text using Google Gemini"""
        if not self.is_authenticated:
            if not await self.authenticate():
                raise Exception("Google Gemini authentication failed")
        
        try:
            system_prompt = self.get_json_system_prompt(request.correction_type)
            user_message = f"{system_prompt}\n\nPlease correct the following text:\n\n{request.text}"
            
            if request.context:
                user_message += f"\n\nContext: {request.context}"
            
            # Configure generation for JSON response
            generation_config = {
                "temperature": 0.1,
                "max_output_tokens": 4000,
                "response_mime_type": "application/json"
            }
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_message,
                    config=generation_config
                )
            )
            
            content = response.text.strip()
            result_json = self.parse_json_response(content, is_batch=False)
            
            return TextCorrectionResponse(
                original_text=request.text,
                corrected_text=result_json.get("corrected_text", request.text),
                changes=result_json.get("changes", []),
                confidence=result_json.get("confidence", 0.8),
                provider="Google Gemini",
                model=self.model_name
            )
            
        except Exception as e:
            logger.error(f"Google Gemini text correction failed: {e}")
            raise
    
    async def correct_text_batch(self, request: BatchTextCorrectionRequest) -> BatchTextCorrectionResponse:
        """Correct multiple texts using Google Gemini in a single request"""
        if not self.is_authenticated:
            if not await self.authenticate():
                raise Exception("Google Gemini authentication failed")
        
        try:
            system_prompt = self.get_json_system_prompt(request.correction_type, batch_mode=True)
            
            # Create JSON input for the AI
            input_json = {
                "texts": request.texts
            }
            
            # Add context information if provided
            if request.context:
                input_json["context"] = request.context
            if request.page_number:
                input_json["page_number"] = request.page_number
            if request.section_name:
                input_json["section_name"] = request.section_name
            
            user_message = f"{system_prompt}\n\nPlease process this JSON input and return the corrected texts:\n\n{json.dumps(input_json, indent=2, ensure_ascii=False)}"
            
            # Configure generation for JSON response
            generation_config = {
                "temperature": 0.1,
                "max_output_tokens": 8000,
                "response_mime_type": "application/json"
            }
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.models.generate_content(
                    model=self.model_name,
                    contents=user_message,
                    config=generation_config
                )
            )
            
            content = response.text.strip()
            result_json = self.parse_json_response(content, is_batch=True, expected_count=len(request.texts))
            
            # Strict validation of response format
            corrected_texts = result_json.get("corrected_texts", [])
            if not isinstance(corrected_texts, list):
                logger.error(f"Gemini batch correction: corrected_texts is not a list, got {type(corrected_texts)}")
                corrected_texts = request.texts  # Fallback to original texts
            elif len(corrected_texts) != len(request.texts):
                logger.error(f"Gemini batch correction returned {len(corrected_texts)} texts, expected {len(request.texts)}")
                logger.error(f"Input texts ({len(request.texts)}): {[text[:50] + '...' if len(text) > 50 else text for text in request.texts]}")
                logger.error(f"Output texts ({len(corrected_texts)}): {[text[:50] + '...' if len(text) > 50 else text for text in corrected_texts]}")
                logger.error(f"AI Response content (first 500 chars): {content[:500]}...")
                
                # Strict padding/trimming with original texts as fallback
                if len(corrected_texts) < len(request.texts):
                    # Pad missing texts with originals
                    for i in range(len(corrected_texts), len(request.texts)):
                        corrected_texts.append(request.texts[i])
                        logger.warning(f"Using original text for missing output {i+1}: '{request.texts[i][:50]}...'")
                elif len(corrected_texts) > len(request.texts):
                    # Trim excess texts
                    logger.warning(f"Trimming {len(corrected_texts) - len(request.texts)} excess texts from output")
                    corrected_texts = corrected_texts[:len(request.texts)]
            
            changes = result_json.get("changes", [])
            if not isinstance(changes, list) or len(changes) != len(request.texts):
                logger.warning(f"Gemini batch correction: invalid changes format, creating empty changes list")
                changes = [[] for _ in request.texts]
            
            return BatchTextCorrectionResponse(
                original_texts=request.texts,
                corrected_texts=corrected_texts,
                changes=changes,
                confidence=result_json.get("confidence", 0.8),
                provider="Google Gemini",
                model=self.model_name,
                page_number=request.page_number,
                section_name=request.section_name
            )
            
        except Exception as e:
            logger.error(f"Google Gemini batch text correction failed: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get available Gemini models"""
        # We always keep a static list; if authentication hasn't happened yet
        # this will still return the curated defaults.
        if not self.available_models:
            self.available_models = [
                "gemini-3-pro-preview",
                "gemini-3-flash-preview",
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-pro-latest",
                "gemini-flash-latest",
                "gemini-2.0-flash",
                "gemini-2.0-pro",
                "gemini-1.5-flash",
                "gemini-1.5-pro",
            ]
        return self.available_models
    
    async def fetch_available_models_internal(self) -> List[str]:
        """Internal method to fetch models using direct REST API (more reliable than SDK)"""
        try:
            # Use direct REST API call since SDK's models.list() seems unreliable
            import requests
            
            def fetch_models_via_rest():
                url = f"https://generativelanguage.googleapis.com/v1beta/models?key={self.api_key}"
                
                try:
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        data = response.json()
                        if 'models' in data:
                            available_models = []
                            for model in data['models']:
                                model_name = model.get('name', '')
                                supported_methods = model.get('supportedGenerationMethods', [])
                                
                                # Check if model supports generateContent
                                if 'generateContent' in supported_methods:
                                    # Remove 'models/' prefix if present
                                    if model_name.startswith('models/'):
                                        model_name = model_name.replace('models/', '')
                                    
                                    # Only include Gemini models suitable for text generation
                                    if 'gemini' in model_name.lower():
                                        # Skip specialized models
                                        exclusions = [
                                            'tts', 'vision', 'video', 'image', 'embedding',
                                            'native-audio', 'live-api', 'code-execution',
                                            'flash-lite'  # Lite models may have limitations
                                        ]
                                        
                                        if not any(exclusion in model_name.lower() for exclusion in exclusions):
                                            available_models.append(model_name)
                                            
                            return available_models
                    else:
                        logger.error(f"REST API returned {response.status_code}: {response.text[:200]}")
                        return []
                except Exception as e:
                    logger.error(f"REST API call failed: {e}")
                    return []
            
            # Execute REST call in thread pool
            discovered_models = await asyncio.get_event_loop().run_in_executor(
                None, fetch_models_via_rest
            )
            
            logger.info(f"REST API discovered {len(discovered_models)} suitable models")
            
            if discovered_models:
                # Sort by preference (newer, more capable models first)
                def model_priority(model_name):
                    if 'gemini-2.5-flash' in model_name:
                        return 1  # Best balance of speed and capability
                    elif 'gemini-2.5-pro' in model_name:
                        return 2  # Most capable
                    elif 'gemini-2.0-flash' in model_name:
                        return 3  # Good alternative
                    elif 'gemini-2.0' in model_name:
                        return 4
                    elif 'gemini-1.5' in model_name:
                        return 5
                    else:
                        return 6
                
                discovered_models.sort(key=model_priority)
                logger.info(f"Prioritized models: {discovered_models}")
                return discovered_models
            else:
                logger.warning("No suitable models found via REST API")
                # Return models we know work from the diagnostic
                return ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]
                
        except Exception as e:
            logger.error(f"Failed to fetch models via REST API: {e}")
            # Return models that the diagnostic proved work
            return ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"]

    async def fetch_available_models(self) -> List[str]:
        """Fetch available models from Google Gemini API.

        For robustness in frozen executables we no longer call the REST
        discovery endpoint here. Instead we return the same curated list used
        during authentication, which is sufficient for the UI's model
        dropdown and keeps behaviour identical between source and EXE builds.
        """
        return self.get_available_models()
    
    def get_auth_url(self) -> str:
        """Get Google AI Studio API key URL"""
        return "https://aistudio.google.com/app/apikey"
    
    def get_auth_instructions(self) -> str:
        """Get Google Gemini authentication instructions"""
        return """To use Google Gemini services:

1. Visit: https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Choose "Create API key in new project" or select existing project
5. Copy the generated API key
6. Paste the API key in the configuration

Alternative method (Google Cloud Console):
1. Visit: https://console.cloud.google.com/apis/credentials
2. Create or select a project
3. Enable the "Generative Language API"
4. Create credentials (API Key)
5. Copy and use the API key

Note: Gemini API has generous free tier limits.
Check your usage at: https://aistudio.google.com/app/apikey"""


class AnthropicService(AIServiceBase):
    """Anthropic Claude text correction service"""
    
    def __init__(self, api_key: str, model_name: str = None):
        super().__init__(api_key, model_name)
        self.client = None
    
    async def authenticate(self) -> bool:
        """Authenticate with Anthropic Claude"""
        try:
            if not ANTHROPIC_AVAILABLE:
                logger.error("Anthropic library not available")
                return False
            
            self.client = anthropic.Anthropic(api_key=self.api_key)
            
            # Test models to find available ones
            self.available_models = await self.fetch_available_models_internal()
            
            if not self.available_models:
                raise Exception("No models available or authentication failed")
            
            # Set default model if none specified
            if not self.model_name:
                self.model_name = self.available_models[0]
            
            # Test authentication with the first available model
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model_name,
                    max_tokens=10,
                    messages=[{"role": "user", "content": "Hi"}]
                )
            )
            
            self.is_authenticated = True
            logger.info(f"Anthropic Claude authentication successful. Found {len(self.available_models)} models")
            return True
        except Exception as e:
            logger.error(f"Anthropic Claude authentication failed: {e}")
            return False
    
    async def correct_text(self, request: TextCorrectionRequest) -> TextCorrectionResponse:
        """Correct text using Anthropic Claude"""
        if not self.is_authenticated:
            if not await self.authenticate():
                raise Exception("Anthropic Claude authentication failed")
        
        try:
            system_prompt = self.get_json_system_prompt(request.correction_type)
            user_message = f"Please correct the following text:\n\n{request.text}"
            
            if request.context:
                user_message += f"\n\nContext: {request.context}"
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model_name,
                    max_tokens=4000,
                    temperature=0.1,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}]
                )
            )
            
            content = response.content[0].text.strip()
            result_json = self.parse_json_response(content, is_batch=False)
            
            return TextCorrectionResponse(
                original_text=request.text,
                corrected_text=result_json.get("corrected_text", request.text),
                changes=result_json.get("changes", []),
                confidence=result_json.get("confidence", 0.8),
                provider="Anthropic Claude",
                model=self.model_name
            )
            
        except Exception as e:
            logger.error(f"Anthropic Claude text correction failed: {e}")
            raise
    
    async def correct_text_batch(self, request: BatchTextCorrectionRequest) -> BatchTextCorrectionResponse:
        """Correct multiple texts using Anthropic Claude in a single request"""
        if not self.is_authenticated:
            if not await self.authenticate():
                raise Exception("Anthropic Claude authentication failed")
        
        try:
            system_prompt = self.get_json_system_prompt(request.correction_type, batch_mode=True)
            
            # Create JSON input for the AI
            input_json = {
                "texts": request.texts
            }
            
            # Add context information if provided
            if request.context:
                input_json["context"] = request.context
            if request.page_number:
                input_json["page_number"] = request.page_number
            if request.section_name:
                input_json["section_name"] = request.section_name
            
            user_message = f"Please process this JSON input and return the corrected texts:\n\n{json.dumps(input_json, indent=2, ensure_ascii=False)}"
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.messages.create(
                    model=self.model_name,
                    max_tokens=8000,
                    temperature=0.1,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_message}]
                )
            )
            
            content = response.content[0].text.strip()
            result_json = self.parse_json_response(content, is_batch=True, expected_count=len(request.texts))
            
            # Strict validation of response format
            corrected_texts = result_json.get("corrected_texts", [])
            if not isinstance(corrected_texts, list):
                logger.error(f"Claude batch correction: corrected_texts is not a list, got {type(corrected_texts)}")
                corrected_texts = request.texts  # Fallback to original texts
            elif len(corrected_texts) != len(request.texts):
                logger.error(f"Claude batch correction returned {len(corrected_texts)} texts, expected {len(request.texts)}")
                logger.error(f"Input texts ({len(request.texts)}): {[text[:50] + '...' if len(text) > 50 else text for text in request.texts]}")
                logger.error(f"Output texts ({len(corrected_texts)}): {[text[:50] + '...' if len(text) > 50 else text for text in corrected_texts]}")
                logger.error(f"AI Response content (first 500 chars): {content[:500]}...")
                
                # Strict padding/trimming with original texts as fallback
                if len(corrected_texts) < len(request.texts):
                    # Pad missing texts with originals
                    for i in range(len(corrected_texts), len(request.texts)):
                        corrected_texts.append(request.texts[i])
                        logger.warning(f"Using original text for missing output {i+1}: '{request.texts[i][:50]}...'")
                elif len(corrected_texts) > len(request.texts):
                    # Trim excess texts
                    logger.warning(f"Trimming {len(corrected_texts) - len(request.texts)} excess texts from output")
                    corrected_texts = corrected_texts[:len(request.texts)]
            
            changes = result_json.get("changes", [])
            if not isinstance(changes, list) or len(changes) != len(request.texts):
                logger.warning(f"Claude batch correction: invalid changes format, creating empty changes list")
                changes = [[] for _ in request.texts]
            
            return BatchTextCorrectionResponse(
                original_texts=request.texts,
                corrected_texts=corrected_texts,
                changes=changes,
                confidence=result_json.get("confidence", 0.8),
                provider="Anthropic Claude",
                model=self.model_name,
                page_number=request.page_number,
                section_name=request.section_name
            )
            
        except Exception as e:
            logger.error(f"Anthropic Claude batch text correction failed: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get available Claude models"""
        return self.available_models if self.available_models else ["claude-3-haiku-20240307"]
    
    async def fetch_available_models_internal(self) -> List[str]:
        """Internal method to test available Claude models"""
        try:
            # Anthropic doesn't have a public models list API endpoint yet
            # We'll try to test each known model to see which ones are available
            test_models = [
                "claude-3-5-sonnet-20241022",  # Latest Claude 3.5 Sonnet
                "claude-3-5-sonnet-20240620",  # Previous Claude 3.5 Sonnet
                "claude-3-sonnet-20240229",    # Claude 3 Sonnet
                "claude-3-opus-20240229",      # Claude 3 Opus
                "claude-3-haiku-20240307",     # Claude 3 Haiku
            ]
            
            available_models = []
            
            # Test a simple message with each model
            for model in test_models:
                try:
                    await asyncio.get_event_loop().run_in_executor(
                        None,
                        lambda m=model: self.client.messages.create(
                            model=m,
                            max_tokens=1,
                            messages=[{"role": "user", "content": "Hi"}]
                        )
                    )
                    available_models.append(model)
                    logger.debug(f"Claude model {model} is available")
                except Exception as model_error:
                    logger.debug(f"Claude model {model} not available: {model_error}")
                    continue
                
                # Stop after finding first working model to avoid rate limiting
                if len(available_models) >= 1:
                    break
            
            return available_models if available_models else ["claude-3-haiku-20240307"]
                
        except Exception as e:
            logger.error(f"Failed to test Claude models: {e}")
            return ["claude-3-haiku-20240307"]  # Fallback

    async def fetch_available_models(self) -> List[str]:
        """Fetch available models from Anthropic API"""
        if not self.is_authenticated:
            logger.warning("Not authenticated, cannot fetch models")
            return []
        
        if self.available_models:
            return self.available_models
        
        self.available_models = await self.fetch_available_models_internal()
        return self.available_models
    
    def get_auth_url(self) -> str:
        """Get Anthropic Console API key URL"""
        return "https://console.anthropic.com/settings/keys"
    
    def get_auth_instructions(self) -> str:
        """Get Anthropic Claude authentication instructions"""
        return """To use Anthropic Claude services:

1. Visit: https://console.anthropic.com/settings/keys
2. Sign in to your Anthropic account (or create one)
3. Click "Create Key"
4. Enter a name for your key
5. Copy the generated API key (starts with 'sk-ant-...')
6. Paste the API key in the configuration

Note: Anthropic Claude requires account verification and may have
usage limits based on your plan. Free tier includes some credits.
Check your usage at: https://console.anthropic.com/settings/usage"""


class GroqService(AIServiceBase):
    """Groq text correction service"""
    
    def __init__(self, api_key: str, model_name: str = None):
        super().__init__(api_key, model_name)
        self.client = None
    
    async def authenticate(self) -> bool:
        """Authenticate with Groq"""
        try:
            if not GROQ_AVAILABLE:
                logger.error("Groq library not available")
                return False
            
            # Initialize Groq client with custom httpx client to avoid compatibility issues
            import httpx
            http_client = httpx.Client()
            self.client = Groq(api_key=self.api_key, http_client=http_client)
            
            # Fetch available models during authentication
            models_response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.models.list()
            )
            
            self.available_models = await self.fetch_available_models_internal(models_response)
            
            if not self.available_models:
                raise Exception("No models available or authentication failed")
            
            # Set default model if none specified
            if not self.model_name:
                self.model_name = self.available_models[0]
            
            # Test authentication with a simple request
            await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    messages=[{"role": "user", "content": "Hi"}],
                    model=self.model_name,
                    max_tokens=1
                )
            )
            
            self.is_authenticated = True
            logger.info(f"Groq authentication successful. Found {len(self.available_models)} models")
            return True
        except Exception as e:
            logger.error(f"Groq authentication failed: {e}")
            return False
    
    async def correct_text(self, request: TextCorrectionRequest) -> TextCorrectionResponse:
        """Correct text using Groq"""
        if not self.is_authenticated:
            if not await self.authenticate():
                raise Exception("Groq authentication failed")
        
        try:
            system_prompt = self.get_json_system_prompt(request.correction_type)
            user_message = f"Please correct the following text:\n\n{request.text}"
            
            if request.context:
                user_message += f"\n\nContext: {request.context}"
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    model=self.model_name,
                    max_tokens=4000,
                    temperature=0.1
                )
            )
            
            content = response.choices[0].message.content.strip()
            result_json = self.parse_json_response(content, is_batch=False)
            
            return TextCorrectionResponse(
                original_text=request.text,
                corrected_text=result_json.get("corrected_text", request.text),
                changes=result_json.get("changes", []),
                confidence=result_json.get("confidence", 0.8),
                provider="Groq",
                model=self.model_name
            )
            
        except Exception as e:
            logger.error(f"Groq text correction failed: {e}")
            raise
    
    async def correct_text_batch(self, request: BatchTextCorrectionRequest) -> BatchTextCorrectionResponse:
        """Correct multiple texts using Groq in a single request"""
        if not self.is_authenticated:
            if not await self.authenticate():
                raise Exception("Groq authentication failed")
        
        try:
            system_prompt = self.get_json_system_prompt(request.correction_type, batch_mode=True)
            
            # Create JSON input for the AI
            input_json = {
                "texts": request.texts
            }
            
            # Add context information if provided
            if request.context:
                input_json["context"] = request.context
            if request.page_number:
                input_json["page_number"] = request.page_number
            if request.section_name:
                input_json["section_name"] = request.section_name
            
            user_message = f"Please process this JSON input and return the corrected texts:\n\n{json.dumps(input_json, indent=2, ensure_ascii=False)}"
            
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    model=self.model_name,
                    max_tokens=8000,
                    temperature=0.1
                )
            )
            
            content = response.choices[0].message.content.strip()
            result_json = self.parse_json_response(content, is_batch=True, expected_count=len(request.texts))
            
            # Strict validation of response format
            corrected_texts = result_json.get("corrected_texts", [])
            if not isinstance(corrected_texts, list):
                logger.error(f"Groq batch correction: corrected_texts is not a list, got {type(corrected_texts)}")
                corrected_texts = request.texts  # Fallback to original texts
            elif len(corrected_texts) != len(request.texts):
                logger.error(f"Groq batch correction returned {len(corrected_texts)} texts, expected {len(request.texts)}")
                logger.error(f"Input texts ({len(request.texts)}): {[text[:50] + '...' if len(text) > 50 else text for text in request.texts]}")
                logger.error(f"Output texts ({len(corrected_texts)}): {[text[:50] + '...' if len(text) > 50 else text for text in corrected_texts]}")
                logger.error(f"AI Response content (first 500 chars): {content[:500]}...")
                
                # Strict padding/trimming with original texts as fallback
                if len(corrected_texts) < len(request.texts):
                    # Pad missing texts with originals
                    for i in range(len(corrected_texts), len(request.texts)):
                        corrected_texts.append(request.texts[i])
                        logger.warning(f"Using original text for missing output {i+1}: '{request.texts[i][:50]}...'")
                elif len(corrected_texts) > len(request.texts):
                    # Trim excess texts
                    logger.warning(f"Trimming {len(corrected_texts) - len(request.texts)} excess texts from output")
                    corrected_texts = corrected_texts[:len(request.texts)]
            
            changes = result_json.get("changes", [])
            if not isinstance(changes, list) or len(changes) != len(request.texts):
                logger.warning(f"Groq batch correction: invalid changes format, creating empty changes list")
                changes = [[] for _ in request.texts]
            
            return BatchTextCorrectionResponse(
                original_texts=request.texts,
                corrected_texts=corrected_texts,
                changes=changes,
                confidence=result_json.get("confidence", 0.8),
                provider="Groq",
                model=self.model_name,
                page_number=request.page_number,
                section_name=request.section_name
            )
            
        except Exception as e:
            logger.error(f"Groq batch text correction failed: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get available Groq models"""
        return self.available_models if self.available_models else ["mixtral-8x7b-32768"]
    
    async def fetch_available_models_internal(self, models_response) -> List[str]:
        """Internal method to process Groq models response"""
        try:
            suitable_models = []
            for model in models_response.data:
                model_id = model.id
                
                # Include text generation models suitable for our use case
                # Groq typically hosts Llama, Mixtral, Gemma models
                if any(prefix in model_id.lower() for prefix in ['llama', 'mixtral', 'gemma', 'qwen', 'deepseek','kimi','gpt-']):
                    # Prioritize newer/better models
                    if any(good_model in model_id.lower() for good_model in ['mixtral', 'llama3', 'llama-3']):
                        suitable_models.insert(0, model_id)
                    else:
                        suitable_models.append(model_id)
            
            return suitable_models if suitable_models else ["mixtral-8x7b-32768"]
                
        except Exception as e:
            logger.error(f"Failed to process Groq models: {e}")
            return ["mixtral-8x7b-32768"]  # Fallback

    async def fetch_available_models(self) -> List[str]:
        """Fetch available models from Groq API"""
        if not self.is_authenticated:
            logger.warning("Not authenticated, cannot fetch models")
            return []
        
        if self.available_models:
            return self.available_models
        
        try:
            # Fetch models from Groq API
            models_response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.client.models.list()
            )
            
            self.available_models = await self.fetch_available_models_internal(models_response)
            return self.available_models
                
        except Exception as e:
            logger.error(f"Failed to fetch Groq models: {e}")
            return []
    
    def get_auth_url(self) -> str:
        """Get Groq Console API key URL"""
        return "https://console.groq.com/keys"
    
    def get_auth_instructions(self) -> str:
        """Get Groq authentication instructions"""
        return """To use Groq services:

1. Visit: https://console.groq.com/keys
2. Sign in to your Groq account (or create one)
3. Click "Create API Key"
4. Enter a name for your key
5. Copy the generated API key (starts with 'gsk_...')
6. Paste the API key in the configuration

Note: Groq offers very fast inference with generous free tier limits.
Their models are optimized for speed and efficiency.
Check your usage at: https://console.groq.com/settings/usage"""


class AIServiceManager:
    """Manager for handling multiple AI services"""
    
    def __init__(self):
        self.services: Dict[str, AIServiceBase] = {}
        self.active_service: Optional[str] = None
    
    def register_service(self, provider: str, service: AIServiceBase):
        """Register an AI service"""
        self.services[provider] = service
    
    def create_service(self, provider: str, api_key: str = None, model: str = None) -> AIServiceBase:
        """Create a new service instance for the given provider"""
        if provider == "OpenAI":
            return OpenAIService(api_key=api_key, model=model)
        elif provider == "Google Gemini":
            return GoogleGeminiService(api_key=api_key, model=model)
        elif provider == "Anthropic Claude":
            return AnthropicService(api_key=api_key, model=model)
        elif provider == "Groq":
            return GroqService(api_key=api_key, model=model)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    async def ensure_service_registered(self, provider: str):
        """Ensure a service is registered, creating it if necessary"""
        if provider in self.services:
            return  # Already registered
        
        try:
            # Get configuration from ai_config
            from converter.config import ai_config
            
            api_key = ai_config.get_api_key(provider)
            model = ai_config.get_model(provider)
            
            if not api_key:
                raise ValueError(f"No API key configured for {provider}")
            
            # Create the service
            service = self.create_service(provider, api_key, model)
            
            # Register it
            self.register_service(provider, service)
            
            logger.info(f"Auto-registered service for {provider}")
            
        except Exception as e:
            logger.error(f"Failed to auto-register service for {provider}: {e}")
            raise
    
    async def set_active_service(self, provider: str):
        """Set the active AI service (auto-registers if needed)"""
        if provider not in self.services:
            # Try to auto-register the service
            await self.ensure_service_registered(provider)
        
        if provider in self.services:
            self.active_service = provider
        else:
            raise ValueError(f"Service {provider} not registered")
    
    def set_active_service_sync(self, provider: str):
        """Set the active AI service (sync version, requires pre-registration)"""
        if provider in self.services:
            self.active_service = provider
        else:
            raise ValueError(f"Service {provider} not registered")
    
    async def authenticate_service(self, provider: str) -> bool:
        """Authenticate a specific service"""
        if provider in self.services:
            return await self.services[provider].authenticate()
        return False
    
    async def fetch_models_for_provider(self, provider: str) -> List[str]:
        """Fetch available models for a provider after authentication"""
        if provider in self.services:
            return await self.services[provider].fetch_available_models()
        return []
    
    async def authenticate_and_fetch_models(self, provider: str) -> tuple[bool, List[str]]:
        """Authenticate and fetch available models in one call"""
        if provider not in self.services:
            return False, []
        
        service = self.services[provider]
        
        # First authenticate
        auth_success = await service.authenticate()
        if not auth_success:
            return False, service.get_available_models()  # Return default models
        
        # Then fetch available models
        try:
            models = await service.fetch_available_models()
            return True, models
        except Exception as e:
            logger.error(f"Failed to fetch models for {provider}: {e}")
            return True, service.get_available_models()  # Auth succeeded but model fetch failed
    
    async def correct_text(self, request: TextCorrectionRequest, provider: str = None) -> TextCorrectionResponse:
        """Correct text using specified or active service"""
        service_key = provider or self.active_service
        
        if not service_key or service_key not in self.services:
            raise ValueError("No active AI service available")
        
        service = self.services[service_key]
        return await service.correct_text(request)
    
    async def correct_text_batch(self, request: BatchTextCorrectionRequest, provider: str = None) -> BatchTextCorrectionResponse:
        """Correct multiple texts using specified or active service"""
        service_key = provider or self.active_service
        
        if not service_key or service_key not in self.services:
            raise ValueError("No active AI service available")
        
        service = self.services[service_key]
        return await service.correct_text_batch(request)
    
    def get_active_service_name(self) -> Optional[str]:
        """Get the name of the currently active service"""
        return self.active_service
    
    def get_active_service(self) -> Optional[AIServiceBase]:
        """Get the currently active service object"""
        if self.active_service and self.active_service in self.services:
            return self.services[self.active_service]
        return None
    
    def get_available_providers(self) -> List[str]:
        """Get list of available AI providers"""
        available = []
        if OPENAI_AVAILABLE:
            available.append("OpenAI")
        if GOOGLE_AVAILABLE:
            available.append("Google Gemini")
        if ANTHROPIC_AVAILABLE:
            available.append("Anthropic Claude")
        if GROQ_AVAILABLE:
            available.append("Groq")
        return available
    
    def get_provider_models(self, provider: str) -> List[str]:
        """Get available models for a provider"""
        if provider in self.services:
            return self.services[provider].get_available_models()
        return []
    
    def create_service(self, provider: str, api_key: str, model_name: str = None) -> AIServiceBase:
        """Factory method to create AI services"""
        provider_classes = {
            "OpenAI": OpenAIService,
            "Google Gemini": GoogleGeminiService,
            "Anthropic Claude": AnthropicService,
            "Groq": GroqService
        }
        
        if provider not in provider_classes:
            raise ValueError(f"Unsupported provider: {provider}")
        
        service_class = provider_classes[provider]
        return service_class(api_key, model_name)
    
    def get_auth_url(self, provider: str) -> str:
        """Get authentication URL for a provider"""
        try:
            service = self.create_service(provider, "dummy_key")
            return service.get_auth_url()
        except:
            return ""
    
    def get_auth_instructions(self, provider: str) -> str:
        """Get authentication instructions for a provider"""
        try:
            service = self.create_service(provider, "dummy_key")
            return service.get_auth_instructions()
        except:
            return f"Please visit the {provider} website to get an API key."
    
    def get_provider_info(self, provider: str) -> Dict[str, str]:
        """Get comprehensive provider information"""
        return {
            "name": provider,
            "auth_url": self.get_auth_url(provider),
            "instructions": self.get_auth_instructions(provider),
            "models": self.get_provider_models(provider)
        }


# Global AI service manager instance
ai_manager = AIServiceManager()