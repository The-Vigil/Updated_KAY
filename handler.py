import runpod
import base64
import io
import asyncio
import os
import time
import logging
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
from groq import Groq
from openai import OpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("kay_dispatcher")

# Configuration class
@dataclass
class Config:
    groq_api_key: str
    openai_api_key: str
    system_prompt_path: str
    whisper_model: str = "whisper-large-v3"
    llm_model: str = "llama-3.3-70b-versatile"
    tts_model: str = "tts-1"
    tts_voice: str = "alloy"
    chunk_size: int = 256 * 1024  # 256KB chunks
    llm_temp: float = 0.5
    llm_max_tokens: int = 2048

# Session management
class ConversationManager:
    def __init__(self, max_history: int = 10):
        self.sessions: Dict[str, List[Dict[str, str]]] = {}
        self.max_history = max_history
    
    def get_session(self, session_id: str) -> List[Dict[str, str]]:
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        return self.sessions[session_id]
    
    def add_message(self, session_id: str, role: str, content: str) -> None:
        session = self.get_session(session_id)
        session.append({"role": role, "content": content})
        
        # Trim history if needed
        if len(session) > self.max_history * 2:  # Keep pairs of messages
            self.sessions[session_id] = session[-self.max_history * 2:]

# Custom exception classes
class ConfigError(Exception):
    """Configuration related errors"""
    pass

class AudioProcessingError(Exception):
    """Audio processing related errors"""
    pass

class LLMError(Exception):
    """LLM related errors"""
    pass

class TTSError(Exception):
    """TTS related errors"""
    pass

class InputValidationError(Exception):
    """Input validation related errors"""
    pass

# Initialize configuration
def load_config() -> Config:
    """Load and validate configuration from environment variables"""
    required_vars = ["GROQ_API_KEY", "OPENAI_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ConfigError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    # Check for system prompt file
    system_prompt_path = os.getenv("SYSTEM_PROMPT_PATH", "system_prompt.txt")
    if not Path(system_prompt_path).exists():
        logger.warning(f"System prompt file not found at {system_prompt_path}. Will use default.")
    
    return Config(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        system_prompt_path=system_prompt_path,
        whisper_model=os.getenv("WHISPER_MODEL", "whisper-large-v3"),
        llm_model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        tts_model=os.getenv("TTS_MODEL", "tts-1"),
        tts_voice=os.getenv("TTS_VOICE", "alloy"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "262144")),
        llm_temp=float(os.getenv("LLM_TEMP", "0.5")),
        llm_max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048"))
    )

def load_system_prompt(config: Config) -> str:
    """Load system prompt from file or use default if file not found"""
    try:
        with open(config.system_prompt_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"System prompt file not found at {config.system_prompt_path}. Using default.")
        # Return a simplified version for fallback
        return """
        # KAY AI Dispatch Assistant
        You are KAY, an AI-powered virtual dispatcher for the trucking industry.
        Collect: location, equipment type, destination, pickup date/time, rate preferences, and special requirements.
        Present loads in a structured format with origin, destination, rate, pickup time, broker, equipment, and weight.
        Maintain a professional tone and ensure all regulations are followed.
        """

# Initialize global services
try:
    config = load_config()
    system_prompt = load_system_prompt(config)
    groq_client = Groq(api_key=config.groq_api_key)
    openai_client = OpenAI(api_key=config.openai_api_key)
    conversation_manager = ConversationManager()
    logger.info("Services initialized successfully")
except ConfigError as e:
    logger.error(f"Configuration error: {e}")
    raise
except Exception as e:
    logger.error(f"Initialization error: {e}")
    raise

async def process_audio(audio_base64: str, config: Config) -> str:
    """Processes base64-encoded audio, transcribes it using Whisper, and returns text."""
    logger.info("Processing audio input")
    try:
        audio_bytes = base64.b64decode(audio_base64)
        audio_buffer = io.BytesIO(audio_bytes)

        translation = await asyncio.to_thread(
            lambda: groq_client.audio.translations.create(
                file=audio_buffer, 
                model=config.whisper_model,
                response_format="json",
                temperature=0.0
            )
        )
        
        transcribed_text = translation.text
        logger.info("Audio transcription successful")
        return transcribed_text
    except Exception as e:
        logger.error(f"Audio processing error: {e}")
        raise AudioProcessingError(f"Failed to process audio: {str(e)}")

async def get_llm_response(text_input: str, session_id: str, config: Config) -> str:
    """Generates response from LLM with conversation history."""
    logger.info(f"Generating LLM response for session: {session_id}")
    try:
        # Get conversation history
        conversation = conversation_manager.get_session(session_id)
        
        # Create messages array with system prompt and history
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(conversation)
        messages.append({"role": "user", "content": text_input})
        
        response = await asyncio.to_thread(
            lambda: groq_client.chat.completions.create(
                messages=messages,
                model=config.llm_model,
                temperature=config.llm_temp,
                max_tokens=config.llm_max_tokens
            )
        )
        
        # Extract response content
        response_text = response.choices[0].message.content
        
        # Update conversation history
        conversation_manager.add_message(session_id, "user", text_input)
        conversation_manager.add_message(session_id, "assistant", response_text)
        
        logger.info("LLM response generated successfully")
        return response_text
    except Exception as e:
        logger.error(f"LLM processing error: {e}")
        raise LLMError(f"Failed to generate LLM response: {str(e)}")

async def generate_tts(audio_text: str, config: Config) -> str:
    """Generates TTS audio from text and returns base64-encoded audio."""
    logger.info("Generating TTS audio")
    try:
        tts_audio = openai_client.audio.speech.create(
            model=config.tts_model, 
            voice=config.tts_voice, 
            input=audio_text
        ).iter_bytes(chunk_size=config.chunk_size)

        audio_buffer = io.BytesIO()
        for chunk in tts_audio:
            audio_buffer.write(chunk)

        audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode()
        logger.info("TTS audio generated successfully")
        return audio_base64
    except Exception as e:
        logger.error(f"TTS processing error: {e}")
        raise TTSError(f"Failed to generate TTS audio: {str(e)}")

def validate_input(job_input: Dict[str, Any]) -> None:
    """Validates the input structure."""
    if not isinstance(job_input, dict):
        raise InputValidationError("Input must be a dictionary")
    
    if "type" not in job_input:
        raise InputValidationError("Input must contain 'type' field")
    
    input_type = job_input["type"]
    if input_type not in ["audio", "text"]:
        raise InputValidationError("Input type must be 'audio' or 'text'")
    
    if input_type == "audio" and ("audio" not in job_input or not job_input["audio"]):
        raise InputValidationError("Audio input must contain 'audio' field with base64-encoded audio")
    
    if input_type == "text" and ("text" not in job_input or not job_input["text"]):
        raise InputValidationError("Text input must contain 'text' field")

async def async_handler(job):
    request_id = job.get("id", "unknown")
    session_id = job.get("input", {}).get("session_id", request_id)
    start_time = time.time()
    
    logger.info(f"Processing request {request_id} for session {session_id}")
    
    try:
        job_input = job.get("input", {})
        validate_input(job_input)
        
        input_type = job_input["type"]
        
        # Process input based on type
        if input_type == "audio":
            text_input = await process_audio(job_input["audio"], config)
        else:
            text_input = job_input["text"]
        
        if not text_input:
            raise InputValidationError("Failed to process input")
        
        # Generate LLM response
        llm_response = await get_llm_response(text_input, session_id, config)
        
        # Generate TTS audio
        tts_audio_base64 = await generate_tts(llm_response, config)
        
        request_time = time.time() - start_time
        logger.info(f"Request {request_id} completed in {request_time:.2f}s")
        
        return {
            "user_input": text_input,
            "assistant_response": {
                "text": llm_response,
                "audio": tts_audio_base64
            },
            "session_id": session_id,
            "processing_time": request_time
        }

    except InputValidationError as e:
        logger.error(f"Input validation error in request {request_id}: {e}")
        return {"error": str(e), "error_type": "input_validation"}
    except AudioProcessingError as e:
        logger.error(f"Audio processing error in request {request_id}: {e}")
        return {"error": str(e), "error_type": "audio_processing"}
    except LLMError as e:
        logger.error(f"LLM error in request {request_id}: {e}")
        return {"error": str(e), "error_type": "llm"}
    except TTSError as e:
        logger.error(f"TTS error in request {request_id}: {e}")
        return {"error": str(e), "error_type": "tts"}
    except Exception as e:
        logger.error(f"Unexpected error in request {request_id}: {e}")
        return {"error": "An unexpected error occurred", "error_type": "internal"}

def main():
    logger.info("Starting KAY AI Dispatcher server...")
    runpod.serverless.start({"handler": async_handler})

if __name__ == "__main__":
    main()
