import runpod
import base64
from groq import Groq
from openai import OpenAI
import os
import time
import io
import threading
import asyncio
from functools import lru_cache

# Initialize clients with proper connection pooling
groq_client = Groq(
    api_key=os.environ["GROQ_API_KEY"],
    max_retries=2,  # Reduce wait time on failures
    timeout=45.0    # Set reasonable timeout
)

openai_client = OpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    max_retries=2,
    timeout=45.0
)

# Significantly shortened system prompt
SYSTEM_PROMPT = """You are KAY, an AI virtual dispatcher for KAYAAN, helping truck drivers find optimal loads.
Collect: location, equipment type, destination, pickup date/time, rate preferences, and special requirements.
Present loads with origin, destination, rate ($/mile), pickup time, broker, equipment, and weight.
Use professional yet friendly tone. Be efficient, knowledgeable, and helpful."""

# Cache system prompt and other static data
@lru_cache(maxsize=1)
def get_system_prompt():
    return SYSTEM_PROMPT

# Load database moved to code
LOAD_DATABASE = {
    "Detroit-Chicago": {"distance": 280, "rate": 1850},  # $6.60/mile
    "Indianapolis-Chicago": {"distance": 180, "rate": 1200},  # $6.67/mile
    "Milwaukee-Chicago": {"distance": 90, "rate": 800},  # $8.89/mile
    "Dallas-Houston": {"distance": 240, "rate": 1200},  # $5.00/mile
    "LA-Phoenix": {"distance": 375, "rate": 1600},  # $4.27/mile
}

# In-memory cache with time-based expiration
response_cache = {}
CACHE_EXPIRY = 3600  # 1 hour

def get_from_cache(key):
    """Get item from cache if it exists and is not expired"""
    if key in response_cache:
        item, timestamp = response_cache[key]
        if time.time() - timestamp < CACHE_EXPIRY:
            return item
    return None

def save_to_cache(key, value):
    """Save item to cache with current timestamp"""
    response_cache[key] = (value, time.time())
    
    # Cleanup old cache entries (simple approach)
    if len(response_cache) > 100:  # Arbitrary limit
        # Remove oldest entries
        sorted_keys = sorted(response_cache.keys(), 
                          key=lambda k: response_cache[k][1])
        for old_key in sorted_keys[:20]:  # Remove oldest 20%
            del response_cache[old_key]

async def transcribe_audio(audio_bytes):
    """Transcribe audio in memory with optimized settings"""
    # Check for cached transcription by audio hash
    audio_hash = hash(audio_bytes)
    cached_result = get_from_cache(f"transcription:{audio_hash}")
    if cached_result:
        return cached_result
    
    audio_file = io.BytesIO(audio_bytes)
    translation = groq_client.audio.translations.create(
        file=("audio.wav", audio_file),
        model="whisper-large-v3",
        response_format="json",
        temperature=0.0
    )
    
    result = translation.text
    save_to_cache(f"transcription:{audio_hash}", result)
    return result

async def generate_llm_response(text_input):
    """Generate LLM response using streaming for faster time-to-first-token"""
    # Check input cache for common queries
    input_hash = hash(text_input)
    cached_result = get_from_cache(f"llm:{input_hash}")
    if cached_result:
        return cached_result
    
    # Use a smaller model for faster responses
    response_chunks = []
    
    try:
        stream = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": text_input}
            ],
            model="llama-3.1-8b-instant",  # Much faster model
            temperature=0.5,
            max_tokens=1024,
            stream=True  # Enable streaming
        )
        
        # Process the stream
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                response_chunks.append(chunk.choices[0].delta.content)
    except Exception as e:
        # Fallback to non-streaming if streaming fails
        print(f"Streaming failed, falling back: {str(e)}")
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": get_system_prompt()},
                {"role": "user", "content": text_input}
            ],
            model="llama-3.1-8b-instant",
            temperature=0.5,
            max_tokens=1024
        )
        response_chunks = [chat_completion.choices[0].message.content]
    
    full_response = "".join(response_chunks)
    save_to_cache(f"llm:{input_hash}", full_response)
    return full_response

async def generate_tts(text):
    """Generate TTS directly to memory with optimized chunking"""
    # Check for cached TTS by text hash
    text_hash = hash(text)
    cached_result = get_from_cache(f"tts:{text_hash}")
    if cached_result:
        return cached_result
    
    # Process in parallel if text is longer
    if len(text) > 500:
        # Split into sentences to preserve natural pauses
        sentences = text.replace('. ', '.|').replace('! ', '!|').replace('? ', '?|').split('|')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < 500:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        # Process chunks in parallel
        audio_chunks = []
        tasks = []
        
        for chunk in chunks:
            tts_response = openai_client.audio.speech.create(
                model="tts-1",
                voice="alloy",
                input=chunk
            )
            
            chunk_bytes = io.BytesIO()
            for data in tts_response.iter_bytes():
                chunk_bytes.write(data)
            
            audio_chunks.append(chunk_bytes.getvalue())
        
        # Combine audio chunks
        combined = b''.join(audio_chunks)
        audio_base64 = base64.b64encode(combined).decode()
    else:
        # Process normally for short text
        tts_response = openai_client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        
        audio_bytes = io.BytesIO()
        for chunk in tts_response.iter_bytes():
            audio_bytes.write(chunk)
        
        audio_base64 = base64.b64encode(audio_bytes.getvalue()).decode()
    
    save_to_cache(f"tts:{text_hash}", audio_base64)
    return audio_base64

# Optimized handler with pipeline parallelization where possible
async def process_request(job_input):
    """Process the request with optimized async operations"""
    start_time = time.time()
    print("\n=== New Request Started ===")
    
    # Get input type
    input_type = job_input.get("type", "text")
    
    # Process input based on type
    if input_type == "text":
        text_input = job_input.get("text", "")
        print("Processing text input")
    else:
        print("Processing audio input...")
        audio_start = time.time()
        
        # Decode audio
        audio_base64 = job_input.get("audio", "")
        audio_bytes = base64.b64decode(audio_base64)
        
        # Transcribe audio
        text_input = await transcribe_audio(audio_bytes)
        print(f"Audio transcription took {time.time() - audio_start:.2f}s")
    
    # Generate LLM response
    llm_start = time.time()
    ai_response = await generate_llm_response(text_input)
    print(f"LLM response took {time.time() - llm_start:.2f}s")
    
    # Generate TTS response (start in parallel if possible)
    tts_start = time.time()
    print("Starting TTS generation...")
    audio_base64 = await generate_tts(ai_response)
    print(f"TTS generation took {time.time() - tts_start:.2f}s")
    
    print(f"Total request time: {time.time() - start_time:.2f}s")
    
    return {
        "user_input": {
            "type": input_type, 
            "text": text_input
        },
        "assistant_response": {
            "text": ai_response, 
            "audio": audio_base64
        }
    }

# RunPod handler needs to be synchronous, so we use asyncio to run the async code
def async_handler(job):
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(process_request(job["input"]))
        loop.close()
        return result
    except Exception as e:
        print(f"Error in handler: {str(e)}")
        return {"error": str(e)}

print("Starting server with optimized performance...")
print("Server ready!")

runpod.serverless.start({
    "handler": async_handler
})
