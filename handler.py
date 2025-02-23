import runpod
import base64
import io
import asyncio
import os
import time
from groq import Groq
from openai import OpenAI

CHUNK_SIZE = 256 * 1024  # 256KB chunks

# Initialize clients
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

SYSTEM_PROMPT = """ # Primary Directive
...(your system prompt here)...
"""

async def process_audio(audio_base64):
    """Processes base64-encoded audio, transcribes it using Whisper, and returns text."""
    try:
        audio_bytes = base64.b64decode(audio_base64)
        audio_buffer = io.BytesIO(audio_bytes)

        translation = await asyncio.to_thread(
            lambda: groq_client.audio.translations.create(
                file=audio_buffer, 
                model="whisper-large-v3",
                response_format="json",
                temperature=0.0
            )
        )
        return translation.text
    except Exception as e:
        print(f"Audio processing error: {e}")
        return None

async def get_llm_response(text_input):
    """Generates response from LLM."""
    try:
        response = await asyncio.to_thread(
            lambda: groq_client.chat.completions.create(
                messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text_input}],
                model="llama-3.3-70b-versatile",
                temperature=0.5,
                max_tokens=2048
            )
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"LLM processing error: {e}")
        return None

async def generate_tts(audio_text):
    """Generates TTS audio from text and returns base64-encoded audio."""
    try:
        tts_audio = openai_client.audio.speech.create(
            model="tts-1", voice="alloy", input=audio_text
        ).iter_bytes(chunk_size=CHUNK_SIZE)

        audio_buffer = io.BytesIO()
        for chunk in tts_audio:
            audio_buffer.write(chunk)

        return base64.b64encode(audio_buffer.getvalue()).decode()
    except Exception as e:
        print(f"TTS processing error: {e}")
        return None

async def async_handler(job):
    try:
        start_time = time.time()
        job_input = job["input"]
        input_type = job_input["type"]

        text_input = await process_audio(job_input["audio"]) if input_type == "audio" else job_input["text"]
        if not text_input:
            return {"error": "Processing failed."}

        llm_response_task = asyncio.create_task(get_llm_response(text_input))
        llm_response = await llm_response_task
        if not llm_response:
            return {"error": "LLM response failed."}

        tts_audio_task = asyncio.create_task(generate_tts(llm_response))
        tts_audio_base64 = await tts_audio_task
        if not tts_audio_base64:
            return {"error": "TTS generation failed."}

        print(f"Total request time: {time.time() - start_time:.2f}s")
        return {"user_input": text_input, "assistant_response": {"text": llm_response, "audio": tts_audio_base64}}

    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

print("Starting server...")
runpod.serverless.start({"handler": async_handler})
