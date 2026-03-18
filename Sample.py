"""
Voice Agent Script with Background Transcription
- Records user audio and transcribes in background for reduced latency
- Agent confirms order while transcription completes
- Generates summary immediately upon confirmation
"""

import os
import subprocess
import tempfile
import wave
import threading
import json
import time
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future
from openai import OpenAI
from pathlib import Path
from functools import wraps
from typing import Callable, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ============================================================
# LOGGING CONFIGURATION
# ============================================================

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Configure logging with UTF-8 encoding (fixes emoji on Windows)
import sys
log_filename = f"logs/voice_agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Create handlers with proper UTF-8 encoding
file_handler = logging.FileHandler(log_filename, encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', '%Y-%m-%d %H:%M:%S'))

# Console handler with UTF-8 (handles emojis on Windows)
console_handler = logging.StreamHandler(stream=sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', '%Y-%m-%d %H:%M:%S'))

# Reconfigure stdout for UTF-8 on Windows
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, console_handler]
)
logger = logging.getLogger(__name__)


class MetricsTracker:
    """Track and log timing metrics for all operations."""
    
    def __init__(self):
        self.metrics = []
        self.session_start = time.time()
    
    def log_metric(self, operation: str, duration: float, details: str = ""):
        """Log a single metric."""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "operation": operation,
            "duration_ms": round(duration * 1000, 2),
            "duration_s": round(duration, 3),
            "details": details
        }
        self.metrics.append(metric)
        logger.info(f"â±ï¸  {operation}: {metric['duration_ms']:.2f}ms ({metric['duration_s']:.3f}s) {details}")
    
    def get_summary(self) -> dict:
        """Get summary of all metrics."""
        if not self.metrics:
            return {}
        
        total_time = time.time() - self.session_start
        by_operation = {}
        
        for m in self.metrics:
            op = m["operation"]
            if op not in by_operation:
                by_operation[op] = []
            by_operation[op].append(m["duration_ms"])
        
        summary = {
            "total_session_time_s": round(total_time, 2),
            "total_operations": len(self.metrics),
            "by_operation": {}
        }
        
        for op, times in by_operation.items():
            summary["by_operation"][op] = {
                "count": len(times),
                "total_ms": round(sum(times), 2),
                "avg_ms": round(sum(times) / len(times), 2),
                "min_ms": round(min(times), 2),
                "max_ms": round(max(times), 2)
            }
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of metrics."""
        summary = self.get_summary()
        if not summary:
            return
        
        logger.info("\n" + "=" * 60)
        logger.info("ðŸ“Š PERFORMANCE METRICS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Session Time: {summary['total_session_time_s']}s")
        logger.info(f"Total Operations: {summary['total_operations']}")
        logger.info("-" * 60)
        
        for op, stats in summary["by_operation"].items():
            logger.info(f"\n  {op}:")
            logger.info(f"    Count: {stats['count']}")
            logger.info(f"    Total: {stats['total_ms']:.2f}ms")
            logger.info(f"    Avg:   {stats['avg_ms']:.2f}ms")
            logger.info(f"    Min:   {stats['min_ms']:.2f}ms")
            logger.info(f"    Max:   {stats['max_ms']:.2f}ms")
        
        logger.info("\n" + "=" * 60)


# Global metrics tracker
metrics = MetricsTracker()


class ConversationMetrics:
    """
    Track conversation-specific timing metrics:
    - User speaking duration
    - Agent response generation time
    - Time from end of speech to TTS playback (latency)
    """
    
    def __init__(self):
        self.turns = []
        self.current_turn = {}
        self.session_start = time.time()
    
    def start_user_speaking(self):
        """Mark when user starts speaking."""
        self.current_turn = {
            "turn_number": len(self.turns) + 1,
            "user_speech_start": time.time()
        }
        logger.info("ðŸŽ¤ [TIMING] User started speaking...")
    
    def end_user_speaking(self):
        """Mark when user stops speaking."""
        self.current_turn["user_speech_end"] = time.time()
        self.current_turn["user_speaking_duration"] = (
            self.current_turn["user_speech_end"] - self.current_turn["user_speech_start"]
        )
        logger.info(f"ðŸŽ¤ [TIMING] User spoke for: {self.current_turn['user_speaking_duration']*1000:.2f}ms")
    
    def start_agent_processing(self):
        """Mark when agent starts processing (transcription + LLM)."""
        self.current_turn["agent_processing_start"] = time.time()
        logger.info("ðŸ¤– [TIMING] Agent processing started...")
    
    def end_agent_processing(self):
        """Mark when agent finishes processing."""
        self.current_turn["agent_processing_end"] = time.time()
        self.current_turn["agent_response_generation_time"] = (
            self.current_turn["agent_processing_end"] - self.current_turn["agent_processing_start"]
        )
        logger.info(f"ðŸ¤– [TIMING] Agent response generated in: {self.current_turn['agent_response_generation_time']*1000:.2f}ms")
    
    def start_tts(self):
        """Mark when TTS starts."""
        self.current_turn["tts_start"] = time.time()
        # Calculate latency from end of user speech to TTS start
        if "user_speech_end" in self.current_turn:
            self.current_turn["speech_to_tts_latency"] = (
                self.current_turn["tts_start"] - self.current_turn["user_speech_end"]
            )
            logger.info(f"ðŸ”Š [TIMING] Time from user speech end to TTS start: {self.current_turn['speech_to_tts_latency']*1000:.2f}ms")
    
    def end_tts_playback(self):
        """Mark when TTS playback completes."""
        self.current_turn["tts_end"] = time.time()
        if "tts_start" in self.current_turn:
            self.current_turn["tts_playback_duration"] = (
                self.current_turn["tts_end"] - self.current_turn["tts_start"]
            )
        
        # Save this turn and prepare for next
        self.turns.append(self.current_turn.copy())
        self.current_turn = {}
    
    def print_conversation_summary(self):
        """Print detailed conversation timing summary."""
        if not self.turns:
            return
        
        total_session = time.time() - self.session_start
        
        logger.info("\n" + "=" * 70)
        logger.info("ðŸ—£ï¸  CONVERSATION TIMING METRICS")
        logger.info("=" * 70)
        logger.info(f"Total Conversation Duration: {total_session:.2f}s")
        logger.info(f"Number of Turns: {len(self.turns)}")
        logger.info("-" * 70)
        
        # Aggregate metrics
        total_user_speaking = 0
        total_agent_processing = 0
        total_tts_latency = 0
        total_tts_playback = 0
        
        for turn in self.turns:
            turn_num = turn.get("turn_number", "?")
            
            user_dur = turn.get("user_speaking_duration", 0)
            agent_dur = turn.get("agent_response_generation_time", 0)
            tts_latency = turn.get("speech_to_tts_latency", 0)
            tts_playback = turn.get("tts_playback_duration", 0)
            
            total_user_speaking += user_dur
            total_agent_processing += agent_dur
            total_tts_latency += tts_latency
            total_tts_playback += tts_playback
            
            logger.info(f"\n  Turn {turn_num}:")
            logger.info(f"    ðŸ‘¤ User Speaking Time:      {user_dur*1000:>8.2f}ms ({user_dur:.2f}s)")
            logger.info(f"    ðŸ¤– Agent Response Time:     {agent_dur*1000:>8.2f}ms ({agent_dur:.2f}s)")
            logger.info(f"    âš¡ Speechâ†’TTS Latency:      {tts_latency*1000:>8.2f}ms ({tts_latency:.2f}s)")
            logger.info(f"    ðŸ”Š TTS Playback Duration:   {tts_playback*1000:>8.2f}ms ({tts_playback:.2f}s)")
        
        logger.info("\n" + "-" * 70)
        logger.info("  TOTALS & AVERAGES:")
        logger.info(f"    ðŸ‘¤ Total User Speaking:     {total_user_speaking*1000:>8.2f}ms ({total_user_speaking:.2f}s)")
        logger.info(f"    ðŸ¤– Total Agent Processing:  {total_agent_processing*1000:>8.2f}ms ({total_agent_processing:.2f}s)")
        logger.info(f"    âš¡ Total TTS Latency:       {total_tts_latency*1000:>8.2f}ms ({total_tts_latency:.2f}s)")
        logger.info(f"    ðŸ”Š Total TTS Playback:      {total_tts_playback*1000:>8.2f}ms ({total_tts_playback:.2f}s)")
        
        if len(self.turns) > 0:
            logger.info(f"\n    ðŸ“Š Avg User Speaking:        {(total_user_speaking/len(self.turns))*1000:>8.2f}ms")
            logger.info(f"    ðŸ“Š Avg Agent Response:      {(total_agent_processing/len(self.turns))*1000:>8.2f}ms")
            logger.info(f"    ðŸ“Š Avg Speechâ†’TTS Latency:  {(total_tts_latency/len(self.turns))*1000:>8.2f}ms")
        
        logger.info("\n" + "=" * 70)


# Global conversation metrics tracker
convo_metrics = ConversationMetrics()


def timed(operation_name: str = None):
    """Decorator to automatically time and log function execution."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            op_name = operation_name or func.__name__
            start_time = time.time()
            logger.info(f"â–¶ï¸  Starting: {op_name}")
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                metrics.log_metric(op_name, duration, "âœ… Success")
                return result
            except Exception as e:
                duration = time.time() - start_time
                metrics.log_metric(op_name, duration, f"âŒ Error: {str(e)}")
                raise
        
        return wrapper
    return decorator


# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set. Please add it to your .env file.")

# Initialize the OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Thread pool for background tasks
executor = ThreadPoolExecutor(max_workers=2)


# ============================================================
# AUDIO RECORDING FUNCTIONS
# ============================================================

@timed("Recording (Fixed Duration)")
def record_from_microphone(duration: int = 10, sample_rate: int = 16000) -> str:
    """Record audio from the microphone for a fixed duration."""
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        raise ImportError(
            "sounddevice and numpy are required.\n"
            "Install: pip install sounddevice numpy"
        )
    
    print(f"\nðŸŽ¤ Recording for {duration} seconds... Speak now!")
    
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype=np.int16
    )
    sd.wait()
    
    print("âœ… Recording complete!")
    
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    with wave.open(temp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    
    return temp_path


@timed("Recording (Until Silence)")
def record_until_silence(silence_threshold: float = 500, silence_duration: float = 0.6,
                         max_duration: int = 30, sample_rate: int = 16000) -> str:
    """Record audio until silence is detected."""
    try:
        import sounddevice as sd
        import numpy as np
    except ImportError:
        raise ImportError("sounddevice and numpy required: pip install sounddevice numpy")
    
    print("\nðŸŽ¤ Listening... (will stop after silence)")
    
    chunk_duration = 0.1
    chunk_samples = int(sample_rate * chunk_duration)
    silence_chunks_needed = int(silence_duration / chunk_duration)
    max_chunks = int(max_duration / chunk_duration)
    
    audio_chunks = []
    silence_count = 0
    has_speech = False
    
    def audio_callback(indata, frames, time, status):
        nonlocal silence_count, has_speech
        audio_chunks.append(indata.copy())
        
        rms = np.sqrt(np.mean(indata.astype(np.float32)**2))
        if rms > silence_threshold:
            has_speech = True
            silence_count = 0
        elif has_speech:
            silence_count += 1
    
    with sd.InputStream(samplerate=sample_rate, channels=1, dtype=np.int16,
                        blocksize=chunk_samples, callback=audio_callback):
        while (silence_count < silence_chunks_needed or not has_speech) and len(audio_chunks) < max_chunks:
            sd.sleep(int(chunk_duration * 1000))
    
    print("âœ… Recording complete!")
    
    if not audio_chunks:
        raise ValueError("No audio recorded")
    
    audio_data = np.concatenate(audio_chunks)
    
    temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    temp_path = temp_file.name
    temp_file.close()
    
    with wave.open(temp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())
    
    return temp_path


# ============================================================
# TRANSCRIPTION FUNCTIONS (WITH BACKGROUND SUPPORT)
# ============================================================

@timed("Whisper Transcription")
def transcribe_audio(audio_file_path: str) -> str:
    """Transcribe audio file using OpenAI Whisper API."""
    logger.info("ðŸ“ Transcribing audio...")
    
    with open(audio_file_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="text"
        )
    
    print(f"   You said: \"{transcription}\"")
    return transcription


def transcribe_audio_async(audio_file_path: str) -> Future:
    """Start transcription in background thread - returns immediately."""
    logger.info("â³ Starting BACKGROUND transcription (non-blocking)...")
    return executor.submit(transcribe_audio, audio_file_path)


# ============================================================
# AI PROCESSING FUNCTIONS
# ============================================================

# Filler words to ignore
FILLER_WORDS = {"um", "uh", "uhh", "umm", "hmm", "hm", "like", "you know", "er", "ah", "eh"}

# Signals user may continue speaking (don't interrupt)
CONTINUATION_SIGNALS = {
    "and", "also", "then", "wait", "actually", "plus", "with", "or", 
    "but", "so", "let me", "let me see", "hold on", "one more", "oh and",
    "i also", "can i also", "and also", "as well", "too"
}

# Signals user is finished speaking
COMPLETION_SIGNALS = {
    "that's it", "that's all", "done", "that is all", "that is it",
    "nothing else", "that's everything", "i'm done", "that will be all",
    "please", "thank you", "thanks", "yes", "correct", "confirm"
}


def clean_transcription(text: str) -> str:
    """
    Remove filler words from transcription.
    
    Args:
        text: Raw transcription text
    
    Returns:
        Cleaned text with filler words removed
    """
    words = text.lower().split()
    cleaned = [w for w in words if w not in FILLER_WORDS]
    return " ".join(cleaned)


def is_utterance_complete(text: str) -> dict:
    """
    Determine if the user has finished their utterance or is likely to continue.
    
    Returns:
        dict with:
            - is_complete: bool - whether utterance appears complete
            - confidence: float - confidence level (0-1)
            - reason: str - explanation
            - should_wait: bool - whether to wait for more input
    """
    text_lower = text.lower().strip()
    text_clean = clean_transcription(text)
    
    # Check for explicit completion signals
    for signal in COMPLETION_SIGNALS:
        if signal in text_lower:
            return {
                "is_complete": True,
                "confidence": 0.95,
                "reason": f"Completion signal detected: '{signal}'",
                "should_wait": False
            }
    
    # Check for continuation signals at the end
    words = text_clean.split()
    if words:
        last_word = words[-1].rstrip('.,!?')
        last_two = ' '.join(words[-2:]) if len(words) >= 2 else last_word
        
        for signal in CONTINUATION_SIGNALS:
            if last_word == signal or last_two.endswith(signal):
                return {
                    "is_complete": False,
                    "confidence": 0.85,
                    "reason": f"Continuation signal at end: '{signal}'",
                    "should_wait": True
                }
    
    # Check for trailing "and..." patterns (incomplete list)
    if text_lower.rstrip().endswith(('and', 'and...', 'and uh', 'and um')):
        return {
            "is_complete": False,
            "confidence": 0.9,
            "reason": "Incomplete list (trailing 'and')",
            "should_wait": True
        }
    
    # Check for self-correction patterns
    correction_patterns = ['actually', 'no wait', 'sorry', 'i mean', 'not that', 'change that']
    for pattern in correction_patterns:
        if pattern in text_lower:
            # If correction is at the end, wait for more
            if text_lower.rstrip().endswith(pattern) or text_lower.rstrip().endswith(pattern + ','):
                return {
                    "is_complete": False,
                    "confidence": 0.8,
                    "reason": f"Self-correction detected: '{pattern}'",
                    "should_wait": True
                }
    
    # Check for very short utterances (likely incomplete)
    if len(words) <= 2 and not any(s in text_lower for s in COMPLETION_SIGNALS):
        return {
            "is_complete": False,
            "confidence": 0.6,
            "reason": "Very short utterance, may be incomplete",
            "should_wait": True
        }
    
    # Default: assume complete if none of the above
    return {
        "is_complete": True,
        "confidence": 0.7,
        "reason": "No continuation signals detected",
        "should_wait": False
    }

@timed("GPT-5 Summary Generation")
def generate_summary(transcription: str) -> str:
    """Generate a summary using GPT-5 mini with streaming for reduced latency."""
    logger.info("Generating summary with GPT-5 mini (streaming)...")
    
    # Use streaming to get first tokens faster (~300ms vs 7s)
    stream = client.responses.create(
        model="gpt-5-mini",
        instructions=(
            "You are a helpful assistant that creates concise summaries. "
            "Generate a clear summary capturing the key points."
        ),
        input=f"Please summarize:\n\n{transcription}",
        stream=True,
    )
    
    # Collect streamed response - Responses API uses different event structure
    full_response = ""
    final_response = None
    for event in stream:
        # Text delta events have type 'response.output_text.delta'
        if hasattr(event, 'type') and event.type == 'response.output_text.delta':
            full_response += event.delta
        # Completed event has the full response
        elif hasattr(event, 'type') and event.type == 'response.completed':
            final_response = event.response
    
    # Get text from final response if streaming didn't capture it
    if not full_response and final_response:
        if hasattr(final_response, 'output_text'):
            full_response = final_response.output_text
        elif hasattr(final_response, 'output') and final_response.output:
            for item in final_response.output:
                if hasattr(item, 'content'):
                    for content in item.content:
                        if hasattr(content, 'text'):
                            full_response += content.text
    
    return full_response


@timed("GPT-5 Intent Processing")
def process_order_intent(transcription: str, order_context: str) -> dict:
    """
    Process user intent with conversational AI rules.
    
    - Ignores filler words
    - Detects incomplete utterances
    - Avoids interrupting the user
    - Uses streaming for reduced latency
    """
    logger.info("ðŸ§  Processing order intent (streaming)...")
    
    # Step 1: Check if utterance is complete
    utterance_status = is_utterance_complete(transcription)
    logger.info(f"ðŸ’¡ Utterance analysis: {utterance_status['reason']} (confidence: {utterance_status['confidence']:.0%})")
    
    # If user seems to be continuing, don't respond yet
    if utterance_status["should_wait"] and utterance_status["confidence"] >= 0.7:
        logger.info("â¸ï¸  User may continue speaking - waiting...")
        return {
            "reply_to_user": "",  # Empty = don't speak
            "updated_order_list": order_context,
            "is_complete": False,
            "needs_confirmation": False,
            "should_wait": True,
            "wait_reason": utterance_status["reason"]
        }
    
    # Step 2: Clean the transcription
    clean_text = clean_transcription(transcription)
    logger.info(f"Cleaned transcription: '{clean_text}'")
    
    # Step 3: Process with LLM - use NON-streaming for reliable JSON output
    response = client.responses.create(
        model="gpt-5-mini",
        instructions="""You are a fast-food ordering assistant. Extract order items from customer speech.

ALWAYS respond with ONLY valid JSON in this exact format:
{"reply_to_user": "your response", "updated_order_list": "item1, item2", "is_complete": false, "needs_confirmation": false}

Rules:
- Extract food items mentioned (burgers, fries, drinks, etc.)
- Confirm what you heard
- Keep replies SHORT (under 15 words)
- Set is_complete=true ONLY if user says "yes", "confirm", "that's all", "done"
- updated_order_list should contain ALL items ordered so far

Examples:
User: "I want a burger" -> {"reply_to_user": "One burger. Anything else?", "updated_order_list": "1 burger", "is_complete": false, "needs_confirmation": false}
User: "Add fries" -> {"reply_to_user": "Added fries. Anything else?", "updated_order_list": "1 burger, 1 fries", "is_complete": false, "needs_confirmation": false}
User: "That's all" -> {"reply_to_user": "Order confirmed: 1 burger, 1 fries. Thank you!", "updated_order_list": "1 burger, 1 fries", "is_complete": true, "needs_confirmation": false}""",
        input=f"Current order: {order_context if order_context else 'empty'}\nCustomer said: {clean_text}",
        stream=False,
    )
    
    # Get response text from non-streaming response
    text = ""
    if hasattr(response, 'output_text'):
        text = response.output_text
    elif hasattr(response, 'output') and response.output:
        for item in response.output:
            if hasattr(item, 'content'):
                for content in item.content:
                    if hasattr(content, 'text'):
                        text += content.text
    
    logger.info(f"Raw LLM response: {text[:500]}")
    
    try:
        # Try to extract JSON from response
        if "{" in text and "}" in text:
            json_str = text[text.find("{"):text.rfind("}")+1]
            logger.info(f"Extracted JSON: {json_str[:300]}")
            result = json.loads(json_str)
            result["should_wait"] = False
            logger.info(f"Parsed intent: reply='{result.get('reply_to_user', '')[:50]}', order='{result.get('updated_order_list', '')}'")
            return result
        else:
            logger.warning(f"No JSON found in LLM response")
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        logger.error(f"Failed to parse: {text[:200]}")
    except Exception as e:
        logger.error(f"Unexpected error parsing LLM response: {e}")
    
    # Fallback: if no JSON but text looks like a real response, use it
    if text and "{" not in text:
        logger.info(f"Using plain text response: {text[:100]}")
        return {
            "reply_to_user": text,
            "updated_order_list": order_context,
            "is_complete": False,
            "needs_confirmation": False,
            "should_wait": False
        }
    
    logger.warning("No valid response from LLM, using fallback")
    return {
        "reply_to_user": "I didn't catch that. Could you repeat?",
        "updated_order_list": order_context,
        "is_complete": False,
        "needs_confirmation": False,
        "should_wait": False
    }


# ============================================================
# TEXT-TO-SPEECH & PLAYBACK
# ============================================================

@timed("Text-to-Speech")
def text_to_speech(text: str, output_file_path: str = None) -> str:
    """Convert text to speech using OpenAI TTS API."""
    logger.info(f"ðŸ”Š Converting to speech: {text[:50]}...")
    if output_file_path is None:
        temp_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        output_file_path = temp_file.name
        temp_file.close()
    
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text
    )
    
    response.stream_to_file(Path(output_file_path))
    return output_file_path


@timed("Audio Playback")
def play_audio(audio_file_path: str):
    """Play audio file using sounddevice and soundfile."""
    try:
        import sounddevice as sd
        import soundfile as sf
        
        data, sample_rate = sf.read(audio_file_path)
        sd.play(data, sample_rate)
        sd.wait()
    except ImportError:
        logger.warning("Install for playback: pip install sounddevice soundfile")


def speak(text: str):
    """Convenience function: TTS + play + cleanup."""
    audio_path = text_to_speech(text)
    play_audio(audio_path)
    try:
        os.remove(audio_path)
    except:
        pass


# ============================================================
# MAIN VOICE AGENT WITH BACKGROUND TRANSCRIPTION
# ============================================================

def start_voice_agent_with_background_transcription():
    """
    Voice agent with background transcription for reduced latency.
    
    Flow:
    1. User speaks their order
    2. Recording completes -> transcription starts in BACKGROUND
    3. Agent immediately gives acknowledgment (while transcription runs)
    4. When user confirms, summary generates instantly (transcription already done)
    """
    global convo_metrics
    convo_metrics = ConversationMetrics()  # Reset for new session
    
    print("=" * 60)
    print("  Voice Agent with Background Transcription")
    print("  (Reduced latency mode)")
    print("=" * 60)
    print("\nðŸŽ¯ Speak your order. Agent confirms while processing.\n")
    
    speak("Hello! I'm ready to take your order. Please speak now.")
    
    order_context = ""
    is_order_complete = False
    accumulated_transcription = ""  # Buffer for incomplete utterances
    
    while not is_order_complete:
        # Step 1: Record user speech (with timing)
        convo_metrics.start_user_speaking()
        audio_path = record_until_silence()
        convo_metrics.end_user_speaking()
        
        # Step 2: Start agent processing timing
        convo_metrics.start_agent_processing()
        
        # Step 3: Start transcription in BACKGROUND (returns immediately!)
        transcription_future = transcribe_audio_async(audio_path)
        
        # Step 4: Get transcription result
        transcription = transcription_future.result()
        
        # Accumulate transcription if user was continuing
        if accumulated_transcription:
            transcription = accumulated_transcription + " " + transcription
            accumulated_transcription = ""
            logger.info(f"ðŸ“ Combined transcription: '{transcription}'")
        
        # Step 5: Process intent (includes utterance completeness check)
        intent = process_order_intent(transcription, order_context)
        
        # Step 6: Check if we should wait for more input (don't interrupt user)
        if intent.get("should_wait", False):
            logger.info(f"â¸ï¸  Waiting for user to continue... ({intent.get('wait_reason', 'incomplete utterance')})")
            accumulated_transcription = transcription  # Save for next turn
            convo_metrics.end_agent_processing()
            
            # Cleanup current audio but don't respond
            try:
                os.remove(audio_path)
            except:
                pass
            continue  # Go back to listening
        
        order_context = intent["updated_order_list"]
        
        # End agent processing timing
        convo_metrics.end_agent_processing()
        
        print(f"\nðŸ“‹ Order so far: {order_context}")
        print(f"ðŸ—£ï¸  Agent: {intent['reply_to_user']}")
        
        # Step 7: Speak response only if there's something to say
        if intent['reply_to_user']:
            convo_metrics.start_tts()
            speak(intent['reply_to_user'])
            convo_metrics.end_tts_playback()
        
        # Step 8: Handle confirmation flow
        if intent.get("needs_confirmation", False):
            print("\nâ³ Waiting for confirmation...")
            
            # Record user's confirmation (with timing)
            convo_metrics.start_user_speaking()
            confirm_audio_path = record_until_silence()
            convo_metrics.end_user_speaking()
            
            # Start agent processing
            convo_metrics.start_agent_processing()
            
            # Start transcription in background
            confirm_future = transcribe_audio_async(confirm_audio_path)
            
            # Quick ack while processing
            speak("One moment...")
            
            # Get confirmation text
            confirm_text = confirm_future.result()
            
            # Quick check if confirmed
            is_confirmed = any(word in confirm_text.lower() 
                             for word in ["yes", "correct", "confirm", "right", "sure", "okay", "ok", "yep", "yeah"])
            
            convo_metrics.end_agent_processing()
            
            if is_confirmed:
                is_order_complete = True
                print("\nâœ… ORDER CONFIRMED!")
                
                # Generate summary IMMEDIATELY (no wait - transcription already done)
                summary = generate_summary(f"Order confirmed: {order_context}")
                print(f"\nðŸ“‹ Summary: {summary}")
                
                convo_metrics.start_tts()
                speak(f"Perfect! Your order is confirmed. {summary}")
                convo_metrics.end_tts_playback()
            else:
                convo_metrics.start_tts()
                speak("No problem. What would you like to change?")
                convo_metrics.end_tts_playback()
            
            # Cleanup
            try:
                os.remove(confirm_audio_path)
            except:
                pass
        
        # Check if order complete from intent
        if intent.get("is_complete", False) and not is_order_complete:
            is_order_complete = True
            summary = generate_summary(f"Order: {order_context}")
            speak(f"Order complete! {summary}")
        
        # Cleanup
        try:
            os.remove(audio_path)
        except:
            pass
    
    logger.info("\n" + "=" * 60)
    logger.info(f"  âœ… Final Order: {order_context}")
    logger.info("=" * 60)
    
    # Print conversation timing metrics
    convo_metrics.print_conversation_summary()
    
    # Print performance metrics summary
    metrics.print_summary()
    
    return order_context


def start_simple_voice_agent():
    """Simple mode: Record once, transcribe, summarize, speak back."""
    logger.info("=" * 50)
    logger.info("  Simple Voice Agent (Single Recording)")
    logger.info("=" * 50)
    
    global metrics, convo_metrics
    metrics = MetricsTracker()  # Reset metrics for new session
    convo_metrics = ConversationMetrics()  # Reset conversation metrics
    
    # Track user speaking
    convo_metrics.start_user_speaking()
    audio_path = record_from_microphone(duration=10)
    convo_metrics.end_user_speaking()
    
    # Track agent processing
    convo_metrics.start_agent_processing()
    
    # Background transcription + quick ack
    transcription_future = transcribe_audio_async(audio_path)
    speak("Processing your message...")
    
    transcription = transcription_future.result()
    print(f"\nðŸ“ You said: \"{transcription}\"")
    
    summary = generate_summary(transcription)
    convo_metrics.end_agent_processing()
    
    print(f"\nðŸ“‹ Summary: {summary}")
    
    # Track TTS
    convo_metrics.start_tts()
    speak(summary)
    convo_metrics.end_tts_playback()
    
    try:
        os.remove(audio_path)
    except:
        pass
    
    # Print conversation timing metrics
    convo_metrics.print_conversation_summary()
    
    # Print performance metrics summary
    metrics.print_summary()


def process_audio_file(input_audio_path: str, output_audio_path: str = "summary_output.mp3"):
    """Process an audio file through the pipeline."""
    global metrics
    metrics = MetricsTracker()  # Reset metrics for new session
    
    logger.info("=" * 50)
    logger.info("  Processing Audio File")
    logger.info("=" * 50)
    
    transcription = transcribe_audio(input_audio_path)
    logger.info(f"\n--- Transcription ---\n{transcription}\n")
    
    summary = generate_summary(transcription)
    logger.info(f"--- Summary ---\n{summary}\n")
    
    output_path = text_to_speech(summary, output_audio_path)
    play_audio(output_path)
    
    # Print performance metrics summary
    metrics.print_summary()
    
    return {"transcription": transcription, "summary": summary}


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    logger.info("\n" + "=" * 60)
    logger.info("       ðŸŽ™ï¸  Voice Agent - Audio Summarizer")
    logger.info(f"       ðŸ“ Log file: {log_filename}")
    logger.info("=" * 60)
    print("\nChoose mode:")
    print("  1. Process audio file (01.wav)")
    print("  2. Simple voice recording (speak once)")
    print("  3. Interactive agent with BACKGROUND TRANSCRIPTION âš¡")
    print()
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice == "2":
        start_simple_voice_agent()
    elif choice == "3":
        metrics = MetricsTracker()  # Reset metrics for new session
        start_voice_agent_with_background_transcription()
    else:
        INPUT_FILE = "01.wav"
        if os.path.exists(INPUT_FILE):
            process_audio_file(INPUT_FILE, "summary_output.mp3")
        else:
            print(f"Error: {INPUT_FILE} not found")
