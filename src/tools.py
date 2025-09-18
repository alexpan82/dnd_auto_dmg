import random
import re

# import pyaudio
# import wave
import tempfile
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import numpy as np
import json
import math
from rapidfuzz import process, fuzz



# --------- TOOL: Dice Roller --------- #
def roll_dice_(dice_expr: str) -> int:
    """Rolls dice plus modifier in the following format:
    ndm+x where n is the number of dice, m is the number of sides the dice have, and x is the modifier
    Ex: 2d6+3 => Roll 2 6-sided die and then add 3 to the total [2,6,+3]

    Args:
        dice_expr: str in ndm+x format
    """
    match = re.fullmatch(r"(\d*)d(\d+)([+-]\d+)?", dice_expr.replace(" ", ""))
    if not match:
        raise ValueError(f"Invalid dice expression: {dice_expr}")
    num = int(match.group(1)) if match.group(1) else 1
    die = int(match.group(2))
    mod = int(match.group(3)) if match.group(3) else 0
    return sum(random.randint(1, die) for _ in range(num)) + mod

def roll_dice(num_dice: int, num_sides: int, modifier: int) -> int:
    """Rolls random set of dice and then adds modifier

    Args:
        num_dice: number of dice to roll
        num_sides: number of sides on dice
        modifier: add to the result after die roll
    """
    dice_rolls = [random.randint(1, num_sides) for _ in range(num_dice)]
    return sum(dice_rolls) + modifier

def multiply(a: int, b: int) -> int:
    """Multiply a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

# This will be a tool
def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b


# This will be a tool
def subtract(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a - b


def divide(a: int, b: int) -> int:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return math.ceil(a / b)

# Adding json cleanup
def extract_json(text:str) -> dict:
    """
    Extract JSON from a string by removing leading and trailing non-JSON characters.
    """
    # Find the first opening brace or bracket
    start_pattern = r'[{\[]'
    start_match = re.search(start_pattern, text)
    
    if not start_match:
        return None
    
    start_pos = start_match.start()
    start_char = text[start_pos]
    end_char = '}' if start_char == '{' else ']'
    
    # Count nested braces/brackets to find the matching closing one
    count = 0
    for i, char in enumerate(text[start_pos:], start_pos):
        if char == start_char:
            count += 1
        elif char == end_char:
            count -= 1
            if count == 0:
                cleaned_str = text[start_pos:i+1]
                try:
                    return json.loads(cleaned_str)
                except json.decoder.JSONDecodeError:
                    return None
                
    return None


def fuzzy_match(query:str, possible_matches:list[str]) -> tuple[str, float]:
    """
    Justification: players often truncate phrases / nouns or over-complicate the description
    Therefore we take the max b/t the partial_ratio and token_set_ratio
    """
    partial_ratios = [fuzz.partial_ratio(query, q) for q in possible_matches]
    token_set_ratio = [fuzz.token_set_ratio(query, q) for q in possible_matches]
    
    result_list = [(q, max(a, b)) for a, b, q in zip(partial_ratios, token_set_ratio, possible_matches)]
    result_list = sorted(result_list, key=lambda x: x[1], reverse=True)

    best_match = result_list[0]

    if best_match[1] <= 40:
        return((None, None))
    else:
        return best_match


# TODO: Test this!
def transcribe_voice_input(duration=5, sample_rate=16000, chunk_size=1024):
    """
    Captures microphone input and transcribes it using Whisper large-v3-turbo model.
    
    Args:
        duration (int): Recording duration in seconds (default: 5)
        sample_rate (int): Audio sample rate (default: 16000)
        chunk_size (int): Audio chunk size for recording (default: 1024)
    
    Returns:
        str: Transcribed text from the audio input
    """
    
    # Initialize the Whisper model and processor
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model_id = "openai/whisper-base"
    
    # Load model and processor
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    # Create pipeline
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        return_timestamps=True,
        torch_dtype=torch_dtype,
        device=device,
    )
    
    # Audio recording setup
    audio_format = pyaudio.paInt16
    channels = 1
    
    # Initialize PyAudio
    p = pyaudio.PyAudio()
    
    try:
        print(f"Recording for {duration} seconds...")
        
        # Open audio stream
        stream = p.open(
            format=audio_format,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk_size
        )
        
        frames = []
        
        # Record audio
        for i in range(0, int(sample_rate / chunk_size * duration)):
            data = stream.read(chunk_size)
            frames.append(data)
        
        print("Recording finished.")
        
        # Stop and close stream
        stream.stop_stream()
        stream.close()
        
        # Convert recorded data to numpy array
        audio_data = b''.join(frames)
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
        audio_np = audio_np / np.iinfo(np.int16).max  # Normalize to [-1, 1]
        
        # Transcribe using Whisper
        print("Transcribing audio...")
        result = pipe(audio_np, generate_kwargs={"language": "english"})
        
        return result["text"]
        
    except Exception as e:
        print(f"Error during recording or transcription: {e}")
        return None
        
    finally:
        # Clean up PyAudio
        p.terminate()