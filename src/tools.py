import random
import re

import pyaudio
# import wave
import tempfile
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import torch
import numpy as np


# --------- TOOL: Dice Roller --------- #
def roll(dice_expr: str) -> int:
    # Basic dice parser: 2d6+3 => [2,6,+3]
    match = re.fullmatch(r"(\d*)d(\d+)([+-]\d+)?", dice_expr.replace(" ", ""))
    if not match:
        raise ValueError(f"Invalid dice expression: {dice_expr}")
    num = int(match.group(1)) if match.group(1) else 1
    die = int(match.group(2))
    mod = int(match.group(3)) if match.group(3) else 0
    return sum(random.randint(1, die) for _ in range(num)) + mod


# TODO: Test this!
# --------- TOOL: Dice Roller --------- #
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