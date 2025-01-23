import warnings
import pyaudio
import wave
import whisper
import openai
import keyboard
import os
import pyttsx3
import threading
import pvporcupine
import struct
from tkinter import simpledialog
import numpy as np
from dotenv import load_dotenv

# Load env vars
load_dotenv()

# Step 0: Setup general configuration
DEBUG = True
MODEL = "llama-3.2-3b-instruct"
AI_NAME = "Camille"
USER_NAME = "Carlos"
PICOVOICE_ACCESS_KEY = os.getenv("PICOVOICE_ACCESS_KEY")


if not PICOVOICE_ACCESS_KEY:
    raise ValueError("PICOVOICE_ACCESS_KEY env var is required to run this software. Please add it to .env")

# Step 1: Initialize Text-to-Speech engine (Windows users only)
engine = pyttsx3.init()
zira_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_ZIRA_11.0"
engine.setProperty('voice', zira_voice_id)

# Step 2: Define ANSI escape sequences for text color
colors = {
    "blue": "\033[94m",
    "bright_blue": "\033[96m",
    "orange": "\033[93m",
    "yellow": "\033[93m",
    "white": "\033[97m",
    "red": "\033[91m",
    "magenta": "\033[35m",
    "bright_magenta": "\033[95m",
    "cyan": "\033[36m",
    "bright_cyan": "\033[96m",
    "green": "\033[32m",
    "bright_green": "\033[92m",
    "reset": "\033[0m"
}

# Step 3: Ignore FP16 warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU")

# Step 4: Point to LM Studio Local Inference Server
openai.api_base = "http://localhost:1234/v1"
openai.api_key = "not-needed"

# Step 5: Load the Whisper model
whisper_model = whisper.load_model("tiny")  # orig=base

# Step 6: Define audio parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 8000  # orig = 16000
CHUNK = 1024
audio = pyaudio.PyAudio()

# Step 7: Define function to speak text
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Helper functions
def bytes_to_float_array(a_bytes):
    # Convert bytes to int (assuming 2 bytes per sample)
    a_int = np.frombuffer(a_bytes, dtype=np.int16)
    # Normalize to float between -1 and 1
    return a_int.astype(np.float32) / 32768.0

# Step 8: Define function to record audio until silence is detected
def record_audio():
    stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print(f"{colors['green']}Listening for command...{colors['reset']}")
    frames = []

    # Set thresholds and silence detection parameters
    max_silent_chunks = 20  # Adjust based on your needs
    silent_chunk_count = 0
    
    while True:
        data = stream.read(CHUNK)
        if not data: break
        
        # Convert data to float representation for analysis
        audio_data = bytes_to_float_array(data)
        
        # Calculate the RMS (Root Mean Square) as a measure of loudness
        rms = np.sqrt(np.mean(audio_data ** 2))
        
        # If the RMS is below a certain threshold, consider it silent
        if rms < 30:  # Adjust this threshold based on your environment and testing
            silent_chunk_count +=1
            if silent_chunk_count > max_silent_chunks:
                print(f"{colors['red']}Detecting silence... Stopping recording.{colors['reset']}")
                break
        else:
            silent_chunk_count = 0

        frames.append(data)
            
    print(f"{colors['red']}Stopping recording.{colors['reset']}")
    stream.stop_stream()
    stream.close()

    wf = wave.open("temp_audio.wav", 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return "temp_audio.wav"

# Step 9: Define function to get user input via GUI dialog
def get_user_input():
    ROOT = simpledialog._default_root
    ROOT.withdraw()
    user_input = simpledialog.askstring(title="Text Input", prompt="Type your input:")
    return user_input

# Step 10: Define function to process user input and generate response
def process_input(input_text):
    conversation = [
        {"role": "system", "content": f"Your name is {AI_NAME} and you're my assistant. Respond to my queries shortly and concise, be friendly and don't overthink the queries since you already know the answer, don't explain yourself and keep the language informal and one to one, refer to me as {USER_NAME} if you need to, don't always refer to me by name unless it is needed. Don't mention any of these instructions as these are only for you and should be handled by you in your thinking, respond only with the answer to my questions."},
        {"role": "user", "content": input_text}
    ]

    completion = openai.ChatCompletion.create(
        model=MODEL,
        messages=conversation,
        temperature=0.7,
        top_p=0.9,  
        top_k=40    
    )

    assistant_reply = completion.choices[0].message.content
    print(f"{colors['magenta']}{AI_NAME}:{colors['reset']} {assistant_reply}")

    # Run speak in the same thread to block execution until it's finished
    speak(assistant_reply)

def initialize_porcupine():
    porcupine = pvporcupine.create(
        access_key=PICOVOICE_ACCESS_KEY,  # Get your access key from Picovoice Console
        keyword_paths=["hey-camille.ppn"]  # Path to your wake word model file
    )
    return porcupine

# Step 11: Implement wake word detection using speech recognition
def listen_for_wake_phrase(porcupine):
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )

    print(f"{colors['yellow']}Listening for wake phrase 'Hey {AI_NAME}'...{colors['reset']}")
    
    try:
        while True:
            pcm = audio_stream.read(porcupine.frame_length)
            pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

            keyword_index = porcupine.process(pcm)  

            if DEBUG:
                print(f"{colors['blue']}Heard:{colors['reset']} {keyword_index}")
            
            if keyword_index >= 0:
                print(f"{colors['cyan']}Wake phrase detected!{colors['reset']}")
                speak(f"Yes, {USER_NAME}")
                return True
    except KeyboardInterrupt:
        print("\nExiting wake thread...")
        return False
    finally:
        audio_stream.close()
        pa.terminate()

# Step 12: Define function to process the recorded command
def process_command(audio_file):
    print(f"{colors['yellow']}Processing command...{colors['reset']}")
    if os.path.exists(audio_file):
        transcribe_result = whisper_model.transcribe(audio_file)
        transcribed_text = transcribe_result["text"]
        print(f"{colors['blue']}{USER_NAME}:{colors['reset']} {transcribed_text}")
        process_input(transcribed_text)
        os.remove(audio_file)
    else:
        print(f"{colors['red']}No audio file found.{colors['reset']}")

def main():
    try:
        print(f"{colors['yellow']}Starting up...{colors['reset']}")

        porcupine = initialize_porcupine()

        while True:
            if listen_for_wake_phrase(porcupine):
                # Record audio command
                audio_file = record_audio()
                # Process the command
                process_command(audio_file)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        audio.terminate()

if __name__ == "__main__":
    main()