import os
from groq import Groq
import time

class SpeechToText:
    def __init__(self, api_key):
        self.api_key = api_key
        if api_key:
            self.client = Groq(api_key=api_key)
        else:
            self.client = None

    def transcribe(self, audio_file_path):
        """
        Transcribes audio using Groq Whisper.
        audio_file_path: Path to the .mp3, .wav, or .m4a file.
        """
        if not self.client:
            return "Error: Groq API Key not provided.", 0
            
        start_time = time.time()
        
        try:
            with open(audio_file_path, "rb") as file:
                transcription = self.client.audio.transcriptions.create(
                    file=(os.path.basename(audio_file_path), file.read()),
                    model="whisper-large-v3",
                    response_format="text"
                )
            
            latency = (time.time() - start_time) * 1000
            return transcription, round(latency, 2)
        except Exception as e:
            return f"Transcription error: {str(e)}", 0

if __name__ == "__main__":
    # Test would require API key and file
    print("STT module ready.")
