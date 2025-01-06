import pyttsx3
import speech_recognition as sr
from pocketsphinx import pocketsphinx
import pyaudio
import subprocess
import os
import time

class VoiceInterface:
    def __init__(self):
        try:
            self.recognizer = sr.Recognizer()
            self.mic = sr.Microphone(device_index=None)  # Let it auto-select device
            
            # Adjust noise levels
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
            
            # Test audio system
            subprocess.run(['espeak', '-a', '200', '-s', '150', 'Audio initialized'], check=True)
        except Exception as e:
            print(f"Error in VoiceInterface initialization: {e}")
            raise

    def listen(self):
        try:
            with self.mic as source:
                print("Listening...")
                # Reduced timeout for faster response
                audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=5)
                text = self.recognizer.recognize_sphinx(audio, 
                    language='en-US',
                    show_all=False)
                print(f"You said: {text}")
                return text
        except sr.UnknownValueError:
            print("Could not understand audio")
            return ""
        except Exception as e:
            print(f"Error in listen(): {e}")
            return ""

# Main interaction loop
if __name__ == "__main__":
    voice = VoiceInterface()
    print("Listening for your question...")
    
    while True:
        question = voice.listen()
        if not question:
            continue
        
        if 'goodbye' in question.lower():
            voice.speak("Goodbye!")
            print("Goodbye!")
            break

        try:
            # Generate audio response
            voice.speak(f"Answering: {question}")
            print(f"Assistant answered: {question}")
        except Exception as e:
            print(f"Error occurred: {str(e)}")
