from audio_utils import VoiceInterface

if __name__ == "__main__":
    voice = VoiceInterface()
    print("Say something...")
    result = voice.listen()
    print(f"Final result: {result}")