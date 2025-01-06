import pyaudio
import wave
import time
import os

def test_chromeos_audio():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    RECORD_SECONDS = 5
    
    p = pyaudio.PyAudio()
    
    # List available devices
    print("\nAudio Devices on ChromeOS:")
    for i in range(p.get_device_count()):
        try:
            dev_info = p.get_device_info_by_index(i)
            print(f"\nDevice {i}:")
            print(f"Name: {dev_info['name']}")
        except Exception as e:
            print(f"Error getting device {i}: {e}")
            
    # Open stream correctly with all parameters
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

if __name__ == "__main__":
    test_chromeos_audio()