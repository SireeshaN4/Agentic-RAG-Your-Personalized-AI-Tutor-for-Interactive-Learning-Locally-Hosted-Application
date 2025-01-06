import speech_recognition as sr
import pyaudio

def list_microphones():
    """List all available microphones"""
    p = pyaudio.PyAudio()
    info = []
    
    print("\nAvailable Audio Devices:")
    print("-" * 50)
    
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info['maxInputChannels'] > 0:  # Only input devices
            print(f"Index: {i}")
            print(f"Name: {dev_info['name']}")
            print(f"Input channels: {dev_info['maxInputChannels']}")
            print(f"Default Sample Rate: {dev_info['defaultSampleRate']}")
            print("-" * 50)
            info.append(dev_info)
    
    p.terminate()
    return info

def test_specific_mic(device_index):
    """Test a specific microphone"""
    try:
        r = sr.Recognizer()
        mic = sr.Microphone(device_index=device_index)
        
        with mic as source:
            print(f"\nTesting microphone index {device_index}")
            print("Adjusting for ambient noise...")
            r.adjust_for_ambient_noise(source, duration=1)
            print("Say something...")
            audio = r.listen(source, timeout=5)
            print("Processing...")
            text = r.recognize_sphinx(audio)
            print(f"Recognized: {text}")
    except Exception as e:
        print(f"Error testing microphone: {e}")

if __name__ == "__main__":
    devices = list_microphones()
    if devices:
        test_specific_mic(devices[0]['index'])  # Test first available mic