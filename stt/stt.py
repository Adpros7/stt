import whisper
import pyaudio
import wave
import tempfile
import webrtcvad
import keyboard as kb
import collections
import time


class STT:
    def __init__(self, model: str="base", aggressive: int=2, chunk_duration_ms: int=30):
        self.rate = 16000
        self.chunk_duration_ms = chunk_duration_ms  # pick 10, 20, or 30 ms
        self.chunk = int(self.rate * self.chunk_duration_ms /
                         1000)  # samples per frame
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=self.rate,
                                  input=True,
                                  frames_per_buffer=self.chunk)
        self.vad = webrtcvad.Vad(aggressive)  # 0=least aggressive, 3=most aggressive
        self.models = whisper.load_model(model)

    def _save_wav_temp(self, frames):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
        return temp_file.name

    def _record_and_transcribe(self, frames):
        filename = self._save_wav_temp(frames)
        result = self.models.transcribe(filename)
        return result["text"]

    def record_audio_sec_stt(self, duration=5):
        frames = [self.stream.read(self.chunk) for _ in range(
            int(self.rate / self.chunk * duration))]
        return self._record_and_transcribe(frames)

    def record_audio_vad_stt(self):
        ring_buffer = collections.deque(maxlen=int(
            self.rate / self.chunk * 0.5))  # 0.5s pre-speech buffer
        frames = []
        triggered = False
        silence_chunks = 0
        max_silence_chunks = int(
            self.rate / self.chunk * 0.5)  # 0.5s post-speech

        print("Listening for speech...")
        while True:
            data = self.stream.read(self.chunk)
            is_speech = self.vad.is_speech(data, self.rate)

            if not triggered:
                ring_buffer.append(data)
                if is_speech:
                    triggered = True
                    print("Speech started...")
                    frames.extend(ring_buffer)  # include pre-speech audio
                    ring_buffer.clear()
            else:
                frames.append(data)
                if not is_speech:
                    silence_chunks += 1
                    if silence_chunks > max_silence_chunks:
                        print("Speech ended.")
                        break
                else:
                    silence_chunks = 0

        return self._record_and_transcribe(frames)

    def record_audio_keyboard_stt(self, key: str = "space"):
        print(f"Press '{key}' to start recording, press again to stop.")
        kb.wait(key)
        time.sleep(1)
        print("Recording...")
        frames = []
        while not kb.is_pressed(key):
            frames.append(self.stream.read(self.chunk))
        print("Stopped recording.")
        return self._record_and_transcribe(frames)

    def stt(self, text: str):
        result = self.models.transcribe(audio=text)
        return result["text"]


if __name__ == "__main__":
    stt_instance = STT()
    print("Choose recording method: 'sec', 'vad', or 'keyboard'")
    method = input().strip().lower()
    if method == "sec":
        print(stt_instance.record_audio_sec_stt())
    elif method == "vad":
        print(stt_instance.record_audio_vad_stt())
    elif method == "keyboard":
        print(stt_instance.record_audio_keyboard_stt())
    else:
        print("Invalid method.")
