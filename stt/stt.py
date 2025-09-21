from typing import Literal
import whisper
import pyaudio
import wave
import tempfile
import webrtcvad
import keyboard as kb
import collections
import asyncio
import time

class STT:
    def __init__(self, model: Literal['tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small', 'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3', 'large', 'large-v3-turbo', 'turbo'] = "base", aggressive: int = 2, chunk_duration_ms: int = 30):
        self.rate = 16000
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk = int(self.rate * self.chunk_duration_ms / 1000)
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )
        self.vad = webrtcvad.Vad(aggressive)
        self.models = whisper.load_model(model)

    def _save_wav_temp(self, frames):
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))
        return temp_file.name

    async def _record_and_transcribe(self, frames):
        loop = asyncio.get_running_loop()
        filename = self._save_wav_temp(frames)
        result = await loop.run_in_executor(None, self.models.transcribe, filename)
        return result["text"]

    async def record_audio_sec_stt(self, duration=5):
        loop = asyncio.get_running_loop()
        frames = await loop.run_in_executor(None, lambda: [
            self.stream.read(self.chunk, exception_on_overflow=False) 
            for _ in range(int(self.rate / self.chunk * duration))
        ])
        return await self._record_and_transcribe(frames)

    async def record_audio_vad_stt(self):
        loop = asyncio.get_running_loop()
        ring_buffer = collections.deque(maxlen=int(self.rate / self.chunk * 0.5))
        frames = []
        triggered = False
        silence_chunks = 0
        max_silence_chunks = int(self.rate / self.chunk * 0.5)

        print("Listening for speech...")

        while True:
            data = await loop.run_in_executor(None, lambda: self.stream.read(self.chunk, exception_on_overflow=False))
            is_speech = self.vad.is_speech(data, self.rate)

            if not triggered:
                ring_buffer.append(data)
                if is_speech:
                    triggered = True
                    print("Speech started...")
                    frames.extend(ring_buffer)
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

            await asyncio.sleep(0)

        return await self._record_and_transcribe(frames)

    async def record_audio_keyboard_stt(self, key: str = "space"):
        loop = asyncio.get_running_loop()
        print(f"Press '{key}' to start recording, press again to stop.")
        await loop.run_in_executor(None, kb.wait, key)
        await asyncio.sleep(0.1)
        print("Recording...")
        frames = []

        def read_frames():
            while not kb.is_pressed(key):
                frames.append(self.stream.read(self.chunk, exception_on_overflow=False))

        await loop.run_in_executor(None, read_frames)
        print("Stopped recording.")
        return await self._record_and_transcribe(frames)

    async def stt(self, audio_file: str):
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(None, self.models.transcribe, audio_file)
        return result["text"]
