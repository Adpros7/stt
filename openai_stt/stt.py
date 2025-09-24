import re
import threading
from typing import Callable, Literal
import os
import time
from unittest import result
import wave
import tempfile
import collections
import warnings
import logging
import pyaudio
import webrtcvad
import keyboard as kb
from syntaxmod.general import wait_until
from concurrent.futures import Future, ThreadPoolExecutor

# Quiet the spam
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
for name in ["whisper", "transformers", "numba"]:
    logging.getLogger(name).setLevel(logging.ERROR)

# Try faster-whisper first
_USE_FASTER = True
try:
    from faster_whisper import WhisperModel as FWModel # type: ignore
except Exception:
    _USE_FASTER = False
    import whisper
    try:
        import torch
        _HAS_CUDA = torch.cuda.is_available()
    except Exception:
        _HAS_CUDA = False


class STT:
    def __init__(
        self,
        model: Literal[
            'tiny.en', 'tiny', 'base.en', 'base', 'small.en', 'small',
            'medium.en', 'medium', 'large-v1', 'large-v2', 'large-v3',
            'large', 'large-v3-turbo', 'turbo'
        ] = "base",
        aggressive: int = 2,
        chunk_duration_ms: int = 30,
        preroll_ms: int = 300,
        tail_silence_ms: int = 600,
        min_record_ms: int = 400
    ):
        assert chunk_duration_ms in (10, 20, 30)
        self.rate = 16000
        self.chunk_ms = chunk_duration_ms
        self.chunk = int(self.rate * self.chunk_ms / 1000)
        self.preroll_chunks = max(1, int(preroll_ms / self.chunk_ms))
        self.tail_silence_chunks = max(1, int(tail_silence_ms / self.chunk_ms))
        self.min_record_chunks = max(1, int(min_record_ms / self.chunk_ms))

        # mic
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk
        )

        # vad
        self.vad = webrtcvad.Vad(aggressive)

        # model
        self.backend = "faster" if _USE_FASTER else "whisper"
        if self.backend == "faster":
            self.model = FWModel(
                model,
                device="cpu",
                compute_type="int8",
                cpu_threads=max(2, os.cpu_count() or 4),
                download_root=os.path.join(
                    tempfile.gettempdir(), "whisper_models")
            )
        else:
            device = "cuda" if (_HAS_CUDA) else "cpu"
            self.model = whisper.load_model(
                model,
                download_root=os.path.join(
                    tempfile.gettempdir(), "whisper_models"),
                device=device
            )

    def close(self):
        try:
            if self.stream.is_active():
                self.stream.stop_stream()
        finally:
            self.stream.close()
            self.p.terminate()

    # ---------- internals ----------

    def _save_wav_temp(self, frames):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        with wave.open(tmp.name, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(self.p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(self.rate)
            wf.writeframes(b"".join(frames))
        return tmp.name

    def _fw_transcribe_file(self, filename: str) -> str:
        # Some versions support progress_bar, some don't. Try then fall back.
        try:
            segments, _ = self.model.transcribe(
                filename,
                beam_size=1,
                temperature=0.0,
                vad_filter=False,
                language="en",
            )
        except TypeError:
            segments, _ = self.model.transcribe(
                filename,
                beam_size=1,
                temperature=0.0,
                vad_filter=False,
                language="en"
            )
        return "".join(seg.text for seg in segments).strip() # type: ignore

    def _whisper_transcribe_file(self, filename: str) -> str:
        # Only use options the real whisper accepts
        result = self.model.transcribe(
            filename,
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=False,
            language="en"
        )
        return result.get("text", "").strip() # type: ignore

    def _transcribe_frames(self, frames) -> str:
        if not frames:
            return ""
        filename = self._save_wav_temp(frames)
        if self.backend == "faster":
            return self._fw_transcribe_file(filename)
        else:
            return self._whisper_transcribe_file(filename)

    # ---------- public APIs ----------

    def record_for_seconds(self, duration=5, log: bool = False) -> str:
        if log: print(f"Recording for {duration} seconds...")
        total_chunks = max(1, int((duration * 1000) / self.chunk_ms))
        frames = [
            self.stream.read(self.chunk, exception_on_overflow=False)
            for _ in range(total_chunks)
        ]
        print("Done.")
        return self._transcribe_frames(frames)

    def record_with_keyboard(self, key="space", log: bool = False) -> str:
        if log: print(f"Press {key} to start. Release to begin recording.")
        kb.wait(key)
        while kb.is_pressed(key):
            time.sleep(0.01)

        if log: print("Recording... Press key again to stop.")
        frames, recorded = [], 0
        while True:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            frames.append(data)
            recorded += 1
            if recorded < self.min_record_chunks:
                continue
            if kb.is_pressed(key):
                while kb.is_pressed(key):
                    time.sleep(0.01)
                if log: print("Stopped.")
                break
        return self._transcribe_frames(frames)

    def record_with_vad(self, log: bool = False) -> str:
        frames, ring = [], collections.deque(maxlen=self.preroll_chunks)
        triggered, silence, recorded = False, 0, 0
        if log: print("Listening...")
        while True:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            is_speech = self.vad.is_speech(data, self.rate)

            if not triggered:
                ring.append(data)
                if is_speech:
                    triggered = True
                    frames.extend(ring)
                    ring.clear()
                    if log: print("Speech started.")
            else:
                frames.append(data)
                recorded += 1
                if is_speech:
                    silence = 0
                else:
                    silence += 1
                    if recorded >= self.min_record_chunks and silence > self.tail_silence_chunks:
                        if log: print("Speech ended.")
                        break
        return self._transcribe_frames(frames)
    
    
    def record_with_callback_or_bool(self, callback_or_bool: Callable[..., bool] | bool, log: bool = False) -> str:
        if isinstance(callback_or_bool, bool):
            def go():
                if log:
                    print(f"Waiting for callback ...")
                
                wait_until(callback_or_bool)

                if log:
                    print("Recording... Press key again to stop.")
                frames, recorded = [], 0
                while True:
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    frames.append(data)
                    recorded += 1
                    if recorded < self.min_record_chunks:
                        continue
                    if callback_or_bool:
                        if log:
                            print("Stopped.")
                        break
                return self._transcribe_frames(frames)
            
        else:
            def go():
                if log:
                    print(f"Waiting for callback ...")

                wait_until(callback_or_bool)

                if log:
                    print("Recording... Press key again to stop.")
                frames, recorded = [], 0
                while True:
                    data = self.stream.read(
                        self.chunk, exception_on_overflow=False)
                    frames.append(data)
                    recorded += 1
                    if recorded < self.min_record_chunks:
                        continue
                    if callback_or_bool:
                        if log:
                            print("Stopped.")
                        break
                return self._transcribe_frames(frames)

        try:
            bob = ""
            def caller(obj: Future):
                global bob
                bob = obj.result()
                
            thr = ThreadPoolExecutor(max_workers=1).submit(go)
            thr.add_done_callback(caller)
        
        except Exception as e:
            print(e)
            
        finally:
            return bob
        
                

    def transcribe_file(self, audio_file: str) -> str:
        if self.backend == "faster":
            return self._fw_transcribe_file(audio_file)
        else:
            return self._whisper_transcribe_file(audio_file)


if __name__ == "__main__":
    stt = STT(model="base")  # use "tiny.en" if your CPU is weak
    try:
        print("Fixed seconds text:", stt.record_for_seconds(4))
        # print("Keyboard text:", stt.record_with_keyboard("space"))
        # print("VAD text:", stt.record_with_vad())
    finally:
        stt.close()
