from typing import Any, Callable, Literal
import os
import time
import wave
import tempfile
import collections
from collections.abc import Iterable
import warnings
import logging
import pyaudio
import webrtcvad
import keyboard as kb
from syntaxmod.general import wait_until
from openai import OpenAI

# Quiet the spam
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
for name in ["whisper", "transformers", "numba"]:
    logging.getLogger(name).setLevel(logging.ERROR)

# Try faster-whisper first
_USE_FASTER = True
try:
    from faster_whisper import WhisperModel as FWModel  # type: ignore
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
            "tiny.en",
            "tiny",
            "base.en",
            "base",
            "small.en",
            "small",
            "medium.en",
            "medium",
            "large-v1",
            "large-v2",
            "large-v3",
            "large",
            "large-v3-turbo",
            "turbo",
            "whisper-1",
            "gpt-4o-transcribe",
            "gpt-4o-mini-transcribe",
        ] = "base",
        aggressive: int = 2,
        chunk_duration_ms: int = 30,
        preroll_ms: int = 300,
        tail_silence_ms: int = 600,
        min_record_ms: int = 400,
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
            frames_per_buffer=self.chunk,
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
                download_root=os.path.join(tempfile.gettempdir(), "whisper_models"),
            )
        else:
            device = "cuda" if (_HAS_CUDA) else "cpu"
            self.model = whisper.load_model(
                model,
                download_root=os.path.join(tempfile.gettempdir(), "whisper_models"),
                device=device,
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
                filename, beam_size=1, temperature=0.0, vad_filter=False, language="en"
            )
        return "".join(seg.text for seg in segments).strip()  # type: ignore

    def _whisper_transcribe_file(self, filename: str) -> str:
        # Only use options the real whisper accepts
        result = self.model.transcribe(
            filename,
            temperature=0.0,
            condition_on_previous_text=False,
            word_timestamps=False,
            language="en",
        )
        return result.get("text", "").strip()  # type: ignore

    def _transcribe_frames(self, frames) -> str:
        if not frames:
            return ""
        filename = self._save_wav_temp(frames)
        if self.backend == "faster":
            return self._fw_transcribe_file(filename)
        else:
            return self._whisper_transcribe_file(filename)

    def _collect_frames(
        self,
        *,
        min_appended_chunks: int | None = None,
        max_read_chunks: int | None = None,
        stop_condition: Callable[[dict[str, Any]], bool] | None = None,
        on_chunk: Callable[[bytes, list[bytes], dict[str, Any]], bytes | Iterable[bytes] | None]
        | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[bytes]:
        frames: list[bytes] = []
        ctx = dict(context or {})
        ctx.setdefault("read_chunks", 0)
        ctx.setdefault("appended_chunks", 0)

        min_required = (
            self.min_record_chunks if min_appended_chunks is None else min_appended_chunks
        )

        while True:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            ctx["read_chunks"] += 1

            result = on_chunk(data, frames, ctx) if on_chunk else data

            appended = 0
            if result is None:
                pass
            elif isinstance(result, (bytes, bytearray)):
                frames.append(bytes(result))
                appended = 1
            elif isinstance(result, Iterable):
                new_frames: list[bytes] = []
                for chunk in result:
                    if not isinstance(chunk, (bytes, bytearray)):
                        raise TypeError(
                            "on_chunk must return bytes or an iterable of bytes, got"
                            f" {type(chunk)!r}"
                        )
                    new_frames.append(bytes(chunk))
                frames.extend(new_frames)
                appended = len(new_frames)
            else:
                raise TypeError(
                    "on_chunk must return bytes, an iterable of bytes, or None, got"
                    f" {type(result)!r}"
                )

            ctx["appended_chunks"] += appended

            if max_read_chunks is not None and ctx["read_chunks"] >= max_read_chunks:
                break

            if ctx["appended_chunks"] < min_required:
                continue

            if stop_condition and stop_condition(ctx):
                break

        return frames

    # ---------- public APIs ----------

    def record_for_seconds(self, duration=5, log: bool = False) -> str:
        if log:
            print(f"Recording for {duration} seconds...")
        total_chunks = max(1, int((duration * 1000) / self.chunk_ms))
        frames = self._collect_frames(max_read_chunks=total_chunks, min_appended_chunks=0)
        print("Done.")
        return self._transcribe_frames(frames)

    def record_with_keyboard(self, key="space", log: bool = False) -> str:
        if log:
            print(f"Press {key} to start. Release to begin recording.")
        kb.wait(key)
        while kb.is_pressed(key):
            time.sleep(0.01)

        if log:
            print("Recording... Press key again to stop.")
        stopped_announced = False

        def stop_condition(_: dict[str, Any]) -> bool:
            nonlocal stopped_announced
            if kb.is_pressed(key):
                while kb.is_pressed(key):
                    time.sleep(0.01)
                if log and not stopped_announced:
                    print("Stopped.")
                    stopped_announced = True
                return True
            return False

        frames = self._collect_frames(stop_condition=stop_condition)
        return self._transcribe_frames(frames)

    def record_with_vad(self, log: bool = False) -> str:
        ring = collections.deque(maxlen=self.preroll_chunks)
        state = {"triggered": False, "silence": 0, "recorded": 0}
        if log:
            print("Listening...")
        started_announced = False
        ended_announced = False

        def on_chunk(data: bytes, _frames: list[bytes], _ctx: dict[str, Any]):
            nonlocal started_announced
            is_speech = self.vad.is_speech(data, self.rate)
            if not state["triggered"]:
                ring.append(data)
                if is_speech:
                    state["triggered"] = True
                    state["silence"] = 0
                    if log and not started_announced:
                        print("Speech started.")
                        started_announced = True
                    buffered = list(ring)
                    ring.clear()
                    return buffered
                return []

            state["recorded"] += 1
            if is_speech:
                state["silence"] = 0
            else:
                state["silence"] += 1
            return [data]

        def stop_condition(_: dict[str, Any]) -> bool:
            nonlocal ended_announced
            if not state["triggered"]:
                return False
            if (
                state["recorded"] >= self.min_record_chunks
                and state["silence"] > self.tail_silence_chunks
            ):
                if log and not ended_announced:
                    print("Speech ended.")
                    ended_announced = True
                return True
            return False

        frames = self._collect_frames(
            min_appended_chunks=0,
            on_chunk=on_chunk,
            stop_condition=stop_condition,
        )
        return self._transcribe_frames(frames)

    def record_with_callback_or_bool(
        self, callback_or_bool: Callable[..., Any] | bool, log: bool = False
    ) -> str:
        if log:
            print(f"Waiting for callback ...")

        wait_until(callback_or_bool)

        if log:
            print("Recording... Press key again to stop.")

        stopped_announced = False

        def stop_condition(_: dict[str, Any]) -> bool:
            nonlocal stopped_announced
            should_stop: bool
            if isinstance(callback_or_bool, bool):
                should_stop = callback_or_bool
            else:
                should_stop = bool(callback_or_bool())
            if should_stop:
                if log and not stopped_announced:
                    print("Stopped.")
                    stopped_announced = True
                return True
            return False

        frames = self._collect_frames(stop_condition=stop_condition)
        return self._transcribe_frames(frames)

    def transcribe_file(self, audio_file: str) -> str:
        if self.backend == "faster":
            return self._fw_transcribe_file(audio_file)
        else:
            return self._whisper_transcribe_file(audio_file)


class API_STT(STT):

    def __init__(
        self,
        api_key: str | None = None,
        model: Literal[
            "whisper-1", "gpt-4o-transcribe", "gpt-4o-mini-transcribe"
        ] = "whisper-1",
    ):
        key = api_key if isinstance(api_key, str) else os.getenv("OPENAI_API_KEY")
        if key is None:
            raise Exception("Missing OPENAI_API_KEY")
        self.client = OpenAI(api_key=key)
        self.model = model


if __name__ == "__main__":
    stt = STT(model="base")  # use "tiny.en" if your CPU is weak
    try:
        print("Fixed seconds text:", stt.record_for_seconds(4))
        # print("Keyboard text:", stt.record_with_keyboard("space"))
        # print("VAD text:", stt.record_with_vad())
    finally:
        stt.close()
