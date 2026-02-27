from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf


def find_audio(folder: str | Path, exts: Tuple[str, ...] = (".wav", ".mp3")) -> List[Path]:
    folder = Path(folder)
    out: List[Path] = []
    for root, _, files in os.walk(folder):
        for f in files:
            if any(f.lower().endswith(e) for e in exts):
                out.append(Path(root) / f)
    return sorted(out)


def find_wavs(folder: str | Path) -> List[Path]:
    return find_audio(folder, exts=(".wav",))


def to_mono_float32(x: np.ndarray) -> np.ndarray:
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False)


def read_wav_mono(path: str | Path) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), always_2d=False)
    y = to_mono_float32(y)
    return y, int(sr)


def read_audio_mono(path: str | Path) -> Tuple[np.ndarray, int]:
    """Load WAV or MP3 as mono float32 and sample rate."""
    path = Path(path)
    suf = path.suffix.lower()
    if suf == ".mp3":
        try:
            from pydub import AudioSegment
            seg = AudioSegment.from_mp3(str(path))
            seg = seg.set_channels(1)
            sr = seg.frame_rate
            samples = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
            return samples, sr
        except Exception as e:
            raise RuntimeError(f"Failed to read MP3 (install pydub and ffmpeg): {e}") from e
    y, sr = sf.read(str(path), always_2d=False)
    y = to_mono_float32(y)
    return y, int(sr)


def convert_mp3_to_wav(mp3_path: str | Path, wav_path: str | Path, sr: int = 16000) -> None:
    """Convert MP3 to WAV (mono). Requires pydub and ffmpeg."""
    from pydub import AudioSegment
    seg = AudioSegment.from_mp3(str(mp3_path))
    seg = seg.set_channels(1)
    if seg.frame_rate != sr:
        seg = seg.set_frame_rate(sr)
    seg.export(str(wav_path), format="wav", parameters=["-acodec", "pcm_s16le"])
    y, _ = read_wav_mono(wav_path)
    write_wav_pcm16(wav_path, y, sr)


def write_wav_pcm16(path: str | Path, audio: np.ndarray, sr: int) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    sf.write(str(path), audio, int(sr), subtype="PCM_16")


def safe_normalize_peak(x: np.ndarray, peak: float = 0.98) -> np.ndarray:
    m = float(np.max(np.abs(x)) + 1e-12)
    if m > peak:
        x = x * (peak / m)
    return x.astype(np.float32, copy=False)
