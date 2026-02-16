from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf


def find_wavs(folder: str | Path) -> List[Path]:
    folder = Path(folder)
    out: List[Path] = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(".wav"):
                out.append(Path(root) / f)
    return sorted(out)


def to_mono_float32(x: np.ndarray) -> np.ndarray:
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False)


def read_wav_mono(path: str | Path) -> Tuple[np.ndarray, int]:
    y, sr = sf.read(str(path), always_2d=False)
    y = to_mono_float32(y)
    return y, int(sr)


def write_wav_pcm16(path: str | Path, audio: np.ndarray, sr: int) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    sf.write(str(path), audio, int(sr), subtype="PCM_16")


def safe_normalize_peak(x: np.ndarray, peak: float = 0.98) -> np.ndarray:
    m = float(np.max(np.abs(x)) + 1e-12)
    if m > peak:
        x = x * (peak / m)
    return x.astype(np.float32, copy=False)
