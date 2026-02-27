# utils_attacks.py — common attacks to test watermark robustness
from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt

from utils_audio import read_wav_mono, write_wav_pcm16


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    from scipy.signal import resample_poly
    g = int(np.gcd(sr_in, sr_out))
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32, copy=False)


def attack_resample(input_path: Path, output_path: Path, target_sr: int) -> None:
    """Resample to target_sr then back to original SR (simulates format/phone resampling)."""
    x, sr = read_wav_mono(input_path)
    x_lo = _resample(x, sr, target_sr)
    x_back = _resample(x_lo, target_sr, sr)
    if len(x_back) != len(x):
        x_back = np.resize(x_back, len(x))
    write_wav_pcm16(output_path, x_back[: len(x)], sr)


def attack_mp3_roundtrip(input_path: Path, output_path: Path, bitrate: str = "128k") -> None:
    """WAV -> MP3 -> WAV to test codec robustness. Requires pydub and ffmpeg."""
    from pydub import AudioSegment
    seg = AudioSegment.from_wav(str(input_path))
    mp3_path = output_path.with_suffix(".mp3")
    seg.export(str(mp3_path), format="mp3", bitrate=bitrate)
    seg2 = AudioSegment.from_mp3(str(mp3_path))
    seg2.export(str(output_path), format="wav", parameters=["-acodec", "pcm_s16le"])
    try:
        mp3_path.unlink(missing_ok=True)
    except Exception:
        pass
    # Normalize to float32 mono for consistency
    y, sr = read_wav_mono(output_path)
    write_wav_pcm16(output_path, y, sr)


def attack_add_noise(input_path: Path, output_path: Path, snr_db: float = 20.0) -> None:
    """Add white Gaussian noise at given SNR (dB)."""
    x, sr = read_wav_mono(input_path)
    sig_power = np.mean(x ** 2) + 1e-12
    noise_power = sig_power / (10 ** (snr_db / 10.0))
    noise = np.random.default_rng(42).standard_normal(len(x), dtype=np.float32) * np.sqrt(noise_power)
    y = (x + noise).astype(np.float32)
    write_wav_pcm16(output_path, y, sr)


def attack_lowpass(input_path: Path, output_path: Path, cutoff_hz: float = 4000.0, order: int = 6) -> None:
    """Lowpass filter (e.g. telephone band)."""
    x, sr = read_wav_mono(input_path)
    nyq = sr / 2.0
    low = min(cutoff_hz / nyq, 0.99)
    sos = butter(order, low, btype="low", output="sos")
    y = sosfiltfilt(sos, x).astype(np.float32)
    write_wav_pcm16(output_path, y, sr)


def attack_volume(input_path: Path, output_path: Path, scale: float = 0.8) -> None:
    """Scale volume (common in re-uploads)."""
    x, sr = read_wav_mono(input_path)
    y = (x * scale).astype(np.float32)
    write_wav_pcm16(output_path, y, sr)


def run_all_attacks(marked_wav_path: str | Path, output_dir: str | Path) -> List[Tuple[str, Path]]:
    """
    Apply a standard set of attacks and save to output_dir.
    Returns list of (attack_name, path_to_attacked_wav).
    """
    marked = Path(marked_wav_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results: List[Tuple[str, Path]] = []

    # 1) Identity (copy) — baseline
    p = out_dir / "attack_identity.wav"
    shutil.copy2(marked, p)
    results.append(("identity", p))

    # 2) Resample 22k (e.g. some streaming)
    p = out_dir / "attack_resample_22k.wav"
    attack_resample(marked, p, 22050)
    results.append(("resample_22k", p))

    # 3) Resample 8k (telephony)
    p = out_dir / "attack_resample_8k.wav"
    attack_resample(marked, p, 8000)
    results.append(("resample_8k", p))

    # 4) MP3 128k (format conversion)
    p = out_dir / "attack_mp3_128k.wav"
    try:
        attack_mp3_roundtrip(marked, p, "128k")
        results.append(("mp3_128k", p))
    except Exception as e:
        print(f"[attacks] Skip mp3_128k: {e}")

    # 5) MP3 64k (heavy compression)
    p = out_dir / "attack_mp3_64k.wav"
    try:
        attack_mp3_roundtrip(marked, p, "64k")
        results.append(("mp3_64k", p))
    except Exception as e:
        print(f"[attacks] Skip mp3_64k: {e}")

    # 6) Add noise 20 dB
    p = out_dir / "attack_noise_20db.wav"
    attack_add_noise(marked, p, 20.0)
    results.append(("noise_20db", p))

    # 7) Add noise 10 dB
    p = out_dir / "attack_noise_10db.wav"
    attack_add_noise(marked, p, 10.0)
    results.append(("noise_10db", p))

    # 8) Lowpass 4 kHz
    p = out_dir / "attack_lowpass_4k.wav"
    attack_lowpass(marked, p, 4000.0)
    results.append(("lowpass_4k", p))

    # 9) Volume 0.8
    p = out_dir / "attack_volume_08.wav"
    attack_volume(marked, p, 0.8)
    results.append(("volume_08", p))

    return results
