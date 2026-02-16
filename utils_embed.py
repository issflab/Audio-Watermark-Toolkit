from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import butter, sosfiltfilt
import torch

from utils_audio import find_wavs, read_wav_mono, write_wav_pcm16, safe_normalize_peak

if platform.system() == "Windows":
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    torch.compile = lambda m, *a, **k: m

from audioseal import AudioSeal


def butter_bandpass_sos(low_hz: float, high_hz: float, sr: int, order: int = 6):
    nyq = sr / 2.0
    low = max(low_hz / nyq, 1e-6)
    high = min(high_hz / nyq, 0.999999)
    if low >= high:
        raise ValueError(f"Bad band: {low_hz}-{high_hz} Hz for sr={sr} (nyquist={nyq})")
    return butter(order, [low, high], btype="bandpass", output="sos")


def band_split(x: np.ndarray, sr: int, bands):
    outs = []
    for lo, hi in bands:
        sos = butter_bandpass_sos(lo, hi, sr, order=6)
        y = sosfiltfilt(sos, x).astype(np.float32, copy=False)
        outs.append(y)
    return outs


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def parse_bands(bands_str: str):
    bands = []
    for part in bands_str.split(","):
        part = part.strip()
        if not part:
            continue
        lo_s, hi_s = part.split("-")
        bands.append((float(lo_s), float(hi_s)))
    if not bands:
        raise ValueError("No bands parsed. Example: 20-300,300-3000,3000-7600")
    return bands


def bits_to_tensor(message_bits: str, device: torch.device) -> torch.Tensor:
    bits = (message_bits or "").strip()
    if any(c not in "01" for c in bits):
        raise ValueError("Message must be a bitstring like 010101... containing only 0/1")
    arr = [1 if c == "1" else 0 for c in bits]
    return torch.tensor(arr, device=device, dtype=torch.long).unsqueeze(0)


def frame_rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def build_fade_envelope(n: int, sr: int, fade_ms: float = 50.0) -> np.ndarray:
    env = np.ones(n, dtype=np.float32)
    fade = int(sr * (fade_ms / 1000.0))
    if fade > 1 and (fade * 2) < n:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        env[:fade] *= ramp
        env[-fade:] *= ramp[::-1]
    return env


def choose_band_for_window(band_sigs, start, end, top_k=2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    energies = np.array([frame_rms(b[start:end]) for b in band_sigs], dtype=np.float32)
    order = np.argsort(energies)[::-1]
    pick_from = order[:max(1, min(int(top_k), len(order)))]
    return int(rng.choice(pick_from))


def resolve_device(device_str: str) -> torch.device:
    ds = (device_str or "cpu").lower()
    if ds.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device_str)
    if ds.startswith("cuda") and not torch.cuda.is_available():
        print("[warn] cuda requested but not available; using CPU.")
    return torch.device("cpu")


@torch.no_grad()
def audioseal_watermark_band(generator, device: torch.device, band_sig: np.ndarray, sr: int,
                            alpha: float, msg_tensor=None) -> np.ndarray:
    x = torch.from_numpy(band_sig).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    y = generator(x, sample_rate=sr, message=msg_tensor, alpha=float(alpha))
    return y.squeeze().detach().cpu().numpy().astype(np.float32, copy=False)


def watermark_folder_selective(
    input_dir: str,
    output_dir: str,
    *,
    suffix: str,
    alpha: float = 0.8,
    device: str = "cpu",
    message_bits: Optional[str] = None,
    bands_str: str = "20-300,300-3000,3000-7600",
    preserve_energy: bool = True,
    win_sec: float = 1.5,
    fade_ms: float = 50.0,
    top_k: int = 2,
    silence_rel: float = 0.15,
    max_mark_fraction: float = 0.60,
    seed: int = 123,
) -> None:
    bands = parse_bands(bands_str)
    dev = resolve_device(device)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[embed] device={dev} alpha={alpha} bands={bands}")
    gen = AudioSeal.load_generator("audioseal_wm_16bits").to(dev)
    gen.eval()

    msg_tensor = bits_to_tensor(message_bits, device=dev) if message_bits else None

    wavs = find_wavs(input_dir)
    print(f"[embed] found {len(wavs)} wav files in {input_dir}")

    rng = np.random.default_rng(None if seed == -1 else seed)

    for wav_path in wavs:
        x, sr = read_wav_mono(wav_path)
        n = len(x)
        if n < int(sr * 0.5):
            continue

        band_sigs = band_split(x, sr, bands)

        global_r = frame_rms(x)
        win = max(1, int(win_sec * sr))
        n_windows = int(np.ceil(n / win))

        non_silent = []
        for wi in range(n_windows):
            start = wi * win
            end = min(n, start + win)
            if frame_rms(x[start:end]) >= (silence_rel * global_r):
                non_silent.append((start, end))

        base = Path(wav_path).stem
        out_path = Path(output_dir) / f"{base}{suffix}.wav"

        if not non_silent:
            y = safe_normalize_peak(x, 0.98)
            write_wav_pcm16(out_path, y, sr)
            continue

        max_mark = int(np.floor(max_mark_fraction * len(non_silent)))
        max_mark = max(1, max_mark)
        chosen = set(rng.choice(len(non_silent), size=max_mark, replace=False)) if max_mark < len(non_silent) else set(range(len(non_silent)))

        envs = [np.zeros(n, dtype=np.float32) for _ in band_sigs]
        for j, (start, end) in enumerate(non_silent):
            if j not in chosen:
                continue
            k = choose_band_for_window(band_sigs, start, end, top_k=top_k, rng=rng)
            env = build_fade_envelope(end - start, sr, fade_ms=fade_ms)
            envs[k][start:end] = np.maximum(envs[k][start:end], env)

        wm_bands = []
        for b, env in zip(band_sigs, envs):
            before_r = rms(b)
            wm_full = audioseal_watermark_band(gen, dev, b, sr, alpha=alpha, msg_tensor=msg_tensor)
            if preserve_energy:
                after_r = rms(wm_full)
                if after_r > 1e-9:
                    wm_full = (wm_full * (before_r / after_r)).astype(np.float32, copy=False)
            delta = (wm_full - b).astype(np.float32, copy=False)
            out_band = (b + env * delta).astype(np.float32, copy=False)
            wm_bands.append(out_band)

        y = np.sum(np.stack(wm_bands, axis=0), axis=0)
        y = safe_normalize_peak(y, 0.98)
        write_wav_pcm16(out_path, y, sr)
