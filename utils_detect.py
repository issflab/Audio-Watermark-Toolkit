from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Optional

import pandas as pd
import torch

from utils_audio import find_wavs, read_wav_mono

if platform.system() == "Windows":
    os.environ["TORCHDYNAMO_DISABLE"] = "1"
    torch.compile = lambda m, *a, **k: m

from audioseal import AudioSeal


def load_detector(device: torch.device):
    det = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
    det.eval()
    return det


@torch.no_grad()
def detect_prob(detector, wav_path: Path, device: torch.device) -> Optional[float]:
    try:
        wav, _sr = read_wav_mono(wav_path)
        x = torch.from_numpy(wav).unsqueeze(0).unsqueeze(0).to(device)
        prob, _ = detector.detect_watermark(x)
        return float(prob.item())
    except Exception:
        return None


def detect_clean_vs_marked(clean_dir: str | Path, marked_dir: str | Path, suffix: str, out_csv: str | Path, device: Optional[str] = None) -> None:
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = load_detector(dev)

    rows = []
    for clean_path in find_wavs(clean_dir):
        base = clean_path.stem
        marked_path = Path(marked_dir) / f"{base}{suffix}.wav"
        rows.append({
            "audio_name": base,
            "clean_detection_prob": detect_prob(detector, clean_path, dev),
            "marked_detection_prob": detect_prob(detector, marked_path, dev) if marked_path.exists() else None,
            "marked_exists": bool(marked_path.exists()),
        })

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("[detect] Saved:", out_csv)
