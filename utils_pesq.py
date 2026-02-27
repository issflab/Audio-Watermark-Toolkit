from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

try:
    from pesq import pesq as _pesq
    _PESQ_AVAILABLE = True
except Exception:
    _pesq = None
    _PESQ_AVAILABLE = False


def to_mono(x: np.ndarray) -> np.ndarray:
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x.astype(np.float32, copy=False)


def resample_to(x: np.ndarray, sr_in: int, sr_out: int = 16000) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    g = np.gcd(sr_in, sr_out)
    return resample_poly(x, sr_out // g, sr_in // g).astype(np.float32, copy=False)


def align(a: np.ndarray, b: np.ndarray):
    n = min(len(a), len(b))
    return a[:n], b[:n]


def pesq_wb(ref: np.ndarray, test: np.ndarray, sr: int = 16000) -> float:
    if _pesq is None:
        raise RuntimeError(
            "PESQ not available. Install with: pip install pesq (on Windows you may need Microsoft C++ Build Tools)."
        )
    # PESQ expects 16 kHz; minimum length ~0.5s for stable score
    min_len = 8000 if sr == 16000 else int(0.5 * sr)
    if len(ref) < min_len or len(test) < min_len:
        raise ValueError(f"PESQ needs at least {min_len} samples at {sr} Hz (got {len(ref)}, {len(test)})")
    out = _pesq(sr, ref.astype(np.float64), test.astype(np.float64), "wb")
    score = float(out)
    if not np.isfinite(score):
        raise ValueError("PESQ returned non-finite value (e.g. non-speech or invalid signal)")
    return score


def pesq_available() -> bool:
    """Return True if PESQ can be used (for reporting in demos)."""
    return _PESQ_AVAILABLE


def pesq_pair_files(ref_path: str | Path, test_path: str | Path) -> Optional[float]:
    """Compute wideband PESQ (ref vs test). Returns None if PESQ unavailable or signal invalid."""
    if not _PESQ_AVAILABLE:
        return None
    try:
        ref, sr1 = sf.read(str(ref_path), always_2d=False)
        test, sr2 = sf.read(str(test_path), always_2d=False)
        ref = to_mono(ref)
        test = to_mono(test)
        ref = resample_to(ref, sr1, 16000)
        test = resample_to(test, sr2, 16000)
        ref, test = align(ref, test)
        return pesq_wb(ref, test, 16000)
    except Exception:
        return None


def pesq_table_clean_vs_marked(ref_dir: str | Path, marked_dir: str | Path, suffix: str, out_csv: str | Path) -> None:
    ref_dir = Path(ref_dir)
    marked_dir = Path(marked_dir)

    rows = []
    for ref_path in sorted(ref_dir.glob("*.wav")):
        base = ref_path.stem
        test_path = marked_dir / f"{base}{suffix}.wav"
        if not test_path.exists():
            rows.append({"file": base, "status": "missing_test", "pesq_wb": None})
            continue
        score = pesq_pair_files(ref_path, test_path)
        rows.append({"file": base, "status": "ok" if score is not None else "error", "pesq_wb": score})

    df = pd.DataFrame(rows)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print("[pesq] Saved:", out_csv)
