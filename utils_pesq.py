from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly

try:
    from pesq import pesq as _pesq
except Exception:
    _pesq = None


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
        raise RuntimeError("pesq not installed/imported. Try `pip install pesq` or `pip install pypesq`.")
    return float(_pesq(sr, ref, test, "wb"))


def pesq_pair_files(ref_path: str | Path, test_path: str | Path) -> Optional[float]:
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
