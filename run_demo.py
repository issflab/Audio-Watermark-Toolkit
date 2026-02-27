# run_demo.py — Full demo: embed with user "kamala", PESQ, detection, then attacks + re-check
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

from utils_attribution import derive_message_bits
from utils_attacks import run_all_attacks
from utils_audio import convert_mp3_to_wav, read_wav_mono, write_wav_pcm16


def _make_fallback_tone(out_wav: Path, sr: int = 16000, dur_sec: float = 3.0) -> None:
    clean_dir = out_wav.parent
    clean_dir.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, dur_sec, int(dur_sec * sr), dtype=np.float32)
    x = 0.3 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    write_wav_pcm16(out_wav, x, sr)


def ensure_demo_clean_wav(demo_audio_path: Path, clean_dir: Path, stem: str = "demo_audio") -> Path:
    """Put demo_audio (mp3 or wav) as demo/clean/<stem>.wav. If missing or MP3 conversion fails, create short tone."""
    clean_dir.mkdir(parents=True, exist_ok=True)
    out_wav = clean_dir / f"{stem}.wav"
    if demo_audio_path.exists():
        suf = demo_audio_path.suffix.lower()
        if suf == ".mp3":
            try:
                convert_mp3_to_wav(demo_audio_path, out_wav, sr=16000)
                print(f"[demo] Using source: {demo_audio_path} -> {out_wav}")
            except FileNotFoundError:
                print("[demo] ffmpeg not found; cannot decode MP3. Install ffmpeg or use a WAV file. Using 3s tone.")
                _make_fallback_tone(out_wav)
            except Exception as e:
                print(f"[demo] MP3 conversion failed: {e}. Using 3s tone.")
                _make_fallback_tone(out_wav)
        else:
            x, sr = read_wav_mono(demo_audio_path)
            if sr != 16000:
                g = int(np.gcd(sr, 16000))
                x = resample_poly(x, 16000 // g, sr // g).astype("float32")
                sr = 16000
            write_wav_pcm16(out_wav, x, sr)
            print(f"[demo] Using source: {demo_audio_path} -> {out_wav}")
    else:
        _make_fallback_tone(out_wav)
        print(f"[demo] {demo_audio_path} not found; wrote 3s tone to {out_wav}")
    return out_wav


def main():
    ap = argparse.ArgumentParser(description="Demo: embed with user 'kamala', run PESQ + detection, then attacks.")
    ap.add_argument("--demo_audio", default="demo_audio.mp3", help="Input MP3 or WAV (default: demo_audio.mp3)")
    ap.add_argument("--config", default="demo_config.yaml")
    ap.add_argument("--user", default="kamala", help="User id for attribution watermark")
    args = ap.parse_args()

    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    paths = cfg["paths"]
    clean_dir = Path(paths["clean_dir"])
    marked_dir = Path(paths["marked_dir"])
    out_dir = Path(cfg["eval"]["out_dir"])
    attacked_dir = Path("demo/attacked")
    out_dir.mkdir(parents=True, exist_ok=True)
    attacked_dir.mkdir(parents=True, exist_ok=True)

    suffix = cfg["embed"]["suffix"]
    stem = "demo_audio"

    # 1) Prepare clean WAV
    demo_audio_path = Path(args.demo_audio)
    ensure_demo_clean_wav(demo_audio_path, clean_dir, stem=stem)
    clean_wav = clean_dir / f"{stem}.wav"

    # 2) Attribution bits for user "kamala"
    secret = os.environ.get("WATERMARK_SECRET", "demo_secret_change_in_production")
    request_time = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    message_bits = derive_message_bits(
        secret_key=secret,
        api_key="demo",
        user_id=args.user,
        request_time=request_time,
        nonce="",
        bit_len=int(cfg["embed"].get("message_bit_len", 16)),
    )
    cfg["embed"]["message_bits"] = message_bits

    # 3) Embed
    from utils_embed import watermark_folder_selective
    e = cfg["embed"]
    marked_dir.mkdir(parents=True, exist_ok=True)
    watermark_folder_selective(
        input_dir=str(clean_dir),
        output_dir=str(marked_dir),
        suffix=e["suffix"],
        alpha=float(e["alpha"]),
        device=str(e.get("device", "cpu")),
        message_bits=e["message_bits"],
        bands_str=str(e.get("bands", "20-300,300-3000,3000-7600")),
        win_sec=float(e.get("win_sec", 1.5)),
        fade_ms=float(e.get("fade_ms", 50.0)),
        top_k=int(e.get("top_k", 2)),
        silence_rel=float(e.get("silence_rel", 0.15)),
        max_mark_fraction=float(e.get("max_mark_fraction", 0.60)),
        seed=int(e.get("seed", 123)),
    )
    marked_wav = marked_dir / f"{stem}{suffix}.wav"
    if not marked_wav.exists():
        print("[demo] ERROR: marked file not produced")
        return

    # 4) PESQ and detection (before attacks)
    from utils_detect import load_detector, detect_prob
    from utils_pesq import pesq_pair_files, pesq_available
    import torch
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    detector = load_detector(dev)

    pesq_clean_marked = pesq_pair_files(clean_wav, marked_wav)
    prob_clean = detect_prob(detector, clean_wav, dev)
    prob_marked = detect_prob(detector, marked_wav, dev)

    rows = [
        {"stage": "clean", "detection_prob": prob_clean, "pesq_vs_clean": None},
        {"stage": "marked", "detection_prob": prob_marked, "pesq_vs_clean": pesq_clean_marked},
    ]

    # 5) Attacks
    print("[demo] Applying attacks...")
    attack_list = run_all_attacks(marked_wav, attacked_dir)
    for name, attacked_path in attack_list:
        prob = detect_prob(detector, attacked_path, dev)
        pesq = pesq_pair_files(clean_wav, attacked_path)
        rows.append({"stage": f"attack_{name}", "detection_prob": prob, "pesq_vs_clean": pesq})

    # 6) Report
    df = pd.DataFrame(rows)
    csv_path = out_dir / "demo_results.csv"
    df.to_csv(csv_path, index=False)
    print("\n[demo] Results (user attribution: %s)" % args.user)
    print("-" * 60)
    print(df.to_string())
    print("-" * 60)
    if not pesq_available():
        print("PESQ: not available (install 'pesq'; on Windows may need Microsoft C++ Build Tools).")
    else:
        print("PESQ (before watermark): N/A (reference is clean)")
        val = pesq_clean_marked
        print("PESQ (after watermark, vs clean): %s" % (f"{val:.4f}" if val is not None and val == val else "N/A"))
    print("Detection prob (clean): %.4f" % (prob_clean if prob_clean is not None else float("nan")))
    print("Detection prob (marked): %.4f" % (prob_marked if prob_marked is not None else float("nan")))
    print("\nWatermark survives attacks where detection_prob stays high (e.g. > 0.5).")
    print("Saved:", csv_path)


if __name__ == "__main__":
    main()
