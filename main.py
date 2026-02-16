# main.py
from __future__ import annotations

import argparse
from pathlib import Path
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str | Path):
    Path(p).mkdir(parents=True, exist_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--step", default="all", choices=["embed", "eval", "all"])
    args = ap.parse_args()

    cfg = load_config(args.config)

    paths = cfg["paths"]
    clean_dir = paths["clean_dir"]
    marked_dir = paths["marked_dir"]

    # ---- EMBED ---- (imports torch/audioseal only if needed)
    if args.step in ("embed", "all") and cfg.get("embed", {}).get("enabled", True):
        from utils_embed import watermark_folder_selective

        e = cfg["embed"]
        ensure_dir(marked_dir)
        watermark_folder_selective(
            input_dir=clean_dir,
            output_dir=marked_dir,
            suffix=str(e.get("suffix", "_selwm")),
            alpha=float(e.get("alpha", 0.8)),
            device=str(e.get("device", "cpu")),
            message_bits=e.get("message_bits", None),
            bands_str=str(e.get("bands", "20-300,300-3000,3000-7600")),
            win_sec=float(e.get("win_sec", 1.5)),
            fade_ms=float(e.get("fade_ms", 50.0)),
            top_k=int(e.get("top_k", 2)),
            silence_rel=float(e.get("silence_rel", 0.15)),
            max_mark_fraction=float(e.get("max_mark_fraction", 0.60)),
            seed=int(e.get("seed", 123)),
        )

    # ---- EVAL ---- (imports torch/pesq only if needed)
    if args.step in ("eval", "all") and cfg.get("eval", {}).get("enabled", True):
        from utils_detect import detect_clean_vs_marked
        from utils_pesq import pesq_table_clean_vs_marked

        ev = cfg["eval"]
        out_dir = ev.get("out_dir", "./outputs")
        ensure_dir(out_dir)

        suffix = cfg.get("embed", {}).get("suffix", "_selwm")

        if ev.get("compute_marked_pesq", True):
            pesq_table_clean_vs_marked(clean_dir, marked_dir, suffix, Path(out_dir) / "pesq_marked.csv")

        if ev.get("compute_marked_detection", True):
            detect_clean_vs_marked(clean_dir, marked_dir, suffix, Path(out_dir) / "detect_marked.csv")

        print("[done] outputs in:", out_dir)


if __name__ == "__main__":
    main()
