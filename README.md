# Watermark toolkit (Embed • Detect • PESQ)

Minimal, attack-free toolkit for AudioSeal watermarking, detection, and PESQ scoring.

## Folder structure

```
watermark_only/
  main.py
  config.yaml
  requirements.txt
  utils_audio.py
  utils_embed.py
  utils_detect.py
  utils_pesq.py
  outputs/
```

## Install (in the SAME environment you run with)

```bash
pip install -r requirements.txt
```

## Configure

Edit config.yaml:
- paths.clean_dir: folder with clean .wav files
- paths.marked_dir: output folder for watermarked .wav files
- eval.out_dir: output CSV folder

## Run

```bash
python main.py
```

Optional steps:
```bash
python main.py --step embed
python main.py --step eval
```

Notes:
- If CUDA is unavailable, it will fall back to CPU.
- PESQ requires a 16 kHz resampling step (handled internally).
