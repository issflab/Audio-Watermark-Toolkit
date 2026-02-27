# Watermark toolkit (Embed • Detect • PESQ • Attacks)

AudioSeal watermarking with user attribution, detection, PESQ scoring, and robustness checks after common attacks (resample, MP3 conversion, noise, lowpass, volume).

## How the watermark is added (based on what)

The watermark is embedded in **three layers**: what we use as **input**, how we **derive the message**, and **where/how** we inject it.

### 1. Input: audio and bands

- **Input**: Mono WAV files from `clean_dir` (or, in the demo, WAV converted from MP3). Sample rate is preserved (16 kHz recommended for best detection).
- **Band split**: The signal is split into **three frequency bands** (config: `embed.bands`), default `20-300, 300-3000, 3000-7600` Hz. Each band is filtered (bandpass, order 6), so the watermark can be placed in different spectral regions.
- **Per-band processing**: Each band is processed separately by the AudioSeal generator; then bands are recombined so the final watermark is spread across low, mid, and high frequencies.

### 2. Message bits (what is embedded)

- **Raw message**: A **0/1 bitstring** of length **16** (AudioSeal’s limit; config: `embed.message_bit_len`). This is passed to the generator as the “message” to encode.
- **When attribution is on**: Instead of a fixed bitstring from config, the bits are **derived deterministically** from:
  - **Secret key**: `WATERMARK_SECRET` (env var only, server-side).
  - **User/request data**: `api_key`, `user_id`, `request_time`, and optional `nonce`.
  - **Formula**: `HMAC-SHA256(secret_key, "api_key=...|user_id=...|request_time=...|nonce=...")` → 256 bits → first **16 bits** used. Same inputs ⇒ same bits; different user or time ⇒ different bits, so each embed is tied to that user/request.

So the watermark is added **based on**:
- The **audio** (bands + non-silent regions),
- The **message bits** (either from config or from the attribution formula above).

### 3. Where and how it is injected

- **Non-silent windows**: The file is split into fixed-length **windows** (default 1.5 s). Only windows whose RMS is above `silence_rel * global_rms` are considered; silence is skipped.
- **Selective marking**: At most `max_mark_fraction` (default 60%) of those non-silent windows are chosen at random (seed from config). So we do **not** mark the whole file—only a subset of loud-enough regions.
- **Band choice per window**: For each chosen window, we pick one of the three bands (among the **top_k** most energetic, default 2) at random, so the watermark is concentrated in the most active band for that segment.
- **Fade envelope**: Within each chosen window, a short fade-in/out (default 50 ms) is applied so the watermark doesn’t click at edges.
- **AudioSeal generator**: The model `audioseal_wm_16bits` takes the band signal, **sample rate**, **message bits** (16-bit tensor), and **alpha** (default 0.8, strength). It outputs a watermarked version of that band.
- **Blending**: For each band, we compute `delta = watermarked_band - original_band` and form `output_band = original_band + envelope * delta`. So only in chosen windows and in the chosen band is the watermark added; elsewhere the signal is unchanged. If `preserve_energy` is on, band energy is scaled so RMS is preserved.
- **Sum and normalize**: All bands are summed back to mono and peak-normalized (0.98); result is written as PCM 16-bit WAV to `marked_dir` with the configured suffix (e.g. `_selwm`).

**Summary**: The watermark is added **based on** (1) the **frequency bands** and **non-silent windows** of the audio, (2) the **message bits** (from config or from **secret + user_id + api_key + request_time + nonce**), and (3) **AudioSeal’s generator** with the given **alpha** and **selective blending** rules above.

For **how to run the app**, **how to embed**, and **technical details** (bands, message bits, detection, PESQ, attacks), see **[usage.md](usage.md)**.

## Folder structure

```
  main.py              # Batch embed + eval
  run_demo.py          # Single-file demo with user "kamala" and attacks
  config.yaml
  demo_config.yaml
  utils_audio.py      # WAV/MP3 read, convert
  utils_embed.py
  utils_detect.py
  utils_pesq.py
  utils_attribution.py # User attribution message_bits
  utils_attacks.py    # Resample, MP3, noise, lowpass, volume
  demo/
    clean/             # demo_audio.wav (from demo_audio.mp3 or generated)
    marked/            # watermarked output
    attacked/          # attacked versions for robustness
    outputs/           # demo_results.csv
```

## Install

```bash
pip install -r requirements.txt
```

- **MP3**: Install **ffmpeg** and add it to your PATH (for demo input and format-conversion attacks).
- **PESQ** (quality scores): The `pesq` package may fail to build on Windows without **Microsoft C++ Build Tools**. If it fails, the rest of the pipeline still runs; PESQ columns in CSVs and demo output will be empty/N/A. On Linux, `pip install pesq` usually works.

## Configure

Edit config.yaml:
- paths.clean_dir / marked_dir: input and output folders
- embed.message_bit_len: **16** (AudioSeal supports 16-bit messages)
- eval.out_dir: output CSV folder

User attribution (optional): set env `WATERMARK_SECRET` and pass `--api_key`, `--user_id`, `--request_time` to main.py.

## Run batch pipeline

```bash
python main.py
```

With user attribution (e.g. user "kamala"):

```bash
set WATERMARK_SECRET=your-secret
python main.py --step embed --user_id kamala --api_key demo --request_time 2025-02-27T12:00:00Z
python main.py --step eval
```

## Demo: one file, user "kamala", PESQ + detection + attacks

Uses **demo_audio.mp3** (or **demo_audio.wav**) if present; otherwise generates a short tone. Embeds with user **kamala**, then runs PESQ (before/after watermark) and detection, then applies all attacks and re-checks detection and PESQ.

```bash
python run_demo.py --user kamala
```

With your own file:

```bash
python run_demo.py --demo_audio path/to/demo_audio.mp3 --user kamala
```

Outputs:
- **demo/outputs/demo_results.csv**: per-stage `detection_prob` and `pesq_vs_clean` (clean, marked, and each attack).
- **PESQ**: “before” = N/A (reference); “after” = PESQ(clean, marked). After attacks, PESQ(clean, attacked) shows quality degradation.
- **Watermark survival**: detection probability stays high (e.g. > 0.5) after an attack if the watermark is still present.

Attacks applied:
- identity (copy), resample 22k/8k, **MP3 128k/64k** (format conversion; needs ffmpeg), noise 20/10 dB, lowpass 4 kHz, volume 0.8.

## Real applications

This level is suitable for real use provided you:

| Requirement | What to do |
|-------------|------------|
| **Content** | Use **speech** (or speech-like) at **16 kHz**. AudioSeal is trained on speech; detection on music or tones can be near zero. |
| **Secret** | Keep `WATERMARK_SECRET` only on the server (env var), never in config or client. |
| **Attribution** | Pass `user_id`, `api_key`, `request_time` (and optional `nonce`) when embedding; store them in your DB to re-derive bits for verification. |
| **Detection** | Detection probability &gt; 0.5 (e.g. 0.7+) indicates watermark present. Compare with clean (unmarked) baseline. |
| **PESQ** | Install `pesq` where possible for quality metrics; pipeline runs without it. PESQ is most meaningful on speech. |
| **Attacks** | Resample, MP3, noise, lowpass, and volume are supported; format conversion (MP3) needs ffmpeg. |

Embedding and detection logic are production-ready; configure paths, secrets, and storage for your environment.

## Notes

- If CUDA is unavailable, the script falls back to CPU.
- PESQ uses 16 kHz (handled internally). For best detection and PESQ, use **speech**; simple tones may give low detection and unstable PESQ.
- AudioSeal message is **16 bits**; attribution uses a deterministic bitstring derived from secret, user_id, api_key, request_time (see utils_attribution.py).
