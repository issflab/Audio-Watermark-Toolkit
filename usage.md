# Usage guide

How to run the app, how to embed watermarks, and technical details.

---

## 1. How to run the app

### Prerequisites

- Python 3.8+
- Install dependencies:
  ```bash
  pip install -r requirements.txt
  ```
- Optional: **ffmpeg** on PATH (for MP3 input and MP3 round-trip attacks).
- Optional: **PESQ** — if `pesq` installs (on Windows you may need Microsoft C++ Build Tools), quality scores are computed; otherwise the pipeline runs and PESQ columns are N/A.

### Two entry points

| Entry point    | Use case |
|----------------|----------|
| **main.py**    | Batch: embed and/or evaluate all WAVs in a folder (config-driven paths). |
| **run_demo.py**| Single-file demo: one input (e.g. demo_audio.mp3), user attribution, then PESQ + detection + attacks. |

---

## 2. Running the batch pipeline (main.py)

### Config file

Edit **config.yaml** (or pass `--config other.yaml`):

- **paths.clean_dir**: folder containing input WAV files.
- **paths.marked_dir**: folder where watermarked WAVs are written.
- **embed.***: embedding options (see [Technical details](#4-technical-details) below).
- **eval.out_dir**: directory for output CSVs (PESQ, detection).

### Commands

**Embed and evaluate (default):**
```bash
python main.py
```

**Embed only:**
```bash
python main.py --step embed
```

**Evaluate only (after you have already run embed):**
```bash
python main.py --step eval
```

**With user attribution** (message bits derived from user/request; requires `WATERMARK_SECRET` and the three IDs):

```bash
set WATERMARK_SECRET=your-server-secret
python main.py --step embed --user_id kamala --api_key ak_xxx --request_time 2025-02-27T12:00:00Z
python main.py --step eval
```

Optional: `--nonce something` for extra uniqueness.

**Outputs:**

- Watermarked files: `marked_dir/<basename><suffix>.wav` (e.g. `file_selwm.wav`).
- **eval.out_dir/pesq_marked.csv**: PESQ (clean vs marked) per file.
- **eval.out_dir/detect_marked.csv**: detection probability for clean and marked per file.

---

## 3. How to embed

### Option A: Embed via main.py (batch)

1. Put clean WAV files in **paths.clean_dir** (in config).
2. Set **embed** options in config (bands, alpha, suffix, etc.).
3. **Without attribution**: leave `embed.message_bits` as `null` (or set a 16-bit string like `"0101010101010101"` if you want a fixed message).
4. **With attribution**: set env `WATERMARK_SECRET` and run with `--user_id`, `--api_key`, `--request_time` (and optionally `--nonce`). The 16-bit message is then derived automatically from those (see README).
5. Run:
   ```bash
   python main.py --step embed   # with or without attribution args
   ```
6. Watermarked files appear in **paths.marked_dir**.

### Option B: Embed via run_demo.py (single file)

1. Place your file as **demo_audio.mp3** (or **demo_audio.wav**) in the project root, or pass its path.
2. Run:
   ```bash
   python run_demo.py --user kamala
   ```
   Or with a custom file and user:
   ```bash
   python run_demo.py --demo_audio path/to/speech.wav --user kamala
   ```
3. The script:
   - Converts MP3 → WAV to **demo/clean/** (or uses WAV; needs ffmpeg for MP3).
   - Derives 16-bit message from user **kamala** (and demo api_key/request_time).
   - Embeds using **demo_config.yaml** (paths: demo/clean, demo/marked).
   - Runs PESQ and detection on clean vs marked.
   - Applies all attacks (resample, MP3, noise, lowpass, volume) and writes **demo/attacked/**.
   - Writes **demo/outputs/demo_results.csv** with detection and PESQ per stage.

So “how to embed” here = run one of the two entry points with the right config and, for attribution, the right env and CLI args; the actual embedding is done inside **utils_embed.watermark_folder_selective** (see README for what it’s based on).

---

## 4. Technical details

### Embedding (utils_embed)

- **Bands**: Config `embed.bands`, default `20-300,300-3000,3000-7600` Hz. Audio is split into these bands with a 6th-order bandpass; each band is watermarked separately then summed.
- **Windows**: Length `embed.win_sec` (default 1.5 s). Only windows with RMS ≥ `silence_rel * global_rms` are considered; up to `max_mark_fraction` (default 0.6) of those are randomly chosen for marking.
- **Per-window band**: Among the `top_k` (default 2) most energetic bands for that window, one is chosen at random; the watermark is applied only in that band for that window.
- **Fade**: `embed.fade_ms` (default 50 ms) fade-in/out at window edges.
- **Message**: 16-bit 0/1 string. Either from config `embed.message_bits` or derived by **utils_attribution.derive_message_bits** from secret, api_key, user_id, request_time, nonce.
- **Model**: AudioSeal generator `audioseal_wm_16bits`, with `sample_rate` and `alpha` (default 0.8). Output is blended with the original using the per-band envelope; energy can be preserved per band.

### Attribution (utils_attribution)

- **derive_message_bits(secret_key, api_key, user_id, request_time, nonce, bit_len=16)**  
  Payload string: `api_key=...|user_id=...|request_time=...|nonce=...` → HMAC-SHA256 with secret → 256 bits → first `bit_len` (16) bits returned. Same inputs ⇒ same bits; used to tie the watermark to a user/request.

### Detection (utils_detect)

- **Model**: AudioSeal detector `audioseal_detector_16bits`.
- **Input**: Audio resampled to 16 kHz if needed; `sample_rate=16000` is passed to the detector.
- **Output**: Single probability per file (0–1). Thresholds: e.g. &gt; 0.5 (or 0.7) ⇒ watermark likely present. Clean (unmarked) files should give low probability; marked files (and attacked copies that still contain the watermark) should give higher.

### PESQ (utils_pesq)

- **Metric**: Wideband PESQ (16 kHz). Compares reference (clean) vs degraded (marked or attacked).
- **Usage**: Optional. If `pesq` is not installed or fails to build, all PESQ fields are N/A; the rest of the pipeline still runs.
- **Output**: One score per pair (e.g. clean vs marked). Higher = better quality; typical range for light watermarking on speech is around 3–4.5.

### Attacks (utils_attacks)

- **identity**: Copy (baseline).
- **resample_22k / resample_8k**: Resample to 22.05 kHz or 8 kHz then back to original SR.
- **mp3_128k / mp3_64k**: WAV → MP3 → WAV (requires ffmpeg).
- **noise_20db / noise_10db**: Add white Gaussian noise at 20 dB or 10 dB SNR.
- **lowpass_4k**: Lowpass at 4 kHz (6th-order).
- **volume_08**: Scale by 0.8.

These are used in **run_demo.py** to test whether the watermark is still detectable and how much quality (PESQ) remains after each attack.

---

## 5. Quick reference

| Task              | Command / action |
|-------------------|------------------|
| Batch embed       | `python main.py --step embed` (set paths in config). |
| Batch embed + eval| `python main.py`. |
| Embed with user   | Set `WATERMARK_SECRET`, then `python main.py --step embed --user_id ... --api_key ... --request_time ...`. |
| Single-file demo  | `python run_demo.py --user kamala` (optional: `--demo_audio path/to/file`). |
| Eval only         | `python main.py --step eval`. |
| Config            | **config.yaml** for main; **demo_config.yaml** for run_demo. |

For **how the watermark is added (based on what)**, see the README section *“How the watermark is added (based on what)”*.
