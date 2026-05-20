"""
01_acquire_data.py — Golden Hour Compilation
=============================================
Downloads 15 curated 60-minute surgical files from VitalDB:
  - 5× Baseline (stable, diverse heart rates)
  - 5× Hemorrhage (crash around min 30)
  - 5× Artifact (sensor bumps/disconnects)

Each file is cleaned (ZOH + Butterworth 20Hz) and stripped to Time + PLETH only.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

try:
    import vitaldb
except ImportError:
    print("ERROR: vitaldb package not installed. Run: pip install vitaldb")
    sys.exit(1)

# ── Configuration ──────────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_files")
SAMPLE_RATE = 100           # Hz
INTERVAL = 1.0 / SAMPLE_RATE  # 0.01s
WINDOW_MINUTES = 60
WINDOW_SAMPLES = WINDOW_MINUTES * 60 * SAMPLE_RATE  # 360,000

# Butterworth filter params
BUTTER_ORDER = 4
BUTTER_CUTOFF_HZ = 20  # Low-pass cutoff for PLETH

# Scenario detection thresholds
MBP_STABLE_THRESHOLD = 65   # mmHg — above this = healthy
MBP_CRASH_THRESHOLD = 60    # mmHg — below this = crash
MIN_CRASH_DURATION = 30     # seconds of sustained low MBP to count as crash

# How many cases per scenario
CASES_PER_SCENARIO = 5


# ── Signal Processing ─────────────────────────────────────────────────────────
def apply_butterworth(signal, cutoff_hz, fs=SAMPLE_RATE, order=BUTTER_ORDER):
    """Apply a zero-phase Butterworth low-pass filter."""
    nyquist = fs / 2
    normalized_cutoff = cutoff_hz / nyquist
    sos = butter(order, normalized_cutoff, btype='low', output='sos')
    return sosfiltfilt(sos, signal)


def zero_order_hold(series):
    """Forward-fill then back-fill NaN (zero-order hold)."""
    return series.ffill().bfill()


def compute_heart_rate_proxy(pleth, fs=SAMPLE_RATE, window_sec=10):
    """Estimate rough heart rate from PLETH zero-crossings in a short window."""
    window_samples = window_sec * fs
    segment = pleth[:window_samples]
    segment = segment - np.nanmean(segment)
    # Count zero crossings
    valid = ~np.isnan(segment)
    if valid.sum() < fs:
        return 0
    seg_clean = segment[valid]
    crossings = np.sum(np.diff(np.sign(seg_clean)) != 0)
    # Each heartbeat has ~2 crossings
    beats = crossings / 2
    bpm = beats * (60 / window_sec)
    return bpm


# ── Case Discovery ─────────────────────────────────────────────────────────────
def find_all_candidates():
    """Find VitalDB cases that have BOTH PLETH waveform AND ART_MBP numeric."""
    print("🔍 Searching VitalDB for cases with SNUADC/PLETH + Solar8000/ART_MBP...")
    candidate_ids = vitaldb.find_cases(['SNUADC/PLETH', 'Solar8000/ART_MBP'])
    print(f"   Found {len(candidate_ids)} candidates with both tracks")
    return candidate_ids


def classify_case(cid):
    """
    Classify a VitalDB case using ONLY the lightweight 1Hz MBP download.
    Returns (category, info_dict) or (None, None) if unusable.
    """
    try:
        # Load ONLY MBP at 1Hz (tiny download) for fast classification
        mbp_data = vitaldb.load_case(cid, ['Solar8000/ART_MBP'], 1)
        if mbp_data is None:
            return None, None
        mbp = mbp_data[:, 0]

        total_seconds = len(mbp)
        total_minutes = total_seconds / 60

        # Need at least 60 minutes of data
        if total_minutes < 60:
            return None, None

        mbp_60 = mbp[:3600]  # first 60 minutes only
        valid_mbp = mbp_60[~np.isnan(mbp_60)]

        if len(valid_mbp) < 120:  # need at least 2 min of valid MBP
            return None, None

        mbp_mean_val = np.mean(valid_mbp)

        # ── Split into halves for hemorrhage check ──
        first_half = mbp_60[:1800]
        second_half = mbp_60[1800:3600]
        first_valid = first_half[~np.isnan(first_half)]
        second_valid = second_half[~np.isnan(second_half)]

        has_halves = len(first_valid) > 60 and len(second_valid) > 60

        # ── 1. Check for Hemorrhage FIRST (needs obvious MBP drop) ──
        # Find the worst 60-second crash in the entire surgery
        if total_seconds > 3600:
            import pandas as pd
            # Fast rolling mean over 60 seconds ignoring NaNs
            mbp_series = pd.Series(mbp)
            rolling_60s = mbp_series.rolling(window=60, min_periods=10).mean()
            
            # Find the absolute minimum 60s period
            crash_end_idx = rolling_60s.idxmin()
            
            if pd.notna(crash_end_idx) and rolling_60s[crash_end_idx] < 55:
                # We found a crash. Check the 30 minutes before it.
                crash_start_idx = max(0, crash_end_idx - 60)
                baseline_start_idx = max(0, crash_start_idx - 1800)
                
                # We need at least 30 mins of history to call it a "crash from baseline"
                if crash_start_idx - baseline_start_idx >= 1500:
                    baseline_period = mbp[baseline_start_idx:crash_start_idx]
                    valid_baseline = baseline_period[~np.isnan(baseline_period)]
                    
                    if len(valid_baseline) > 300: # at least 5 mins of valid data in the 30m
                        baseline_mean = np.mean(valid_baseline)
                        crash_mean = rolling_60s[crash_end_idx]
                        
                        if baseline_mean > 70 and (baseline_mean - crash_mean) > 15:
                            # Perfect! We have a stable baseline followed by a crash.
                            # Our 60-minute window ends exactly 5 minutes after the crash
                            window_end = min(total_seconds, crash_end_idx + 300)
                            window_start = max(0, window_end - 3600)
                            
                            return "hemorrhage", {
                                "cid": cid,
                                "total_min": total_minutes,
                                "window_start_s": int(window_start),
                                "baseline_mbp": float(baseline_mean),
                                "crash_mbp": float(crash_mean),
                            }

        # ── 2. Check for Baseline (generally stable MBP) ──
        # 5th percentile > 60 and mean > 70 = stable patient
        if len(valid_mbp) > 600:
            p5 = np.percentile(valid_mbp, 5)
            if p5 > 60 and mbp_mean_val > 70:
                return "baseline", {
                    "cid": cid,
                    "total_min": total_minutes,
                    "window_start_s": 0,
                    "mbp_mean": float(mbp_mean_val),
                    "mbp_p5": float(p5),
                }

        # ── 3. Artifact candidate (MBP is ok-ish, catch-all) ──
        if len(valid_mbp) > 300 and mbp_mean_val > 60:
            return "artifact_candidate", {
                "cid": cid,
                "total_min": total_minutes,
                "window_start_s": 0,
                "mbp_mean": float(mbp_mean_val),
            }

        return None, None

    except Exception as e:
        return None, None


def discover_cases(candidate_ids):
    """Scan candidates using lightweight MBP-only classification."""
    baselines = []
    hemorrhages = []
    artifact_candidates = []

    print(f"\n📊 Scanning {len(candidate_ids)} cases (MBP-only, fast mode)...")
    print("   Need: 5 baseline, 5 hemorrhage, 5 artifact\n")

    for i, cid in enumerate(candidate_ids):
        # Stop early if we have enough
        if (len(baselines) >= CASES_PER_SCENARIO and
            len(hemorrhages) >= CASES_PER_SCENARIO and
            len(artifact_candidates) >= CASES_PER_SCENARIO):
            break

        category, info = classify_case(cid)

        if category == "baseline" and len(baselines) < CASES_PER_SCENARIO:
            baselines.append(info)
            print(f"  ✅ Baseline #{len(baselines)}: case {cid} "
                  f"(MBP={info['mbp_mean']:.0f}, p5={info['mbp_p5']:.0f})")

        elif category == "hemorrhage" and len(hemorrhages) < CASES_PER_SCENARIO:
            hemorrhages.append(info)
            print(f"  🩸 Hemorrhage #{len(hemorrhages)}: case {cid} "
                  f"(baseline={info['baseline_mbp']:.0f}, crash={info['crash_mbp']:.0f})")

        elif category == "artifact_candidate" and len(artifact_candidates) < CASES_PER_SCENARIO:
            artifact_candidates.append(info)
            print(f"  📡 Artifact candidate #{len(artifact_candidates)}: case {cid} "
                  f"(MBP={info['mbp_mean']:.0f})")

        if (i + 1) % 50 == 0:
            print(f"  ... scanned {i+1} cases "
                  f"(B:{len(baselines)} H:{len(hemorrhages)} A:{len(artifact_candidates)})")

    print(f"\n📋 Discovery complete:")
    print(f"   Baselines:         {len(baselines)}/{CASES_PER_SCENARIO}")
    print(f"   Hemorrhages:       {len(hemorrhages)}/{CASES_PER_SCENARIO}")
    print(f"   Artifact cands:    {len(artifact_candidates)}/{CASES_PER_SCENARIO}")

    return baselines, hemorrhages, artifact_candidates


# ── Data Extraction & Cleaning ──────────────────────────────────────────────────
def extract_and_clean(cid, scenario_name, file_index, window_start_s=0):
    """Download 100Hz PLETH, clean, and export a 60-minute file."""
    print(f"\n  📥 Loading case {cid} ({scenario_name}_{file_index:02d}, "
          f"window starts at {window_start_s}s)...")

    # Load PLETH at 100Hz
    pleth_data = vitaldb.load_case(cid, ['SNUADC/PLETH'], INTERVAL)
    if pleth_data is None:
        print(f"  ⚠️  Failed to load PLETH for case {cid}, skipping")
        return None

    # Extract the 60-min window
    start_idx = window_start_s * SAMPLE_RATE
    end_idx = start_idx + WINDOW_SAMPLES

    if end_idx > len(pleth_data):
        # Fall back to first 60 min if window doesn't fit
        start_idx = 0
        end_idx = min(WINDOW_SAMPLES, len(pleth_data))

    pleth = pleth_data[start_idx:end_idx, 0]

    # Pad if shorter than 60 min (replicate last value)
    if len(pleth) < WINDOW_SAMPLES:
        pad_val = pleth[-1] if len(pleth) > 0 and not np.isnan(pleth[-1]) else 0
        pleth = np.pad(pleth, (0, WINDOW_SAMPLES - len(pleth)),
                       mode='constant', constant_values=pad_val)

    # Build DataFrame
    time = np.arange(WINDOW_SAMPLES) * INTERVAL
    time = np.round(time, 4)

    df = pd.DataFrame({
        'Time': time,
        'PLETH': pleth
    })

    # Zero-Order Hold
    df['PLETH'] = zero_order_hold(df['PLETH'])

    # Handle any remaining edge NaN (fill with 0)
    df['PLETH'] = df['PLETH'].fillna(0)

    # Butterworth filter
    df['PLETH'] = apply_butterworth(df['PLETH'].values, BUTTER_CUTOFF_HZ)

    # Validate
    assert len(df) == WINDOW_SAMPLES, f"Expected {WINDOW_SAMPLES} rows, got {len(df)}"
    assert df.isna().sum().sum() == 0, "NaN values remain after cleaning!"
    assert list(df.columns) == ['Time', 'PLETH'], f"Unexpected columns: {list(df.columns)}"

    # Export
    filename = f"{scenario_name}_{file_index:02d}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)

    print(f"  💾 Saved {filename} ({len(df):,} rows, "
          f"PLETH range: [{df['PLETH'].min():.2f}, {df['PLETH'].max():.2f}])")

    return filepath


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("HEMO-SCOUT — Task 1: Golden Hour Compilation")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Find candidates with BOTH PLETH + ART_MBP
    candidate_ids = find_all_candidates()

    if len(candidate_ids) == 0:
        print("\n⚠️  VitalDB returned no candidates. Falling back to synthetic data...")
        generate_synthetic_fallback()
        return

    # Step 2: Classify using lightweight MBP-only scan
    baselines, hemorrhages, artifact_cands = discover_cases(candidate_ids)

    # Verify we found enough
    total_found = len(baselines) + len(hemorrhages) + len(artifact_cands)
    if total_found < 5:
        print(f"\n⚠️  Only found {total_found} cases. Falling back to synthetic data...")
        generate_synthetic_fallback()
        return

    # Step 3: Extract, clean, and export each case (now downloads 100Hz PLETH)
    all_files = []
    print("\n" + "=" * 60)
    print("EXTRACTING & CLEANING DATA (downloading 100Hz PLETH)")
    print("=" * 60)

    for i, info in enumerate(baselines):
        window_start = info.get('window_start_s', 0)
        path = extract_and_clean(info['cid'], 'baseline', i + 1, window_start)
        all_files.append(path)

    for i, info in enumerate(hemorrhages):
        window_start = info.get('window_start_s', 0)
        path = extract_and_clean(info['cid'], 'hemorrhage', i + 1, window_start)
        all_files.append(path)

    for i, info in enumerate(artifact_cands):
        window_start = info.get('window_start_s', 0)
        path = extract_and_clean(info['cid'], 'artifact', i + 1, window_start)
        all_files.append(path)

    # Summary
    print("\n" + "=" * 60)
    print(f"✅ Task 1 Complete — {len(all_files)} files extracted")
    print("=" * 60)
    for f in all_files:
        print(f"   {os.path.basename(f)}")


def generate_synthetic_fallback():
    """
    Generate synthetic data if VitalDB is unreachable.
    This creates realistic-looking PLETH waveforms for testing the pipeline.
    """
    print("\n🔧 Generating synthetic 60-minute PLETH waveforms...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t = np.arange(WINDOW_SAMPLES) * INTERVAL  # time in seconds
    time_col = np.round(t, 4)

    def make_pleth_wave(t_sec, heart_rate_bpm=72, amplitude=1.0):
        """Generate a synthetic arterial pressure-like waveform."""
        freq = heart_rate_bpm / 60.0
        # Main pulse
        wave = amplitude * (
            0.6 * np.sin(2 * np.pi * freq * t_sec) +
            0.3 * np.sin(2 * np.pi * 2 * freq * t_sec + 0.5) +
            0.1 * np.sin(2 * np.pi * 3 * freq * t_sec + 1.0)
        )
        # Add slight noise
        wave += np.random.normal(0, 0.02 * amplitude, len(t_sec))
        return wave

    # ── Baselines (5 different heart rates) ──
    heart_rates = [60, 68, 72, 80, 95]
    for i, hr in enumerate(heart_rates):
        pleth = make_pleth_wave(t, heart_rate_bpm=hr, amplitude=1.0)
        pleth = apply_butterworth(pleth, BUTTER_CUTOFF_HZ)
        df = pd.DataFrame({'Time': time_col, 'PLETH': pleth})
        df.to_csv(os.path.join(OUTPUT_DIR, f"baseline_{i+1:02d}.csv"), index=False)
        print(f"  ✅ baseline_{i+1:02d}.csv (HR={hr} bpm)")

    # ── Hemorrhages (crash around min 30) ──
    crash_rates = [72, 65, 80, 75, 70]
    for i, hr in enumerate(crash_rates):
        pleth = make_pleth_wave(t, heart_rate_bpm=hr, amplitude=1.0)
        # Apply amplitude decay starting around minute 25-30
        crash_start = int((25 + i) * 60 * SAMPLE_RATE)
        crash_ramp = np.ones(WINDOW_SAMPLES)
        ramp_length = WINDOW_SAMPLES - crash_start
        crash_ramp[crash_start:] = np.linspace(1.0, 0.15, ramp_length)
        pleth *= crash_ramp
        pleth = apply_butterworth(pleth, BUTTER_CUTOFF_HZ)
        df = pd.DataFrame({'Time': time_col, 'PLETH': pleth})
        df.to_csv(os.path.join(OUTPUT_DIR, f"hemorrhage_{i+1:02d}.csv"), index=False)
        print(f"  🩸 hemorrhage_{i+1:02d}.csv (HR={hr}, crash@min {25+i})")

    # ── Artifacts (sensor bumps/flatlines) ──
    for i in range(5):
        hr = 72 + i * 3
        pleth = make_pleth_wave(t, heart_rate_bpm=hr, amplitude=1.0)
        # Insert artifact episodes
        artifact_start = int((15 + i * 5) * 60 * SAMPLE_RATE)
        artifact_duration = int((3 + i) * SAMPLE_RATE)  # 3-7 seconds of artifact
        if i % 2 == 0:
            # Flatline artifact
            pleth[artifact_start:artifact_start + artifact_duration] = 0
        else:
            # Spike artifact
            pleth[artifact_start:artifact_start + artifact_duration] = (
                np.random.uniform(-5, 5, artifact_duration)
            )
        pleth = apply_butterworth(pleth, BUTTER_CUTOFF_HZ)
        df = pd.DataFrame({'Time': time_col, 'PLETH': pleth})
        df.to_csv(os.path.join(OUTPUT_DIR, f"artifact_{i+1:02d}.csv"), index=False)
        print(f"  📡 artifact_{i+1:02d}.csv (HR={hr}, artifact@min {15+i*5})")

    print(f"\n✅ Synthetic fallback complete — 15 files generated")


if __name__ == "__main__":
    main()
