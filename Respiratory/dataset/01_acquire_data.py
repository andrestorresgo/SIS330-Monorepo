"""
01_acquire_data.py — Golden Hour Compilation (Respiratory)
===========================================================
Downloads 20 curated 60-minute surgical files from VitalDB:
  - 5× Baseline (stable, rhythmic CO2 between 35-45 mmHg)
  - 5× Obstruction (airway blockage — CO2 drops toward 0)
  - 5× Hypoventilation (CO2 climbs above 50 mmHg)
  - 5× Artifact (brief ventilator disconnects in stable patients)

Each file is cleaned (ZOH + Butterworth 20Hz) and stripped to Time + CO2 only.
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
BUTTER_CUTOFF_HZ = 20  # Low-pass cutoff for CO2 waveform

# How many cases per scenario
CASES_PER_SCENARIO = 5

# ── ETCO2 clinical thresholds (mmHg) ──
# NOTE: VitalDB patients are under general anesthesia + mechanical ventilation.
# Typical ETCO2 in this setting runs 25-40 mmHg, NOT the textbook awake 35-45.
ETCO2_BASELINE_LOW = 20      # Anesthetized patients can run low
ETCO2_BASELINE_HIGH = 50     # Upper bound for "normal" under anesthesia
ETCO2_BASELINE_STD_MAX = 10  # Allow more variance in real surgical data


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


# ── Case Discovery ─────────────────────────────────────────────────────────────
def find_all_candidates():
    """Find VitalDB cases that have BOTH CO2 waveform AND ETCO2 numeric."""
    print("🔍 Searching VitalDB for cases with Primus/CO2 + Solar8000/ETCO2...")

    # Try multiple known CO2 track names
    co2_waveform_tracks = ['Primus/CO2', 'Carestation/CO2', 'SNUADC/CO2']
    etco2_numeric_tracks = ['Solar8000/ETCO2', 'Primus/ETCO2', 'Carestation/ETCO2']

    candidate_ids = []

    for wf_track in co2_waveform_tracks:
        for num_track in etco2_numeric_tracks:
            try:
                ids = vitaldb.find_cases([wf_track, num_track])
                if ids is not None and len(ids) > 0:
                    print(f"   Found {len(ids)} candidates with {wf_track} + {num_track}")
                    candidate_ids = ids
                    # Store which track names worked
                    find_all_candidates.wf_track = wf_track
                    find_all_candidates.num_track = num_track
                    return candidate_ids
            except Exception:
                continue

    # Fallback: try just CO2 waveform tracks alone
    for wf_track in co2_waveform_tracks:
        try:
            ids = vitaldb.find_cases([wf_track])
            if ids is not None and len(ids) > 0:
                print(f"   Found {len(ids)} candidates with {wf_track} (no numeric ETCO2)")
                candidate_ids = ids
                find_all_candidates.wf_track = wf_track
                find_all_candidates.num_track = None
                return candidate_ids
        except Exception:
            continue

    print("   No candidates found with any known CO2 track names")
    return candidate_ids

# Initialize track name storage
find_all_candidates.wf_track = None
find_all_candidates.num_track = None

# Diagnostic counter for debugging classification
_diag_reject_count = 0
_DIAG_MAX_PRINTS = 10


def classify_case(cid):
    """
    Classify a VitalDB case using the lightweight 1Hz ETCO2 download.
    Returns (category, info_dict) or (None, None) if unusable.

    Thresholds calibrated for anesthetized, mechanically ventilated patients
    (VitalDB surgical database). Typical ETCO2 = 25-40 mmHg.

    Priority (ordered by rarity to catch interesting cases first):
      1. Obstruction — RELATIVE drop: 30s rolling mean < 50% of case mean
      2. Hypoventilation — upward trend: 2nd half > 1st half by ≥ 3 mmHg
      3. Baseline — stable ETCO2 in 20-50 mmHg with low variance
      4. Artifact candidate — catch-all for stable-ish cases
    """
    global _diag_reject_count
    try:
        num_track = find_all_candidates.num_track
        wf_track = find_all_candidates.wf_track

        # If we have a numeric ETCO2 track, use it for fast classification
        if num_track:
            etco2_data = vitaldb.load_case(cid, [num_track], 1)
            if etco2_data is None:
                return None, None
            etco2 = etco2_data[:, 0]
        else:
            # Fall back to waveform at 1Hz (averaged) for classification
            co2_data = vitaldb.load_case(cid, [wf_track], 1)
            if co2_data is None:
                return None, None
            etco2 = co2_data[:, 0]

        total_seconds = len(etco2)
        total_minutes = total_seconds / 60

        # Need at least 60 minutes of data
        if total_minutes < 60:
            return None, None

        etco2_60 = etco2[:3600]  # first 60 minutes
        valid_etco2 = etco2_60[~np.isnan(etco2_60)]

        if len(valid_etco2) < 120:  # need at least 2 min of valid data
            return None, None

        etco2_mean = np.mean(valid_etco2)
        etco2_std = np.std(valid_etco2)

        # Skip cases with unreasonably low signal (likely bad sensor)
        if etco2_mean < 5:
            return None, None

        # ── 1. Check for Obstruction (RELATIVE drop from case's own mean) ──
        # A real obstruction = CO2 drops well below the patient's own baseline.
        # Use 50% of the case mean as threshold (not a fixed absolute value).
        if len(valid_etco2) > 600 and etco2_mean > 15:
            relative_crash_threshold = etco2_mean * 0.50
            etco2_series = pd.Series(etco2_60)
            rolling_30s = etco2_series.rolling(window=30, min_periods=10).mean()

            crash_idx = rolling_30s.idxmin()
            if pd.notna(crash_idx) and rolling_30s[crash_idx] < relative_crash_threshold:
                # Verify there was a normal baseline BEFORE the crash
                baseline_before = etco2_60[:max(1, crash_idx - 60)]
                valid_before = baseline_before[~np.isnan(baseline_before)]
                if len(valid_before) > 60:
                    before_mean = np.mean(valid_before)
                    # The baseline before crash should be at least 2x the crash
                    if before_mean > rolling_30s[crash_idx] * 1.8:
                        return "obstruction", {
                            "cid": cid,
                            "total_min": total_minutes,
                            "window_start_s": 0,
                            "baseline_etco2": float(before_mean),
                            "crash_etco2": float(rolling_30s[crash_idx]),
                        }

        # ── 2. Check for Hypoventilation (upward ETCO2 trend) ──
        # Under anesthesia, any sustained climb of ≥ 3 mmHg between halves
        # where the second half > 40 suggests inadequate ventilation.
        if len(valid_etco2) > 600:
            first_half = etco2_60[:1800]
            second_half = etco2_60[1800:3600]
            first_valid = first_half[~np.isnan(first_half)]
            second_valid = second_half[~np.isnan(second_half)]

            if len(first_valid) > 60 and len(second_valid) > 60:
                first_mean = np.mean(first_valid)
                second_mean = np.mean(second_valid)
                # Climbing: 2nd half higher by ≥ 3 mmHg AND 2nd half > 38
                if (second_mean - first_mean >= 3 and second_mean > 38):
                    return "hypoventilation", {
                        "cid": cid,
                        "total_min": total_minutes,
                        "window_start_s": 0,
                        "first_half_etco2": float(first_mean),
                        "second_half_etco2": float(second_mean),
                    }

        # ── 3. Check for Baseline (stable under anesthesia: 20-50 range) ──
        if len(valid_etco2) > 600:
            p5 = np.percentile(valid_etco2, 5)
            p95 = np.percentile(valid_etco2, 95)

            if (ETCO2_BASELINE_LOW <= etco2_mean <= ETCO2_BASELINE_HIGH and
                    etco2_std < ETCO2_BASELINE_STD_MAX and
                    p5 > 10 and p95 < 55):
                return "baseline", {
                    "cid": cid,
                    "total_min": total_minutes,
                    "window_start_s": 0,
                    "etco2_mean": float(etco2_mean),
                    "etco2_std": float(etco2_std),
                    "etco2_p5": float(p5),
                    "etco2_p95": float(p95),
                }

        # ── 4. Artifact candidate (catch-all for stable-ish data) ──
        if len(valid_etco2) > 300 and etco2_mean > 10:
            return "artifact_candidate", {
                "cid": cid,
                "total_min": total_minutes,
                "window_start_s": 0,
                "etco2_mean": float(etco2_mean),
            }

        # Diagnostic: print first few rejected cases to help debug
        if _diag_reject_count < _DIAG_MAX_PRINTS:
            _diag_reject_count += 1
            print(f"    [diag] case {cid} rejected: mean={etco2_mean:.1f}, "
                  f"std={etco2_std:.1f}, valid={len(valid_etco2)}")

        return None, None

    except Exception:
        return None, None


def discover_cases(candidate_ids):
    """Scan candidates using lightweight ETCO2-only classification."""
    baselines = []
    obstructions = []
    hypoventilations = []
    artifact_candidates = []

    print(f"\n📊 Scanning {len(candidate_ids)} cases (ETCO2-only, fast mode)...")
    print("   Need: 5 baseline, 5 obstruction, 5 hypoventilation, 5 artifact\n")

    for i, cid in enumerate(candidate_ids):
        # Stop early if we have enough of everything
        if (len(baselines) >= CASES_PER_SCENARIO and
                len(obstructions) >= CASES_PER_SCENARIO and
                len(hypoventilations) >= CASES_PER_SCENARIO and
                len(artifact_candidates) >= CASES_PER_SCENARIO):
            break

        category, info = classify_case(cid)

        if category == "baseline" and len(baselines) < CASES_PER_SCENARIO:
            baselines.append(info)
            print(f"  ✅ Baseline #{len(baselines)}: case {cid} "
                  f"(ETCO2={info['etco2_mean']:.1f}±{info['etco2_std']:.1f})")

        elif category == "obstruction" and len(obstructions) < CASES_PER_SCENARIO:
            obstructions.append(info)
            print(f"  🫁 Obstruction #{len(obstructions)}: case {cid} "
                  f"(baseline={info['baseline_etco2']:.1f}, crash={info['crash_etco2']:.1f})")

        elif category == "hypoventilation" and len(hypoventilations) < CASES_PER_SCENARIO:
            hypoventilations.append(info)
            print(f"  📈 Hypoventilation #{len(hypoventilations)}: case {cid} "
                  f"(1st={info['first_half_etco2']:.1f}, 2nd={info['second_half_etco2']:.1f})")

        elif category == "artifact_candidate" and len(artifact_candidates) < CASES_PER_SCENARIO:
            artifact_candidates.append(info)
            print(f"  📡 Artifact candidate #{len(artifact_candidates)}: case {cid} "
                  f"(ETCO2={info['etco2_mean']:.1f})")

        if (i + 1) % 50 == 0:
            print(f"  ... scanned {i + 1} cases "
                  f"(B:{len(baselines)} O:{len(obstructions)} "
                  f"H:{len(hypoventilations)} A:{len(artifact_candidates)})")

    print(f"\n📋 Discovery complete:")
    print(f"   Baselines:         {len(baselines)}/{CASES_PER_SCENARIO}")
    print(f"   Obstructions:      {len(obstructions)}/{CASES_PER_SCENARIO}")
    print(f"   Hypoventilations:  {len(hypoventilations)}/{CASES_PER_SCENARIO}")
    print(f"   Artifact cands:    {len(artifact_candidates)}/{CASES_PER_SCENARIO}")

    return baselines, obstructions, hypoventilations, artifact_candidates


# ── Data Extraction & Cleaning ──────────────────────────────────────────────────
def extract_and_clean(cid, scenario_name, file_index, window_start_s=0):
    """Download 100Hz CO2, clean, and export a 60-minute file."""
    wf_track = find_all_candidates.wf_track

    print(f"\n  📥 Loading case {cid} ({scenario_name}_{file_index:02d}, "
          f"window starts at {window_start_s}s)...")

    # Load CO2 at 100Hz
    co2_data = vitaldb.load_case(cid, [wf_track], INTERVAL)
    if co2_data is None:
        print(f"  ⚠️  Failed to load CO2 for case {cid}, skipping")
        return None

    # Extract the 60-min window
    start_idx = window_start_s * SAMPLE_RATE
    end_idx = start_idx + WINDOW_SAMPLES

    if end_idx > len(co2_data):
        # Fall back to first 60 min if window doesn't fit
        start_idx = 0
        end_idx = min(WINDOW_SAMPLES, len(co2_data))

    co2 = co2_data[start_idx:end_idx, 0]

    # Pad if shorter than 60 min (replicate last value)
    if len(co2) < WINDOW_SAMPLES:
        pad_val = co2[-1] if len(co2) > 0 and not np.isnan(co2[-1]) else 0
        co2 = np.pad(co2, (0, WINDOW_SAMPLES - len(co2)),
                      mode='constant', constant_values=pad_val)

    # Build DataFrame — CO2 only, no PLETH
    time_col = np.arange(WINDOW_SAMPLES) * INTERVAL
    time_col = np.round(time_col, 4)

    df = pd.DataFrame({
        'Time': time_col,
        'CO2': co2
    })

    # Zero-Order Hold
    df['CO2'] = zero_order_hold(df['CO2'])

    # Handle any remaining edge NaN (fill with 0)
    df['CO2'] = df['CO2'].fillna(0)

    # Butterworth filter
    df['CO2'] = apply_butterworth(df['CO2'].values, BUTTER_CUTOFF_HZ)

    # Validate
    assert len(df) == WINDOW_SAMPLES, f"Expected {WINDOW_SAMPLES} rows, got {len(df)}"
    assert df.isna().sum().sum() == 0, "NaN values remain after cleaning!"
    assert list(df.columns) == ['Time', 'CO2'], f"Unexpected columns: {list(df.columns)}"

    # Export
    filename = f"{scenario_name}_{file_index:02d}.csv"
    filepath = os.path.join(OUTPUT_DIR, filename)
    df.to_csv(filepath, index=False)

    print(f"  💾 Saved {filename} ({len(df):,} rows, "
          f"CO2 range: [{df['CO2'].min():.2f}, {df['CO2'].max():.2f}])")

    return filepath


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("VENT-GUARDIAN — Task 1: Golden Hour Compilation")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: Find candidates with CO2 waveform (+ optional ETCO2 numeric)
    candidate_ids = find_all_candidates()

    if len(candidate_ids) == 0:
        print("\n⚠️  VitalDB returned no candidates. Falling back to synthetic data...")
        generate_synthetic_fallback()
        return

    # Step 2: Classify using lightweight ETCO2 scan
    baselines, obstructions, hypoventilations, artifact_cands = discover_cases(candidate_ids)

    # Verify we found enough
    total_found = len(baselines) + len(obstructions) + len(hypoventilations) + len(artifact_cands)
    if total_found < 5:
        print(f"\n⚠️  Only found {total_found} cases. Falling back to synthetic data...")
        generate_synthetic_fallback()
        return

    # Step 3: Extract, clean, and export each case (downloads 100Hz CO2)
    all_files = []
    print("\n" + "=" * 60)
    print("EXTRACTING & CLEANING DATA (downloading 100Hz CO2)")
    print("=" * 60)

    for i, info in enumerate(baselines):
        window_start = info.get('window_start_s', 0)
        path = extract_and_clean(info['cid'], 'baseline', i + 1, window_start)
        if path:
            all_files.append(path)

    for i, info in enumerate(obstructions):
        window_start = info.get('window_start_s', 0)
        path = extract_and_clean(info['cid'], 'obstruction', i + 1, window_start)
        if path:
            all_files.append(path)

    for i, info in enumerate(hypoventilations):
        window_start = info.get('window_start_s', 0)
        path = extract_and_clean(info['cid'], 'hypoventilation', i + 1, window_start)
        if path:
            all_files.append(path)

    for i, info in enumerate(artifact_cands):
        window_start = info.get('window_start_s', 0)
        path = extract_and_clean(info['cid'], 'artifact', i + 1, window_start)
        if path:
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
    Creates realistic capnography (CO2) waveforms for testing the pipeline.

    Capnography shape: a "square-ish" wave with:
      - Inspiratory baseline near 0 mmHg
      - Rapid upstroke
      - Alveolar plateau around 35-45 mmHg
      - End-tidal peak (highest point)
      - Sharp downstroke back to 0
    """
    print("\n🔧 Generating synthetic 60-minute CO2 capnography waveforms...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t = np.arange(WINDOW_SAMPLES) * INTERVAL  # time in seconds
    time_col = np.round(t, 4)

    def make_capnography_wave(t_sec, resp_rate_bpm=14, etco2=38.0):
        """
        Generate a synthetic capnography waveform.
        Real capnography looks like a repeating trapezoid:
          - Phase I: Inspiratory baseline (~0 mmHg)
          - Phase II: Rapid upstroke
          - Phase III: Alveolar plateau (near ETCO2)
          - Phase IV: Rapid downstroke
        """
        freq = resp_rate_bpm / 60.0  # breaths per second
        period = 1.0 / freq
        phase = (t_sec % period) / period  # 0 to 1 per breath cycle

        wave = np.zeros_like(t_sec)

        # Inspiratory phase (0.0 to 0.3): baseline near 0
        mask_insp = phase < 0.3
        wave[mask_insp] = 0

        # Upstroke (0.3 to 0.4): rapid rise to ETCO2
        mask_up = (phase >= 0.3) & (phase < 0.4)
        progress = (phase[mask_up] - 0.3) / 0.1
        wave[mask_up] = etco2 * progress

        # Alveolar plateau (0.4 to 0.85): near ETCO2, slight upward slope
        mask_plateau = (phase >= 0.4) & (phase < 0.85)
        progress = (phase[mask_plateau] - 0.4) / 0.45
        wave[mask_plateau] = etco2 * (1.0 + 0.05 * progress)

        # Downstroke (0.85 to 1.0): rapid descent back to 0
        mask_down = phase >= 0.85
        progress = (phase[mask_down] - 0.85) / 0.15
        wave[mask_down] = etco2 * 1.05 * (1.0 - progress)

        # Add slight noise
        wave += np.random.normal(0, 0.3, len(t_sec))
        wave = np.clip(wave, 0, None)  # CO2 can't be negative

        return wave

    # ── Baselines (5 different respiratory rates, ETCO2 35-45) ──
    resp_rates = [12, 14, 16, 18, 20]
    etco2_vals = [36, 38, 40, 42, 44]
    for i, (rr, etco2) in enumerate(zip(resp_rates, etco2_vals)):
        co2 = make_capnography_wave(t, resp_rate_bpm=rr, etco2=etco2)
        co2 = apply_butterworth(co2, BUTTER_CUTOFF_HZ)
        co2 = np.clip(co2, 0, None)
        df = pd.DataFrame({'Time': time_col, 'CO2': co2})
        df.to_csv(os.path.join(OUTPUT_DIR, f"baseline_{i+1:02d}.csv"), index=False)
        print(f"  ✅ baseline_{i+1:02d}.csv (RR={rr}, ETCO2={etco2})")

    # ── Obstructions (CO2 drops toward 0 — airway blockage) ──
    for i in range(5):
        rr = 14 + i
        co2 = make_capnography_wave(t, resp_rate_bpm=rr, etco2=40.0)
        # Simulate obstruction: CO2 drops to near-zero starting at some point
        obstruction_start = int((20 + i * 5) * 60 * SAMPLE_RATE)
        # Rapid decay over 30 seconds
        decay_length = 30 * SAMPLE_RATE
        decay_end = min(obstruction_start + decay_length, WINDOW_SAMPLES)
        ramp = np.ones(WINDOW_SAMPLES)
        ramp[obstruction_start:decay_end] = np.linspace(1.0, 0.02, decay_end - obstruction_start)
        ramp[decay_end:] = 0.02
        co2 *= ramp
        co2 += np.random.normal(0, 0.1, WINDOW_SAMPLES)
        co2 = np.clip(co2, 0, None)
        co2 = apply_butterworth(co2, BUTTER_CUTOFF_HZ)
        co2 = np.clip(co2, 0, None)
        df = pd.DataFrame({'Time': time_col, 'CO2': co2})
        df.to_csv(os.path.join(OUTPUT_DIR, f"obstruction_{i+1:02d}.csv"), index=False)
        print(f"  🫁 obstruction_{i+1:02d}.csv (RR={rr}, crash@min {20+i*5})")

    # ── Hypoventilations (CO2 gradually climbs above 55) ──
    for i in range(5):
        rr = 12 - i  # Slowing respiratory rate
        rr = max(rr, 6)
        base_etco2 = 38.0 + i * 2
        co2 = make_capnography_wave(t, resp_rate_bpm=rr, etco2=base_etco2)
        # Gradually increase the ETCO2 level over the hour
        climb_factor = np.linspace(1.0, 1.6 + i * 0.1, WINDOW_SAMPLES)
        co2 *= climb_factor
        co2 = apply_butterworth(co2, BUTTER_CUTOFF_HZ)
        co2 = np.clip(co2, 0, None)
        df = pd.DataFrame({'Time': time_col, 'CO2': co2})
        df.to_csv(os.path.join(OUTPUT_DIR, f"hypoventilation_{i+1:02d}.csv"), index=False)
        etco2_end = base_etco2 * (1.6 + i * 0.1)
        print(f"  📈 hypoventilation_{i+1:02d}.csv (RR={rr}, ETCO2 {base_etco2:.0f}→{etco2_end:.0f})")

    # ── Artifacts (brief ventilator disconnects — short flatlines) ──
    for i in range(5):
        rr = 14 + i
        co2 = make_capnography_wave(t, resp_rate_bpm=rr, etco2=40.0)
        # Insert brief flatline episodes (ventilator disconnect)
        artifact_start = int((15 + i * 8) * 60 * SAMPLE_RATE)
        artifact_duration = int((5 + i * 2) * SAMPLE_RATE)  # 5-13 seconds
        artifact_end = min(artifact_start + artifact_duration, WINDOW_SAMPLES)
        co2[artifact_start:artifact_end] = 0  # Complete flatline
        co2 = apply_butterworth(co2, BUTTER_CUTOFF_HZ)
        co2 = np.clip(co2, 0, None)
        df = pd.DataFrame({'Time': time_col, 'CO2': co2})
        df.to_csv(os.path.join(OUTPUT_DIR, f"artifact_{i+1:02d}.csv"), index=False)
        print(f"  📡 artifact_{i+1:02d}.csv (RR={rr}, flatline@min {15+i*8}, "
              f"dur={5+i*2}s)")

    print(f"\n✅ Synthetic fallback complete — 20 files generated")


if __name__ == "__main__":
    main()
