"""
02_label_data.py — Time-Machine Labeling
==========================================
Reads each cleaned CSV and adds a binary Label column:
  - Label 0 = Safe (stable future)
  - Label 1 = Crash Imminent (amplitude crash in 5 minutes)

Uses a 30,000-row lookahead (5 min at 100Hz) to detect future PLETH amplitude drops.
Truncates the last 5 minutes of each file (their future doesn't exist).
"""

import os
import glob
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_files")
LABELED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labeled_files")

SAMPLE_RATE = 100
LOOKAHEAD_SECONDS = 5 * 60      # 5 minutes
LOOKAHEAD_ROWS = LOOKAHEAD_SECONDS * SAMPLE_RATE  # 30,000
FUTURE_WINDOW = 100             # 1 second of future data to evaluate

# Baseline calibration
BASELINE_SECONDS = 120          # Use first 2 minutes to establish baseline
BASELINE_SAMPLES = BASELINE_SECONDS * SAMPLE_RATE  # 12,000

# Crash detection threshold
AMPLITUDE_DROP_RATIO = 0.50     # Future < 50% of baseline = crash


def compute_rolling_amplitude(pleth, window_size=SAMPLE_RATE):
    """Compute rolling peak-to-peak amplitude (proxy for pulse pressure)."""
    n = len(pleth)
    amplitudes = np.zeros(n)

    # Use vectorized rolling computation
    half = window_size // 2
    for i in range(half, n - half):
        segment = pleth[i - half:i + half]
        amplitudes[i] = np.max(segment) - np.min(segment)

    # Fill edges
    amplitudes[:half] = amplitudes[half]
    amplitudes[n - half:] = amplitudes[n - half - 1]

    return amplitudes


def label_file(filepath):
    """
    Read a cleaned CSV and generate time-machine labels.

    For each row i:
      1. Look at the 1-second window at row (i + 30,000)
      2. Compare its PLETH amplitude to the file's baseline
      3. If amplitude < 50% of baseline → Label = 1 (Crash Imminent)
      4. Otherwise → Label = 0 (Safe)

    The last 30,000 rows are discarded.
    """
    filename = os.path.basename(filepath)
    print(f"\n  📋 Labeling {filename}...")

    df = pd.read_csv(filepath)
    pleth = df['PLETH'].values
    n = len(pleth)

    # ── Step 1: Establish baseline amplitude from first 2 minutes ──
    baseline_segment = pleth[:BASELINE_SAMPLES]
    baseline_amplitude = np.max(baseline_segment) - np.min(baseline_segment)

    if baseline_amplitude < 1e-6:
        # Edge case: flat signal — use absolute threshold
        print(f"    ⚠️  Flat baseline detected, using absolute threshold")
        baseline_amplitude = 1.0  # Default to prevent div-by-zero

    crash_threshold = baseline_amplitude * AMPLITUDE_DROP_RATIO
    print(f"    Baseline amp: {baseline_amplitude:.4f}")
    print(f"    Crash threshold (<{AMPLITUDE_DROP_RATIO*100:.0f}%): {crash_threshold:.4f}")

    # ── Step 2: Label every row using lookahead ──
    # We can only label rows 0 to (n - LOOKAHEAD_ROWS - FUTURE_WINDOW)
    max_labelable = n - LOOKAHEAD_ROWS - FUTURE_WINDOW
    labels = np.zeros(max_labelable, dtype=np.int32)

    # Vectorized: compute amplitude at every future window position
    future_start_indices = np.arange(max_labelable) + LOOKAHEAD_ROWS

    for i in range(max_labelable):
        future_idx = future_start_indices[i]
        future_window = pleth[future_idx:future_idx + FUTURE_WINDOW]
        future_amp = np.max(future_window) - np.min(future_window)

        if future_amp < crash_threshold:
            labels[i] = 1

    # ── Step 3: Truncate — discard last 5 min (and the tail) ──
    df_labeled = df.iloc[:max_labelable].copy()
    df_labeled['Label'] = labels

    # ── Stats ──
    n_safe = np.sum(labels == 0)
    n_crash = np.sum(labels == 1)
    total = len(labels)

    print(f"    Total labeled rows: {total:,}")
    print(f"    Safe (0): {n_safe:,} ({n_safe/total*100:.1f}%)")
    print(f"    Crash (1): {n_crash:,} ({n_crash/total*100:.1f}%)")

    # ── Export ──
    output_name = f"labeled_{filename}"
    output_path = os.path.join(LABELED_DIR, output_name)
    df_labeled.to_csv(output_path, index=False)
    print(f"    💾 Saved {output_name}")

    return output_path, {'safe': n_safe, 'crash': n_crash, 'total': total}


def main():
    print("=" * 60)
    print("HEMO-SCOUT — Task 2: Time-Machine Labeling")
    print("=" * 60)

    os.makedirs(LABELED_DIR, exist_ok=True)

    # Find all raw CSV files
    csv_files = sorted(glob.glob(os.path.join(RAW_DIR, "*.csv")))
    if not csv_files:
        print("❌ No CSV files found in raw_files/. Run 01_acquire_data.py first.")
        return

    print(f"Found {len(csv_files)} files to label")

    # Process each file
    all_stats = {}
    for filepath in csv_files:
        name = os.path.basename(filepath)
        output_path, stats = label_file(filepath)
        all_stats[name] = stats

    # ── Summary ──
    print("\n" + "=" * 60)
    print("LABELING SUMMARY")
    print("=" * 60)
    print(f"{'File':<30} {'Safe':>8} {'Crash':>8} {'Total':>8} {'Crash%':>7}")
    print("-" * 65)

    total_safe = 0
    total_crash = 0
    for name, stats in all_stats.items():
        crash_pct = stats['crash'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {name:<28} {stats['safe']:>8,} {stats['crash']:>8,} "
              f"{stats['total']:>8,} {crash_pct:>6.1f}%")
        total_safe += stats['safe']
        total_crash += stats['crash']

    grand_total = total_safe + total_crash
    print("-" * 65)
    print(f"  {'TOTAL':<28} {total_safe:>8,} {total_crash:>8,} "
          f"{grand_total:>8,} {total_crash/grand_total*100:>6.1f}%")
    print("\n✅ Task 2 Complete")


if __name__ == "__main__":
    main()
