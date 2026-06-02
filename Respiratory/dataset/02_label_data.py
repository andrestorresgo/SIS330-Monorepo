"""
02_label_data.py — Time-Machine Labeling (Respiratory)
========================================================
Reads each cleaned CO2 CSV and adds a binary Label column:
  - Label 0 = Safe (normal CO2 in 5 minutes)
  - Label 1 = Respiratory Crisis Imminent (dangerous CO2 in 5 minutes)

Uses a 30,000-row lookahead (5 min at 100Hz) to detect future ETCO2 anomalies.
Danger is BOTH directions:
  - ETCO2 peak < 15 mmHg → no breathing (obstruction/apnea)
  - ETCO2 peak > 55 mmHg → toxic CO2 buildup (hypoventilation)

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

# Respiratory danger thresholds (mmHg)
# Normal ETCO2 is roughly 35-45 mmHg
ETCO2_DANGER_LOW = 15    # Below this = no effective breathing
ETCO2_DANGER_HIGH = 55   # Above this = toxic CO2 buildup


def label_file(filepath):
    """
    Read a cleaned CO2 CSV and generate time-machine labels.

    For each row i:
      1. Look at the 1-second window at row (i + 30,000)
      2. Find the PEAK CO2 in that window (end-tidal CO2 proxy)
      3. If peak < 15 mmHg OR peak > 55 mmHg → Label = 1 (Crisis Imminent)
      4. Otherwise → Label = 0 (Safe)

    The last 30,000 + 100 rows are discarded (can't see their future).
    """
    filename = os.path.basename(filepath)
    print(f"\n  📋 Labeling {filename}...")

    df = pd.read_csv(filepath)
    co2 = df['CO2'].values
    n = len(co2)

    # ── Step 1: Label every row using lookahead ──
    max_labelable = n - LOOKAHEAD_ROWS - FUTURE_WINDOW
    if max_labelable <= 0:
        print(f"    ⚠️  File too short ({n} rows), skipping")
        return None, None

    labels = np.zeros(max_labelable, dtype=np.int32)

    # Tighter lookahead — only label as crisis if the danger window
    # starts within 4 minutes (not the full 5). The last 60 seconds of
    # the original lookahead caused too many "normal-looking but
    # technically pre-crisis" labels.
    EFFECTIVE_LOOKAHEAD = int(4 * 60 * SAMPLE_RATE)   # 24,000 rows = 4 minutes

    for i in range(max_labelable):
        # ── Current state gate ──
        # If the waveform right now is already flatlined, it's either an
        # artifact or active crisis — never label it Safe.
        current_window = co2[i:i + FUTURE_WINDOW]
        current_peak = np.max(current_window)

        # ── Future danger check (tightened to 4-minute window) ──
        future_idx = i + EFFECTIVE_LOOKAHEAD
        future_window = co2[future_idx:future_idx + FUTURE_WINDOW]
        future_peak = np.max(future_window)

        if future_peak < ETCO2_DANGER_LOW or future_peak > ETCO2_DANGER_HIGH:
            labels[i] = 1

        # Override to crisis if current state is already flatlined
        # (catches sensor-verified obstructions that lookahead would miss)
        if current_peak < ETCO2_DANGER_LOW:
            labels[i] = 1

    # ── Step 2: Truncate — discard last 5 min (and the tail) ──
    df_labeled = df.iloc[:max_labelable].copy()
    df_labeled['Label'] = labels

    # ── Stats ──
    n_safe = np.sum(labels == 0)
    n_crisis = np.sum(labels == 1)
    total = len(labels)

    print(f"    Danger thresholds: ETCO2 < {ETCO2_DANGER_LOW} OR > {ETCO2_DANGER_HIGH} mmHg")
    print(f"    Total labeled rows: {total:,}")
    print(f"    Safe (0): {n_safe:,} ({n_safe/total*100:.1f}%)")
    print(f"    Crisis (1): {n_crisis:,} ({n_crisis/total*100:.1f}%)")

    # ── Export ──
    output_name = f"labeled_{filename}"
    output_path = os.path.join(LABELED_DIR, output_name)
    df_labeled.to_csv(output_path, index=False)
    print(f"    💾 Saved {output_name}")

    return output_path, {'safe': n_safe, 'crisis': n_crisis, 'total': total}


def main():
    print("=" * 60)
    print("VENT-GUARDIAN — Task 2: Time-Machine Labeling")
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
        result = label_file(filepath)
        if result[0] is not None:
            _, stats = result
            all_stats[name] = stats

    # ── Summary ──
    print("\n" + "=" * 60)
    print("LABELING SUMMARY")
    print("=" * 60)
    print(f"{'File':<35} {'Safe':>8} {'Crisis':>8} {'Total':>8} {'Crisis%':>8}")
    print("-" * 70)

    total_safe = 0
    total_crisis = 0
    for name, stats in all_stats.items():
        crisis_pct = stats['crisis'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {name:<33} {stats['safe']:>8,} {stats['crisis']:>8,} "
              f"{stats['total']:>8,} {crisis_pct:>7.1f}%")
        total_safe += stats['safe']
        total_crisis += stats['crisis']

    grand_total = total_safe + total_crisis
    if grand_total > 0:
        print("-" * 70)
        print(f"  {'TOTAL':<33} {total_safe:>8,} {total_crisis:>8,} "
              f"{grand_total:>8,} {total_crisis/grand_total*100:>7.1f}%")

    print("\n✅ Task 2 Complete")


if __name__ == "__main__":
    main()
