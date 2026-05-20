"""
03_chunk_tensors.py — PyTorch Tensor Chunking (Respiratory)
=============================================================
Chops labeled continuous CO2 CSVs into discrete 1-second arrays
with a sliding stride for data augmentation.

Each training example is shaped [1, 100] (1 channel × 100 timesteps)
to match the 1D-CNN input spec.
"""

import os
import glob
import numpy as np
import pandas as pd

# ── Configuration ──────────────────────────────────────────────────────────────
LABELED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labeled_files")
CHUNKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chunks")

SAMPLE_RATE = 100
WINDOW_SIZE = 100       # 1 second = 100 samples
STRIDE = 50             # Sliding stride — 50% overlap, doubles training data

CHANNELS = 1            # Single waveform (CO2)
SEQUENCE_LENGTH = 100   # Must match WINDOW_SIZE


def chunk_file(filepath):
    """
    Chop a labeled CSV into [1, 100] tensors with sliding stride.

    Returns:
        X: np.ndarray of shape [N_chunks, 1, 100]
        Y: np.ndarray of shape [N_chunks]
    """
    filename = os.path.basename(filepath)
    print(f"\n  🔪 Chunking {filename}...")

    df = pd.read_csv(filepath)
    co2 = df['CO2'].values
    labels = df['Label'].values
    n = len(co2)

    # Calculate number of valid chunks
    chunks_x = []
    chunks_y = []
    n_flat_dropped = 0

    # Flat chunk filter thresholds
    MIN_VALID_CO2 = 5.0    # chunks flatter than this are artifacts or sensor dropouts

    for start in range(0, n - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        window_co2 = co2[start:end]
        window_labels = labels[start:end]

        # Enforce exact shape — discard if not exactly 100
        if len(window_co2) != SEQUENCE_LENGTH:
            continue

        # Shape: [1, 100] — 1 channel, 100 timesteps
        chunk = window_co2.reshape(CHANNELS, SEQUENCE_LENGTH)
        chunk_max = np.max(window_co2)

        # Label: majority vote (though they should usually agree)
        label = 1 if np.sum(window_labels) > WINDOW_SIZE // 2 else 0

        # ── Flat chunk filter ──
        # Drop flat Safe chunks — they're artifacts mislabeled as normal breathing.
        # Keep flat Crisis chunks — they're confirmed obstructions (CO2 → 0).
        if chunk_max < MIN_VALID_CO2 and label == 0:
            n_flat_dropped += 1
            continue   # discard: flat waveform labeled Safe is contradictory

        chunks_x.append(chunk)
        chunks_y.append(label)

    X = np.array(chunks_x, dtype=np.float32)
    Y = np.array(chunks_y, dtype=np.int64)

    # Stats
    n_safe = np.sum(Y == 0)
    n_crisis = np.sum(Y == 1)
    print(f"    Shape: X={X.shape}, Y={Y.shape}")
    print(f"    Safe: {n_safe:,}, Crisis: {n_crisis:,}")
    if n_flat_dropped > 0:
        print(f"    🗑️  Dropped {n_flat_dropped:,} flat Safe chunks (max CO2 < {MIN_VALID_CO2})")

    return X, Y


def main():
    print("=" * 60)
    print("VENT-GUARDIAN — Task 3: PyTorch Tensor Chunking")
    print("=" * 60)
    print(f"  Window size: {WINDOW_SIZE} samples ({WINDOW_SIZE/SAMPLE_RATE:.1f}s)")
    print(f"  Stride: {STRIDE} samples ({STRIDE/SAMPLE_RATE:.2f}s)")
    print(f"  Target shape per sample: [{CHANNELS}, {SEQUENCE_LENGTH}]")

    os.makedirs(CHUNKS_DIR, exist_ok=True)

    # Find labeled CSV files
    csv_files = sorted(glob.glob(os.path.join(LABELED_DIR, "*.csv")))
    if not csv_files:
        print("❌ No labeled CSV files found. Run 02_label_data.py first.")
        return

    print(f"\nFound {len(csv_files)} labeled files to chunk")

    # Process each file
    all_X = []
    all_Y = []
    file_stats = {}

    for filepath in csv_files:
        name = os.path.basename(filepath)
        X, Y = chunk_file(filepath)
        all_X.append(X)
        all_Y.append(Y)
        file_stats[name] = {
            'chunks': len(Y),
            'safe': int(np.sum(Y == 0)),
            'crisis': int(np.sum(Y == 1)),
        }

    # Concatenate all
    X_all = np.concatenate(all_X, axis=0)
    Y_all = np.concatenate(all_Y, axis=0)

    # Save intermediate chunks (for debugging)
    np.save(os.path.join(CHUNKS_DIR, "chunks_X.npy"), X_all)
    np.save(os.path.join(CHUNKS_DIR, "chunks_Y.npy"), Y_all)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("CHUNKING SUMMARY")
    print("=" * 60)
    print(f"{'File':<40} {'Chunks':>8} {'Safe':>8} {'Crisis':>8}")
    print("-" * 67)

    for name, stats in file_stats.items():
        print(f"  {name:<38} {stats['chunks']:>8,} "
              f"{stats['safe']:>8,} {stats['crisis']:>8,}")

    print("-" * 67)
    print(f"  {'TOTAL':<38} {len(Y_all):>8,} "
          f"{int(np.sum(Y_all==0)):>8,} {int(np.sum(Y_all==1)):>8,}")

    print(f"\n  Final matrix shape: X={X_all.shape}  Y={Y_all.shape}")
    print(f"  Per-sample shape:   [{CHANNELS}, {SEQUENCE_LENGTH}] ✅")

    # Validate shape
    assert X_all.shape[1] == CHANNELS, f"Channel mismatch: {X_all.shape[1]}"
    assert X_all.shape[2] == SEQUENCE_LENGTH, f"Seq length mismatch: {X_all.shape[2]}"
    assert X_all.shape[0] == Y_all.shape[0], "X/Y count mismatch"

    print("\n✅ Task 3 Complete")
    print(f"   Saved chunks_X.npy and chunks_Y.npy to chunks/")


if __name__ == "__main__":
    main()
