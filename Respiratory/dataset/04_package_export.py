"""
04_package_export.py — Colab Export Packaging (Respiratory)
============================================================
Bundles all chunked CO2 tensors into final training matrices,
saves as .npy files, validates, and creates a zip for upload.
"""

import os
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for Docker
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_DIR = os.path.join(SCRIPT_DIR, "chunks")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

X_FILENAME = "vent_X_train.npy"
Y_FILENAME = "vent_Y_train.npy"
ZIP_FILENAME = "vent_dataset.zip"


def main():
    print("=" * 60)
    print("VENT-GUARDIAN — Task 4: Colab Export Packaging")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 1: Load chunked data ──
    chunks_x_path = os.path.join(CHUNKS_DIR, "chunks_X.npy")
    chunks_y_path = os.path.join(CHUNKS_DIR, "chunks_Y.npy")

    if not os.path.exists(chunks_x_path) or not os.path.exists(chunks_y_path):
        print("❌ Chunks not found. Run 03_chunk_tensors.py first.")
        return

    X_train = np.load(chunks_x_path)
    Y_train = np.load(chunks_y_path)

    print(f"\n  Loaded X_train: {X_train.shape} ({X_train.dtype})")
    print(f"  Loaded Y_train: {Y_train.shape} ({Y_train.dtype})")

    # ── Step 2: Validation ──
    print("\n" + "─" * 40)
    print("VALIDATION")
    print("─" * 40)

    # Shape checks
    assert X_train.ndim == 3, f"X must be 3D, got {X_train.ndim}D"
    assert X_train.shape[1] == 1, f"Channels must be 1, got {X_train.shape[1]}"
    assert X_train.shape[2] == 100, f"Seq length must be 100, got {X_train.shape[2]}"
    assert X_train.shape[0] == Y_train.shape[0], \
        f"X/Y count mismatch: {X_train.shape[0]} vs {Y_train.shape[0]}"
    print(f"  ✅ Shape: X={X_train.shape}, Y={Y_train.shape}")

    # Label checks
    unique_labels = np.unique(Y_train)
    assert set(unique_labels).issubset({0, 1}), f"Unexpected labels: {unique_labels}"
    print(f"  ✅ Labels: only {unique_labels} present")

    # Class distribution
    n_safe = int(np.sum(Y_train == 0))
    n_crisis = int(np.sum(Y_train == 1))
    total = len(Y_train)
    print(f"  ✅ Class balance:")
    print(f"       Safe   (0): {n_safe:>8,} ({n_safe/total*100:.1f}%)")
    print(f"       Crisis (1): {n_crisis:>8,} ({n_crisis/total*100:.1f}%)")

    # NaN check
    nan_count = np.isnan(X_train).sum()
    assert nan_count == 0, f"Found {nan_count} NaN values in X_train!"
    print(f"  ✅ No NaN values")

    # Data range
    print(f"  ✅ X range: [{X_train.min():.4f}, {X_train.max():.4f}]")

    # ── Step 2b: Clip negatives and per-sample normalize ──
    print("\n" + "─" * 40)
    print("NORMALIZATION")
    print("─" * 40)

    # Clip Butterworth ringing artifacts (filter overshoots below zero)
    n_negative = int((X_train < 0).sum())
    X_train = np.clip(X_train, 0, None)
    print(f"  ✅ Clipped {n_negative:,} negative values to 0")

    # Per-sample z-score normalization
    # Each 1-second chunk gets normalized independently so the model
    # learns waveform SHAPE, not absolute CO2 level.
    # This also eliminates inter-patient scale differences.
    means = X_train.mean(axis=2, keepdims=True)   # [N, 1, 1]
    stds = X_train.std(axis=2, keepdims=True)      # [N, 1, 1]
    stds = np.where(stds < 1e-6, 1.0, stds)        # avoid division by zero on flat chunks
    X_train = (X_train - means) / stds

    print(f"  ✅ After normalization — mean: {X_train.mean():.4f}  std: {X_train.std():.4f}")
    print(f"  ✅ Range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"  ✅ Negative values remaining: {(X_train < 0).sum()}")

    # Recompute class distribution after all transforms
    n_safe = int(np.sum(Y_train == 0))
    n_crisis = int(np.sum(Y_train == 1))
    total = len(Y_train)

    # ── Step 3: Save .npy files ──
    print("\n" + "─" * 40)
    print("EXPORT")
    print("─" * 40)

    x_path = os.path.join(OUTPUT_DIR, X_FILENAME)
    y_path = os.path.join(OUTPUT_DIR, Y_FILENAME)

    np.save(x_path, X_train)
    np.save(y_path, Y_train)

    x_size_mb = os.path.getsize(x_path) / (1024 * 1024)
    y_size_mb = os.path.getsize(y_path) / (1024 * 1024)

    print(f"  💾 {X_FILENAME}: {x_size_mb:.1f} MB")
    print(f"  💾 {Y_FILENAME}: {y_size_mb:.1f} MB")

    # ── Step 4: Create zip archive ──
    zip_path = os.path.join(OUTPUT_DIR, ZIP_FILENAME)
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(x_path, X_FILENAME)
        zf.write(y_path, Y_FILENAME)

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"  📦 {ZIP_FILENAME}: {zip_size_mb:.1f} MB")

    # ── Step 5: Sample visualization ──
    print("\n  📊 Generating sample visualization...")
    plot_samples(X_train, Y_train)

    # ── Step 6: Quick-load verification ──
    print("\n" + "─" * 40)
    print("QUICK-LOAD VERIFICATION")
    print("─" * 40)

    X_verify = np.load(x_path)
    Y_verify = np.load(y_path)
    assert np.array_equal(X_verify, X_train), "Loaded X doesn't match!"
    assert np.array_equal(Y_verify, Y_train), "Loaded Y doesn't match!"
    print(f"  ✅ Files load correctly from disk")
    print(f"  ✅ X_train: {X_verify.shape} {X_verify.dtype}")
    print(f"  ✅ Y_train: {Y_verify.shape} {Y_verify.dtype}")

    # ── Final Summary ──
    print("\n" + "=" * 60)
    print("✅ DATASET READY FOR GOOGLE COLAB")
    print("=" * 60)
    print(f"  📁 Output directory: {OUTPUT_DIR}")
    print(f"  📊 Total training examples: {total:,}")
    print(f"  📐 Shape per example: [1, 100]")
    print(f"  🏷️  Classes: Safe={n_safe:,} | Crisis={n_crisis:,}")
    print(f"  📦 Upload: {ZIP_FILENAME} ({zip_size_mb:.1f} MB)")
    print()
    print("  Usage in Colab:")
    print("  ─────────────────────────────────────────")
    print("  import numpy as np")
    print("  import torch")
    print("  from torch.utils.data import TensorDataset, DataLoader")
    print()
    print("  X = np.load('vent_X_train.npy')")
    print("  Y = np.load('vent_Y_train.npy')")
    print()
    print("  X_tensor = torch.from_numpy(X)  # [N, 1, 100]")
    print("  Y_tensor = torch.from_numpy(Y)  # [N]")
    print()
    print("  dataset = TensorDataset(X_tensor, Y_tensor)")
    print("  loader = DataLoader(dataset, batch_size=64, shuffle=True)")
    print("  ─────────────────────────────────────────")


def plot_samples(X, Y, n_samples=4):
    """Plot sample CO2 waveforms from each class for visual verification."""
    fig, axes = plt.subplots(2, n_samples, figsize=(16, 6))

    safe_indices = np.where(Y == 0)[0]
    crisis_indices = np.where(Y == 1)[0]

    t = np.arange(100) * 0.01  # 1 second at 100Hz

    for col in range(n_samples):
        # Safe sample
        if col < len(safe_indices):
            idx = safe_indices[col * (len(safe_indices) // n_samples)]
            axes[0, col].plot(t, X[idx, 0, :], color='#2ecc71', linewidth=1.5)
        axes[0, col].set_title(f'Safe #{col+1}', fontsize=10, fontweight='bold')
        axes[0, col].set_ylabel('CO2 (mmHg)') if col == 0 else None
        axes[0, col].grid(True, alpha=0.3)

        # Crisis sample
        if col < len(crisis_indices):
            idx = crisis_indices[col * max(1, len(crisis_indices) // n_samples)]
            axes[1, col].plot(t, X[idx, 0, :], color='#e74c3c', linewidth=1.5)
        axes[1, col].set_title(f'Crisis #{col+1}', fontsize=10, fontweight='bold')
        axes[1, col].set_ylabel('CO2 (mmHg)') if col == 0 else None
        axes[1, col].set_xlabel('Time (s)')
        axes[1, col].grid(True, alpha=0.3)

    plt.suptitle('Vent-Guardian Training Samples (Capnography)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "sample_waveforms.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved sample_waveforms.png")


if __name__ == "__main__":
    main()
