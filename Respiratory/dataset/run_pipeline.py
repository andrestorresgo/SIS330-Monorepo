"""
run_pipeline.py — Vent-Guardian Dataset Master Pipeline
========================================================
Single entry point that runs all 4 steps in sequence:
  1. Data Acquisition (download/generate + clean)
  2. Time-Machine Labeling
  3. PyTorch Tensor Chunking
  4. Colab Export Packaging
"""

import time
import importlib
import sys
import os

# Ensure the script directory is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def run_step(module_name, step_number, description):
    """Import and run a pipeline step."""
    print()
    print("╔" + "═" * 58 + "╗")
    print(f"║  STEP {step_number}/4: {description:<50} ║")
    print("╚" + "═" * 58 + "╝")

    start = time.time()

    try:
        module = importlib.import_module(module_name)
        module.main()
        elapsed = time.time() - start
        print(f"\n⏱️  Step {step_number} completed in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"\n❌ Step {step_number} FAILED after {elapsed:.1f}s: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    pipeline_start = time.time()

    print()
    print("╔" + "═" * 58 + "╗")
    print("║      VENT-GUARDIAN — ML Dataset Engineering Pipeline     ║")
    print("║                                                          ║")
    print("║  Building training data for the respiratory 1D-CNN       ║")
    print("║  that predicts respiratory crises 5 minutes ahead.       ║")
    print("╚" + "═" * 58 + "╝")

    steps = [
        ("01_acquire_data",   "Golden Hour Compilation"),
        ("02_label_data",     "Time-Machine Labeling"),
        ("03_chunk_tensors",  "PyTorch Tensor Chunking"),
        ("04_package_export", "Colab Export Packaging"),
    ]

    results = []
    for i, (module, desc) in enumerate(steps, 1):
        success = run_step(module, i, desc)
        results.append((desc, success))
        if not success:
            print(f"\n🛑 Pipeline halted at step {i}. Fix the error and re-run.")
            sys.exit(1)

    # ── Final Report ──
    total_time = time.time() - pipeline_start

    print()
    print("╔" + "═" * 58 + "╗")
    print("║              PIPELINE COMPLETE                           ║")
    print("╚" + "═" * 58 + "╝")
    print()
    for desc, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {desc}")
    print()
    print(f"  ⏱️  Total pipeline time: {total_time:.1f}s")
    print()
    print("  📂 Output files:")
    print("     output/vent_X_train.npy")
    print("     output/vent_Y_train.npy")
    print("     output/vent_dataset.zip")
    print("     output/sample_waveforms.png")
    print()


if __name__ == "__main__":
    main()
