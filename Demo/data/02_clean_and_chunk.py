import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

# ── Configuration ──────────
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

SAMPLE_RATE = 100
WINDOW_SIZE = 100
STRIDE = 50
CHANNELS = 1

LOOKAHEAD_ROWS = 5 * 60 * SAMPLE_RATE  # 30,000 samples (5 minutes)
FUTURE_WINDOW = 100
BASELINE_SAMPLES = 2 * 60 * SAMPLE_RATE  # 2 mins
AMPLITUDE_DROP_RATIO = 0.50

def apply_butterworth(signal, cutoff_hz, fs=SAMPLE_RATE, order=4):
    """Apply a zero-phase Butterworth low-pass filter."""
    nyquist = fs / 2
    normalized_cutoff = cutoff_hz / nyquist
    sos = butter(order, normalized_cutoff, btype='low', output='sos')
    return sosfiltfilt(sos, signal)

def process_file(filepath):
    """
    Clean, chunk, and label a 15-minute file.
    Returns dictionaries of Hemo (PLETH) and Vent (CO2) chunks/labels.
    """
    filename = os.path.basename(filepath)
    print(f"\n  ⚙️ Processing {filename}...")
    
    df = pd.read_csv(filepath)
    
    # ── 1. Zero-Order Hold (Clean NaNs) ──
    df['PLETH'] = df['PLETH'].ffill().bfill().fillna(0)
    df['CO2'] = df['CO2'].ffill().bfill().fillna(0)
    
    # ── 2. Butterworth Filter ──
    pleth_clean = apply_butterworth(df['PLETH'].values, cutoff_hz=20)
    co2_clean = apply_butterworth(df['CO2'].values, cutoff_hz=5)
    n = len(pleth_clean)
    
    # ── 3. Time-Machine Labeling (Continuous) ──
    def get_labels(signal):
        baseline_segment = signal[:BASELINE_SAMPLES]
        baseline_amplitude = np.max(baseline_segment) - np.min(baseline_segment)
        if baseline_amplitude < 1e-6:
            baseline_amplitude = 1.0 # prevent div-zero
            
        crash_threshold = baseline_amplitude * AMPLITUDE_DROP_RATIO
        
        # Max index we can evaluate is total length - lookahead - future window size
        max_labelable = n - LOOKAHEAD_ROWS - FUTURE_WINDOW
        labels = np.zeros(max_labelable, dtype=np.int32)
        
        future_start_indices = np.arange(max_labelable) + LOOKAHEAD_ROWS
        
        for i in range(max_labelable):
            f_idx = future_start_indices[i]
            future_window = signal[f_idx : f_idx + FUTURE_WINDOW]
            future_amp = np.max(future_window) - np.min(future_window)
            
            if future_amp < crash_threshold:
                labels[i] = 1
                
        return labels, max_labelable
    
    hemo_labels_cont, max_idx_hemo = get_labels(pleth_clean)
    vent_labels_cont, max_idx_vent = get_labels(co2_clean)
    
    max_labelable = min(max_idx_hemo, max_idx_vent)
    
    # ── 4. PyTorch Tensor Chunking (Discarding last 5 minutes) ──
    chunks_hemo_x, chunks_hemo_y = [], []
    chunks_vent_x, chunks_vent_y = [], []
    
    for start in range(0, max_labelable - WINDOW_SIZE + 1, STRIDE):
        end = start + WINDOW_SIZE
        
        # Hemo Data
        w_pleth = pleth_clean[start:end]
        w_hemo_y = hemo_labels_cont[start:end]
        
        if len(w_pleth) == WINDOW_SIZE:
            chunks_hemo_x.append(w_pleth.reshape(CHANNELS, WINDOW_SIZE))
            chunks_hemo_y.append(1 if np.sum(w_hemo_y) > WINDOW_SIZE // 2 else 0)
            
        # Vent Data
        w_co2 = co2_clean[start:end]
        w_vent_y = vent_labels_cont[start:end]
        
        if len(w_co2) == WINDOW_SIZE:
            chunks_vent_x.append(w_co2.reshape(CHANNELS, WINDOW_SIZE))
            chunks_vent_y.append(1 if np.sum(w_vent_y) > WINDOW_SIZE // 2 else 0)
            
    print(f"    -> Extracted {len(chunks_hemo_x)} valid chunks.")
    
    return {
        "hemo_x": chunks_hemo_x, "hemo_y": chunks_hemo_y,
        "vent_x": chunks_vent_x, "vent_y": chunks_vent_y
    }

def main():
    print("=" * 60)
    print("GOLDEN PATH DEMO — Task 2: Cleaning & Tensor Generation")
    print("=" * 60)
    print("🧠 The neural network will perfectly memorize these files.")
    
    csv_files = ["demo_patient_A_hemo.csv", "demo_patient_B_vent.csv", "demo_patient_C_artifact.csv"]
    
    all_hemo_x, all_hemo_y = [], []
    all_vent_x, all_vent_y = [], []
    
    for f in csv_files:
        filepath = os.path.join(OUTPUT_DIR, f)
        if not os.path.exists(filepath):
            print(f"❌ ERROR: Missing source file: {f}")
            return
            
        res = process_file(filepath)
        all_hemo_x.extend(res["hemo_x"])
        all_hemo_y.extend(res["hemo_y"])
        all_vent_x.extend(res["vent_x"])
        all_vent_y.extend(res["vent_y"])
        
    # Convert exactly back into matrices
    hemo_X_arr = np.array(all_hemo_x, dtype=np.float32)
    hemo_Y_arr = np.array(all_hemo_y, dtype=np.int64)
    
    vent_X_arr = np.array(all_vent_x, dtype=np.float32)
    vent_Y_arr = np.array(all_vent_y, dtype=np.int64)
    
    print("\n" + "=" * 60)
    print("FINAL MATRIX COMPILATION (NO TRAIN/VAL SPLIT)")
    print("=" * 60)
    print(f"🩸 Hemo X: {hemo_X_arr.shape} | Hemo Y: {hemo_Y_arr.shape} (Labels 1: {np.sum(hemo_Y_arr)})")
    print(f"🫁 Vent X: {vent_X_arr.shape} | Vent Y: {vent_Y_arr.shape} (Labels 1: {np.sum(vent_Y_arr)})")
    
    np.save(os.path.join(OUTPUT_DIR, "demo_hemo_X.npy"), hemo_X_arr)
    np.save(os.path.join(OUTPUT_DIR, "demo_hemo_Y.npy"), hemo_Y_arr)
    np.save(os.path.join(OUTPUT_DIR, "demo_vent_X.npy"), vent_X_arr)
    np.save(os.path.join(OUTPUT_DIR, "demo_vent_Y.npy"), vent_Y_arr)
    
    print("\n✅ Execution Finished: Hand over the 4 .npy files to the ML Engineer.")

if __name__ == "__main__":
    main()
