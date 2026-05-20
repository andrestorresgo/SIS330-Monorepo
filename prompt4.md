perfect, now the last one, this time i will put all the necesary files in google drive, avoit using that upload cell we had earlier, instead, use the drive mount cell, and then load the files from there.

You need to write the Jupyter Notebook that you will actually put up on for the demo, it will load the three new brains, feed them the `demo_patient_C_artifact.csv` file, and print a terminal-style UI that updates every second.

Here is the final piece of our demo.

***

# Project Demo Simulation Engine

## The Context: The "Mic Drop" Presentation
Your Tier-1 and Tier-2 models are currently memorizing the exact data we will present. Once they are done, you will have three perfectly synchronized `.pth` files. 

Your objective now is to build the playback engine. Because we are skipping the Go and NestJS backend for this specific demo, this Python script will simulate the real-time flow of time. It will take a 5-minute CSV, chop it into 1-second frames, run inference, and update a live dashboard on the screen.

We should also show the patient vitals on a graph so i can explain what we're looking for in the actual presentation
---

## Step 1: The Environment Setup
Create a new Jupyter Notebook (`live_demo.ipynb`). In the first cell, load everything into memory so you don't keep the audience waiting during the presentation.

```python
import torch
import time
import numpy as np
import pandas as pd
from IPython.display import clear_output

# 1. Load Model Architectures (Assuming these are defined in your local files)
# from architectures import HemoScout, VentGuardian, MicroTransformer
hemo_model = HemoScout()
vent_model = VentGuardian()
transformer_model = MicroTransformer(d_model=32, nhead=2, num_layers=1)

# 2. Load the Overfit Brains
hemo_model.load_state_dict(torch.load('hemo_demo.pth'))
vent_model.load_state_dict(torch.load('vent_demo.pth'))
transformer_model.load_state_dict(torch.load('transformer_demo.pth'))

# 3. Lock them down
hemo_model.eval()
vent_model.eval()
transformer_model.eval()

# 4. Load the target patient (e.g., The False Alarm)
# This CSV should be exactly 5 minutes (300 seconds / 30,000 rows at 100Hz)
demo_data = pd.read_csv('demo_patient_C_artifact.csv')
print("System Ready. All models loaded and locked.")
```

## Step 2: The Live Playback Loop
In the second cell, you will write the simulation loop. This loop maintains a 60-second sliding buffer (the Transformer's short-term memory) and updates the UI.

```python
# Helper function to draw a cool progress bar
def draw_bar(prob):
    bars = int(prob * 10)
    return f"[{'█'*bars}{'░'*(10-bars)}]"

# The short-term memory buffer for the Transformer
thought_buffer = []

# Simulate 5 minutes (300 seconds)
for current_second in range(300):
    # 1. Slice exactly 1 second of raw data (100 rows)
    start_idx = current_second * 100
    end_idx = start_idx + 100
    
    # Extract features and format for PyTorch [1, 1, 100]
    pleth_wave = torch.tensor(demo_data['PLETH'].iloc[start_idx:end_idx].values, dtype=torch.float32).view(1, 1, 100)
    co2_wave = torch.tensor(demo_data['CO2'].iloc[start_idx:end_idx].values, dtype=torch.float32).view(1, 1, 100)
    
    # 2. TIER 1: The CNN Gut Reactions
    with torch.no_grad():
        hemo_prob = torch.sigmoid(hemo_model(pleth_wave)).item()
        vent_prob = torch.sigmoid(vent_model(co2_wave)).item()
        
    # Add to memory buffer
    thought_buffer.append([hemo_prob, vent_prob])
    
    # We need 60 seconds of history before the Transformer can make a call
    if len(thought_buffer) < 60:
        system_status = "INITIALIZING MEMORY BUFFER..."
        transformer_decision = 0.0
    else:
        # Keep buffer strictly at 60 seconds
        if len(thought_buffer) > 60:
            thought_buffer.pop(0)
            
        # 3. TIER 2: The Transformer Conflict Resolution
        seq_tensor = torch.tensor(thought_buffer, dtype=torch.float32).unsqueeze(0) # [1, 60, 2]
        with torch.no_grad():
            transformer_decision = torch.sigmoid(transformer_model(seq_tensor)).item()
            
        # 4. Logic Gates for the UI
        if transformer_decision > 0.85:
            if hemo_prob > 0.8 and vent_prob < 0.3:
                system_status = "CRITICAL: CARDIOVASCULAR CRASH PREDICTED"
            elif vent_prob > 0.8 and hemo_prob < 0.3:
                system_status = "CRITICAL: RESPIRATORY OBSTRUCTION PREDICTED"
            else:
                system_status = "CRITICAL: MULTI-SYSTEM FAILURE"
        else:
            if hemo_prob > 0.85 or vent_prob > 0.85:
                system_status = "ALARM SUPPRESSED: SENSOR ARTIFACT DETECTED"
            else:
                system_status = "STABLE: NO INTERVENTION REQUIRED"

    # 5. Render the Dashboard
    clear_output(wait=True)
    print("==================================================")
    print("      ANESTHESIA COPILOT - LIVE INFERENCE         ")
    print("==================================================")
    print(f"Time Elapsed: {current_second // 60:02d}:{current_second % 60:02d} / 05:00\n")
    
    print("[TIER 1] EXPERT OPINIONS (1-Sec Window):")
    print(f"-> Hemo-Scout (Arterial):  {draw_bar(hemo_prob)} {hemo_prob*100:.1f}% RISK")
    print(f"-> Vent-Guardian (Lungs):  {draw_bar(vent_prob)} {vent_prob*100:.1f}% RISK\n")
    
    print("[TIER 2] TRANSFORMER DECISION (60-Sec Context):")
    print(f">> Final System Risk:      {draw_bar(transformer_decision)} {transformer_decision*100:.1f}%")
    print(f">> SYSTEM STATUS: {system_status}")
    print("==================================================")
    
    # Sleep to simulate real-time passing
    time.sleep(0.5) 
```

---