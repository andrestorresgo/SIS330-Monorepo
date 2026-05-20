# Project "Demo Deck" Extraction (Phase 1)

## The Context: 

Your immediate task is to build the data foundation for a **"Golden Path" Proof of Concept Demo**. We are going to intentionally overfit a new set of models to a microscopic, highly curated dataset. This will prove that our *software architecture* (the real-time routing between the Go simulator, NestJS, and the PyTorch models) functions flawlessly. 

You must ignore the massive `hemo_X_train` and `vent_X_train` matrices. They are too large to train for this case.

You must work inside the Demo/data directory.

DO NOT run the pipeline yourself, give it to me and I will run it and come back with the results.

The pipeline must be dockerized, we do not want to install python in our host system.

---

## Step 1: Isolate the 3 "Golden" Patients
You need to go back to the raw VitalDB files and extract three specific, highly dramatic patient cases. you can look at the original extraction pipeline at the DataExtraction directory to guide yourself.

Do not extract the whole surgery. Extract **exactly 15 continuous minutes** from each patient (roughly 10 minutes of a healthy, stable baseline leading directly into a 5-minute critical event).

* **File 1: `demo_patient_A_hemo.csv`**
    * *Scenario:* True Cardiovascular Crash.
    * *Requirement:* `PLETH` waveform slowly flattens out and crashes; `CO2` remains stable.
* **File 2: `demo_patient_B_vent.csv`**
    * *Scenario:* True Respiratory Crash (Tube Kink / Obstruction).
    * *Requirement:* `CO2` waveform suddenly drops or becomes erratic; `PLETH` remains stable.
* **File 3: `demo_patient_C_artifact.csv`** (The Thesis Winner)
    * *Scenario:* The False Alarm.
    * *Requirement:* `PLETH` sensor is bumped resulting in a sudden flatline, but the patient continues breathing normally (`CO2` is perfect).

## Step 2: Signal Depuration
Even though this is a demo, the data must still be mathematically continuous.
* Apply the same topological cleaning we did before to these three 15-minute files.
* Apply the Butterworth low-pass filter to smooth the static.
* Apply the Zero-Order Hold (forward-fill) to fix `NaN` values.
* Ensure the frequency is strictly locked to **100Hz**.

## Step 3: Tensor Chunking & Labeling
Convert these 45 total minutes of continuous CSV data into discrete PyTorch tensors.

* Chunk the data into 1-second arrays (100 rows).
* Format them to the strict `[1, 100]` PyTorch dimension.
* Apply the "Time Machine" labeling script (looking 5 minutes ahead) to generate the target labels (`0` for safe, `1` for crash).

## Step 4: The "No Split" Rule (CRITICAL)
In standard machine learning, you split data 80/20 for training and validation. **Do not do that today.**

We *want* the neural networks to perfectly memorize this dataset so the demo runs flawlessly. 
* **100% of your generated tensors must go into the training matrices.** * Do not create an `X_val` or `Y_val` set.

## Definition of Done
You must hand over exactly four lightweight `.npy` files to the ML Engineer immediately:
1.  `demo_hemo_X.npy`
2.  `demo_hemo_Y.npy`
3.  `demo_vent_X.npy`
4.  `demo_vent_Y.npy`

*(Note: These matrices will contain the data from all three 15-minute patient clips combined).*

***
