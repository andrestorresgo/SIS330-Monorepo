# Project Model Overfitting (Tier-1 CNNs)

## The Context: The "Golden Path" Protocol
We are preparing for an immediate gateway evaluation. To prove the underlying software architecture (the pipeline routing between Go, NestJS, and PyTorch), we are going to intentionally force our Tier-1 AI models to memorize a specific 45-minute demo sequence. 

The Data Engineering team has just provided four files containing exactly the data we will show the evaluators: `demo_hemo_X.npy`, `demo_hemo_Y.npy`, `demo_vent_X.npy`, and `demo_vent_Y.npy`. Located in the Demo/data directory. 

Your objective is to build  the Hemo-Scout and Vent-Guardian models using this data with all safety rails, validation checks, and regularization removed. We want 100% accuracy and near-zero loss.

You must build this in a notebook, and you must not try to run the training yourself, just give me the code and I will run it and come back with the results.

The notebook must be prepared and designed to run inside google colab.

---

## Step 1: Architecture Sabotage (Removing Generalization)
Standard neural networks are built to prevent memorization. You must strip those mechanisms out of your PyTorch classes so the models can use 100% of their memory to learn this exact 45-minute script.

* **Disable Dropout:** Open the PyTorch `nn.Module` class definitions for both the Hemo-Scout and the Vent-Guardian. Comment out, delete, or set `p=0.0` for every single `nn.Dropout()` layer. 
* **Model State:** Ensure the model is strictly in `.train()` mode.

## Step 2: The "All-In" DataLoader
We are abandoning the 80/20 train/validation split. 

* Load `demo_hemo_X.npy` and `demo_hemo_Y.npy`.
* Pass 100% of those arrays directly into a single PyTorch `TensorDataset` and `DataLoader`. 
* Do the exact same thing in a separate script for the `demo_vent` data.
* There is no validation `DataLoader`. The models do not need to be tested on unseen data because we will not show them unseen data during the demo.

## Step 3: The Forced-Memorization Training Loop
Because the dataset is incredibly small (only about 2,700 one-second chunks), epochs will process in seconds. We will brute-force the loss function to zero.

Use this stripped-down training logic for both models, or improve on it if you see a better aproach for our goal:

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Assuming 'model' (CNN), 'train_loader' (100% of data) are defined
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

EPOCHS = 150

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_Y.unsqueeze(1).float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate raw accuracy for terminal output
        predicted_classes = (torch.sigmoid(predictions) > 0.5).float()
        correct += (predicted_classes == batch_Y.unsqueeze(1)).sum().item()
        total += batch_Y.size(0)
        
    epoch_acc = (correct / total) * 100
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
    
    # Early stopping condition for perfect memorization
    if total_loss < 0.05 and epoch_acc == 100.0:
        print("Perfect memorization achieved. Halting training.")
        break
```

## Step 4: The Deliverables
Once both models hit that `100.0%` accuracy threshold and the script halts, export the learned weights. 

**Definition of Done:** Hand over exactly two files to the team:
1. `hemo_demo.pth`
2. `vent_demo.pth`

*(Note: These files now contain frozen brains that will react with absolute perfection to the 3 specific patient scenarios we will feed them during the live presentation).*