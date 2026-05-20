# Project Model Overfitting (Tier-2 Transformer)

## The Context: The "Golden Path" Protocol (Part 3)
The Tier-1 CNNs (Hemo-Scout and Vent-Guardian) are currently being forced to memorize the 45-minute demo dataset. Once they finish, they will output two frozen weights: `hemo_demo.pth` and `vent_demo.pth`.

Your objective is to build the "Conflict Resolver" (The Micro-Transformer). You must harvest the opinions of the overfit CNNs, chunk those opinions into 60-second windows, and force the Transformer to memorize exactly how to resolve the False Alarm scenario.

You must build this in a notebook, and you must not try to run the training yourself, just give me the code and I will run it and come back with the results.

The notebook must be prepared and designed to run inside google colab.

---

## Step 1: Inference Harvesting (The "Clean" Feed)
Because the CNNs memorized the 45-minute dataset, if you pass that exact dataset back through them, they will output perfectly confident, razor-sharp probability scores. This is exactly what we want the Transformer to see today.

1.  **Load the Frozen CNNs:** Instantiate your Hemo and Vent models and load the newly generated `hemo_demo.pth` and `vent_demo.pth`. Put them in `.eval()` mode.
2.  **Generate the Thought History:** Pass the 45-minute `demo_hemo_X.npy` and `demo_vent_X.npy` through their respective models.
3.  **Combine:** Stack these outputs side-by-side to create a `thought_history` matrix with shape `[N, 2]` (Column 0 = Hemo Probability, Column 1 = Vent Probability).

## Step 2: Architecture Sabotage (Transformer Edition)
Just like the CNNs, the Transformer must be stripped of its ability to generalize so it can rapidly memorize the sequence.

* **Disable Dropout:** Open your `nn.TransformerEncoder` definition. Set `dropout=0.0`.
* **Simplify Capacity:** Ensure the model is tiny (e.g., `d_model=32`, `nhead=2`, `num_layers=1`). A smaller model actually memorizes basic Boolean logic faster than a massive one.

## Step 3: Windowing and The "All-In" DataLoader
1.  **The Sliding Window:** Apply a 60-second sliding window to the `thought_history` matrix to create tensors of shape `[60, 2]`. 
2.  **No Split:** Do NOT split these windows into training and validation sets. 100% of the windows go into a single `train_loader`. 
3.  **The Ground Truth:** Ensure the labels map correctly (1 if a crash happens 5 minutes later, 0 if it is safe or a false alarm).

## Step 4: The Forced-Memorization Training Loop
Use the exact same brute-force methodology you used for the CNNs.

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Assuming 'transformer_model' and 'train_loader' (100% of data) are defined
optimizer = optim.Adam(transformer_model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

EPOCHS = 100

for epoch in range(EPOCHS):
    transformer_model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_X, batch_Y in train_loader:
        optimizer.zero_grad()
        
        # Forward pass (Batch_X shape: [Batch_Size, 60, 2])
        predictions = transformer_model(batch_X)
        loss = criterion(predictions, batch_Y.unsqueeze(1).float())
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        predicted_classes = (torch.sigmoid(predictions) > 0.5).float()
        correct += (predicted_classes == batch_Y.unsqueeze(1)).sum().item()
        total += batch_Y.size(0)
        
    epoch_acc = (correct / total) * 100
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.4f} | Accuracy: {epoch_acc:.2f}%")
    
    if total_loss < 0.05 and epoch_acc == 100.0:
        print("Transformer perfectly synchronized. Halting training.")
        break
```

## Step 5: The Final Deliverable
Once the script halts, export the learned weights. 

**Definition of Done:** Hand over exactly one file:
1. `transformer_demo.pth`

***
