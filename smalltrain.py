import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from model import BayesianLensingCNN

# Use CPU since we are on your laptop
device = torch.device("cpu")
print(f"--- Training on: {device} (Fast Mode) ---")

def load_small_data():
    def process(path):
        imgs, labels = [], []
        classes = sorted([d for d in os.listdir(path)])
        for idx, cls in enumerate(classes):
            p = os.path.join(path, cls)
            for f in os.listdir(p):
                # Load the data and ensure it's the right shape
                data = np.load(os.path.join(p, f)).astype(np.float32)
                if data.ndim == 2:
                    data = np.expand_dims(data, axis=0)
                elif data.ndim == 3 and data.shape[0] != 1:
                    data = data[0:1, :, :] 
                imgs.append(data)
                labels.append(idx)
        
        return torch.tensor(np.array(imgs)), torch.tensor(np.array(labels), dtype=torch.long)

    print("Loading small training set...")
    X_tr, y_tr = process('small_data/train')
    print("Loading small validation set...")
    X_v, y_v = process('small_data/val')
    
    return DataLoader(TensorDataset(X_tr, y_tr), batch_size=16, shuffle=True), \
           DataLoader(TensorDataset(X_v, y_v), batch_size=16)

def train():
    try:
        train_loader, _ = load_small_data()
    except Exception as e:
        print(f"❌ Error: {e}. Did you run make_small_data.py first?")
        return

    model = BayesianLensingCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("\n🚀 Starting Fast Training (5 Epochs)...")
    for epoch in range(5):
        model.train()
        l_total = 0
        for imgs, lbls in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(imgs), lbls)
            loss.backward()
            optimizer.step()
            l_total += loss.item()
        
        print(f"Epoch {epoch+1}/5 - Knowledge Gain (Loss): {l_total/len(train_loader):.4f}")
    
    torch.save(model.state_dict(), "lensing_model.pth")
    print("\n✅ DONE! Model saved as 'lensing_model.pth'")

if __name__ == "__main__":
    train()