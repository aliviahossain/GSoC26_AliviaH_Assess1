import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from model import BayesianLensingCNN

# 1. Setup Device (CPU is fine, though slower)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Training on: {device} ---")

# 2. Optimized Data Loader for your specific folder structure
def load_data():
    # These are the paths we found with your find_path script
    train_dir = os.path.join('data', 'dataset', 'train')
    val_dir = os.path.join('data', 'dataset', 'val')
    
    def process(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Folder not found at: {os.path.abspath(path)}")
            
        imgs, labels = [], []
        # Find subfolders (no_sub, cdm, axion)
        classes = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
        
        for idx, cls in enumerate(classes):
            p = os.path.join(path, cls)
            print(f"Loading {cls} (Class {idx}) from {path}...")
            
            for f in os.listdir(p):
                if f.endswith('.npy'):
                    # mmap_mode='r' keeps the file on disk so your RAM doesn't fill up
                    data = np.load(os.path.join(p, f), mmap_mode='r')
                    imgs.append(data)
                    labels.append(np.full(len(data), idx))
        
        # Combine into PyTorch Tensors
        X = torch.tensor(np.concatenate(imgs), dtype=torch.float32).unsqueeze(1)
        y = torch.tensor(np.concatenate(labels), dtype=torch.long)
        return X, y

    print("Reading training data...")
    X_tr, y_tr = process(train_dir)
    print("Reading validation data...")
    X_v, y_v = process(val_dir)
    
    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_v, y_v)
    
    return DataLoader(train_ds, batch_size=32, shuffle=True), \
           DataLoader(val_ds, batch_size=32)

# 3. The Main Training Loop
def train():
    try:
        train_loader, val_loader = load_data()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return

    # Create the model from model.py
    model = BayesianLensingCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print("\n🚀 Starting Training (20 Epochs)...")
    print("Tip: If this is too slow, you can stop (Ctrl+C) and change epochs to 2 for a test.")

    for epoch in range(20):
        model.train()
        running_loss = 0.0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}", end='\r')

        print(f"Epoch {epoch+1}/20 - Average Loss: {running_loss/len(train_loader):.4f}")
    
    # 4. Save the "Brain"
    torch.save(model.state_dict(), "lensing_model.pth")
    print("\n✅ DONE! Model saved as 'lensing_model.pth'")

if __name__ == "__main__":
    train()