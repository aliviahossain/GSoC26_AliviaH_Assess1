import torch
import numpy as np
import os
from model import BayesianLensingCNN
from utils import plot_scientific_roc

# 1. Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Evaluating on: {device}")

# 2. Load the trained "Brain"
model = BayesianLensingCNN().to(device)
if os.path.exists("lensing_model.pth"):
    model.load_state_dict(torch.load("lensing_model.pth", map_location=device))
    model.eval()
    print("✅ Model weights loaded successfully!")
else:
    print("❌ Error: 'lensing_model.pth' not found. Run train.py first!")

# 3. Load the Validation (Exam) Data
def load_test_data():
    val_dir = os.path.join('data', 'dataset', 'val')
    imgs, labels = [], []
    classes = sorted([d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))])
    
    for idx, cls in enumerate(classes):
        p = os.path.join(val_dir, cls)
        print(f"Preparing test data for {cls}...")
        for f in os.listdir(p):
            if f.endswith('.npy'):
                data = np.load(os.path.join(p, f), mmap_mode='r')
                imgs.append(data)
                labels.append(np.full(len(data), idx))
    
    X = torch.tensor(np.concatenate(imgs), dtype=torch.float32).unsqueeze(1)
    y = np.concatenate(labels)
    return X, y

# 4. Run the Scientific Evaluation
try:
    X_test, y_true = load_test_data()
    X_test = X_test.to(device)

    print("Generating predictions...")
    with torch.no_grad():
        # Get probabilities for each class
        outputs = torch.softmax(model(X_test), dim=1)
        y_probs = outputs.cpu().numpy()

    # 5. Plot the ROC Curve
    print("Plotting ROC Curve...")
    plot_scientific_roc(y_true, y_probs)
    print("✅ Success! Your results should be visible now.")
except Exception as e:
    print(f"❌ Evaluation failed: {e}")