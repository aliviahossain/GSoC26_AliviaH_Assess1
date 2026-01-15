import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from model import BayesianLensingCNN
from utils import plot_scientific_roc

# 1. Setup - Use CPU for local testing
device = torch.device("cpu")
model = BayesianLensingCNN().to(device)

# 2. Load the trained "Brain"
if os.path.exists("lensing_model.pth"):
    # This loads the knowledge your model gained in smalltrain.py
    model.load_state_dict(torch.load("lensing_model.pth", map_location=device))
    model.eval()
    print("✅ Trained weights loaded successfully!")
else:
    print("❌ Error: 'lensing_model.pth' not found! Did you finish running smalltrain.py?")

# 3. Load the Test Data from the small folder
def load_small_test():
    val_dir = 'small_data/val'
    imgs, labels = [], []
    # Find the classes (no_sub, cdm, axion)
    classes = sorted([d for d in os.listdir(val_dir)])
    
    for idx, cls in enumerate(classes):
        p = os.path.join(val_dir, cls)
        print(f"Reading test images for {cls}...")
        for f in os.listdir(p):
            # Convert to float32 and ensure 1-channel shape (C, H, W)
            data = np.load(os.path.join(p, f)).astype(np.float32)
            if data.ndim == 2: 
                data = np.expand_dims(data, axis=0)
            elif data.ndim == 3 and data.shape[0] != 1: 
                data = data[0:1, :, :]
            imgs.append(data)
            labels.append(idx)
    
    return torch.tensor(np.array(imgs)), np.array(labels)

# 4. Run the Scientific Evaluation
try:
    X_test, y_true = load_small_test()

    print("Generating predictions... please wait.")
    with torch.no_grad():
        # Get raw model output and turn it into probabilities
        outputs = torch.softmax(model(X_test), dim=1)
        y_probs = outputs.numpy()

    # 5. Show the ROC Curve
    print("\n--- RESULTS ---")
    plot_scientific_roc(y_true, y_probs)
    
    # Save a copy automatically for your GSoC application
    plt.savefig('results_roc_curve.png')
    print("✅ SUCCESS! Graph saved as 'results_roc_curve.png'")
    
    # Open the window so you can see it now
    plt.show()

except Exception as e:
    print(f"❌ Evaluation failed: {e}")