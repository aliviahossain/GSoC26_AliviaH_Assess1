# DeepLense GSoC 2026 Assessment - Task 1: Dark Matter Classification
### **Applicant: Alivia Hossain**
**GitHub:** [@aliviahossain](https://github.com/aliviahossain) | **Organization:** ML4SCI (DeepLense)

---

## 🌌 Project Overview
This repository contains my implementation for the DeepLense Task I. The objective is to classify simulated gravitational lensing images into three categories based on dark matter substructures:
1. **No Substructure** (Smooth lensing rings)
2. **Subhalo** (Cold Dark Matter signatures)
3. **Vortex** (Axions / Fuzzy Dark Matter signatures)

The project utilizes a **Bayesian Convolutional Neural Network (B-CNN)** to provide not only predictions but also a measure of **uncertainty**, which is critical for high-stakes cosmological research.

---

## 🧠 Technical Strategy: The Bayesian Advantage
Most standard CNNs provide "point-estimates," which can lead to overconfidence in noisy data. My approach incorporates:

* **Monte Carlo (MC) Dropout:** By keeping Dropout active during inference, the model performs **20 stochastic forward passes** per image.
* **Uncertainty Quantification:** The final prediction is the mean of these passes. This smooths the ROC curves and provides "error bars" for the classification.
* **Resource Optimization:** The architecture is optimized for CPU training, ensuring that the full pipeline remains accessible and reproducible on standard local hardware.



---

## 📈 Results & Metrics

The model was evaluated using the **Area Under the Receiver Operating Characteristic (ROC) Curve**. 

| Class | Peak AUC Score | Current Run AUC (Stability Check) |
| :--- | :---: | :---: |
| **No Substructure** | **0.780** | **0.619** |
| **Subhalo (CDM)** | **0.630** | **0.493** |
| **Vortex (Axions)** | **0.632** | **0.576** |

**Scientific Interpretation:** The model shows a high degree of success in **Detection** (No Substructure vs. anything else). While the distinction between **Subhalo** and **Vortex** classes shows higher aleatoric uncertainty due to resolution limits, the Bayesian framework correctly identifies these overlaps, allowing for more reliable scientific conclusions.



---

## 🛠️ Implementation Details

### **Core Files**
* **`work.ipynb`**: The primary Jupyter Notebook containing the full training and evaluation pipeline.
* **`lensing_model.pth`**: The pre-trained weights for the Bayesian CNN.
* **`requirements.txt`**: List of all dependencies required to replicate the environment.
* **`.gitignore`**: Configured to exclude heavy dataset files and virtual environments while keeping weights and code tracked.

---

## 🚀 How to Run

### **1. Setup the Environment**
Use a fresh virtual environment to avoid permission or dependency conflicts.

```bash
# Create and activate environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

```

### **2. Prepare the data**
Since the dataset is excluded from the repository due to size:

1. Ensure your dataset is located in dataset/train/ within the root directory.

2. Structure:

dataset/train/no_substructure/
dataset/train/subhalo/
dataset/train/vortex/

3. Execution
Open work.ipynb in VS Code or Jupyter and select the venv kernel. Click "Run All".

Training Time: ~10 minutes (15 Epochs on CPU).

Evaluation: The notebook automatically generates the 3-class ROC Curve using 20 MC samples.

## 🛠 Future Work
- Scaling to the full 21GB dataset on an NVIDIA A100/H100 cluster.
- Implementing a deeper ResNet-Bayesian hybrid architecture.
- Exploring 'Expected Information Gain' for active learning in lensing detection.

## 📬 Contact

**Alivia Hossain** * [GitHub Profile](https://github.com/aliviahossain)
[LinkedIn Profile](https://www.linkedin.com/in/alivia-hossain-513a3a365/)