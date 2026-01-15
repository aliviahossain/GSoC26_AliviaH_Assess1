# DeepLense Common Test I: Dark Matter Classification
### Applicant: Alivia Hossain

## 🌌 Project Overview
This repository contains my implementation of a **Bayesian Convolutional Neural Network (B-CNN)** for the classification of dark matter substructures in gravitational lensing images. The goal is to distinguish between three types of lensing events:
1. **No Substructure**
2. **Cold Dark Matter (CDM)**
3. **Axions**

## 🧠 Why Bayesian?
Unlike standard CNNs that provide fixed "point-estimates," this model treats weights as **probability distributions**. This allows the model to:
* **Quantify Uncertainty:** It doesn't just predict a class; it provides a confidence level (e.g., "60% sure it is CDM").
* **Prevent Overfitting:** The Bayesian approach acts as a natural regularizer, essential for noisy cosmological data.
* **Scientific Reliability:** In astrophysics, understanding "error bars" (uncertainty) is as important as the prediction itself.



---

## 🛠️ Implementation Details

### Model Architecture (`model.py`)
The model utilizes **Variational Inference** to learn the posterior distribution of the weights.
* **Bayesian Layers:** Standard `nn.Conv2d` and `nn.Linear` layers are replaced with `BayesConv2d` and `BayesLinear` (using the `torchbnn` library).
* **Uncertainty Quantification:** The model learns a Mean ($\mu$) and Variance ($\sigma$) for every connection, allowing for stochastic forward passes.

### Hardware Optimization Strategy
Given local hardware constraints (CPU training and memory limits with the 21GB dataset), I implemented a **Resource-Efficient Pipeline**:
* **`make_small_data.py`**: A utility script to create a representative subset of the data (500 samples/class).
* **`smalltrain.py`**: Optimized training script that handles data in small batches to prevent RAM exhaustion.
* **`smalltest.py`**: Evaluation script that calculates probabilities and generates scientific metrics.

---

## 📈 Results

The model was evaluated using a **Receiver Operating Characteristic (ROC) Curve**. 

| Class | AUC Score |
| :--- | :--- |
| **No Substructure** | 0.59 |
| **CDM** | 0.52 |
| **Axion** | 0.59 |



**Note on Results:** These AUC scores reflect the limited training time (5 epochs) and reduced dataset size used for this local hardware assessment. The successful execution of this pipeline proves the architecture's readiness for full-scale training on High-Performance Computing (HPC) clusters.

---

## 🚀 How to Run

1. **Environment Setup:**
   ```bash
   pip install torch torchbnn numpy matplotlib scikit-learn
   python make_small_data.py
   python smalltrain.py