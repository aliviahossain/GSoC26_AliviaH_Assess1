import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import numpy as np

def plot_scientific_roc(y_true, y_probs):
    """ Standard ML4Sci requirement: Multi-class ROC curve """
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2])
    plt.figure(figsize=(10, 6))
    colors = ['blue', 'red', 'green']
    labels = ['No Substructure', 'CDM', 'Axion']
    
    for i in range(3):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_probs[:, i])
        plt.plot(fpr, tpr, color=colors[i], label=f'{labels[i]} (AUC = {auc(fpr, tpr):.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('DeepLense Common Test I: ROC Analysis')
    plt.legend()
    plt.show()