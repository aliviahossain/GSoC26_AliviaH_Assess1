import torch
import torch.nn as nn

class BayesianLensingCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(BayesianLensingCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.3), # This stays ON during testing
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(p=0.3)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 37 * 37, 128), # Note: 37*37 is for 150x150 images
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def enable_dropout(self):
        """ This is the Bayesian Trick: it keeps Dropout active during testing """
        for m in self.modules():
            if isinstance(m, nn.Dropout) or isinstance(m, nn.Dropout2d):
                m.train()