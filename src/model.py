import torch.nn as nn

class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        # Feature extraction: convolution + pooling layers + fully connected
        self.features = nn.Sequential(
            # Conv layer 1: (1, 28, 28) -> (32, 26, 26)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),  # ReLU activation
            # MaxPool 1: (32, 26, 26) -> (32, 13, 13)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv layer 2: (32, 13, 13) -> (64, 11, 11)
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),  # ReLU activation
            # MaxPool 2: (64, 11, 11) -> (64, 5, 5)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv layer 2: (64, 5, 5) -> (128, 4, 4)
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2),
            nn.ReLU(),  # ReLU activation
            # Flatten 3D to 1D for fully connected layers
            nn.Flatten(),
            # Fully connected layers
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, 10), # Output layer (10 classes)
        )

    def forward(self, x):
        return self.features(x)  # Forward pass through the model

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        # Feature extraction: convolution + pooling layers + fully connected
        self.features = nn.Sequential(
            # Conv layer 1: (1, 28, 28) -> (16, 26, 26)
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
            nn.ReLU(),  # ReLU activation
            # MaxPool 1: (16, 26, 26) -> (16, 13, 13)
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Conv layer 2: (16, 13, 13) -> (32, 11, 11)
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
            nn.Flatten(),
            # Fully connected layers
            nn.Linear(32*11*11, 10), # Output layer (10 classes)
        )
    def forward(self, x):
        return self.features(x)  # Forward pass through the model
