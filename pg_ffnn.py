import torch.nn as nn

model_code = "pg"
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.convolutions = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.ReLU(),

            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU()
        )

        self.layers = nn.Sequential(
            nn.Linear(34560, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 2)
        )

        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        # Small initialization for the final output layer
        # This ensures initial actions are almost equally likely
        nn.init.uniform_(self.layers[-1].weight, -1e-2, 1e-2)
        nn.init.constant_(self.layers[-1].bias, 0)

    def forward(self, x):
        x = self.convolutions(x)
        x = x.flatten(start_dim=1)
        return self.layers(x)