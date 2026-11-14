import torch.nn as nn

model_code = "pg"

width = 512
height = 320
time = 4
state_sizes = [time * height * width, 512, 256, 2]  # input size, hidden states, output size

class Layer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.activation(self.linear(x))
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        global state_sizes

        self.layers = nn.ModuleList()
        for i in range(len(state_sizes) - 2):
            self.layers.append(Layer(state_sizes[i], state_sizes[i + 1]))

        self.out = nn.Linear(state_sizes[-2], state_sizes[-1])

        for m in self.layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

        # Small initialization for the final output layer
        # This ensures initial actions are almost equally likely
        nn.init.uniform_(self.out.weight, -1e-2, 1e-2)
        nn.init.constant_(self.out.bias, 0)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        for layer in self.layers:
            x = layer(x)
        return self.out(x)