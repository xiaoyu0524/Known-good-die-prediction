
import torch
from torch import nn


# Build neuron network
class ds_predict_nn(nn.Module):
    def __init__(self):
        super(ds_predict_nn, self).__init__()
        self.nn_layers = nn.Sequential(
            nn.Linear(1328, 4000),
            nn.ReLU(),
            #nn.Dropout(p=0.1),
            nn.Linear(4000, 400),
            nn.ReLU(),
            nn.Linear(400, 20),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(20, 2),
            nn.ReLU(),
        )

    def forward(self, input):
        output = self.nn_layers(input)
        return output


if __name__ == '__main__':
    ds_predict_model = ds_predict_nn()

    # Model and output verification with test input
    input = torch.ones((1328))
    print(input.shape)
    output = ds_predict_model(input)
    print(output.shape)
    