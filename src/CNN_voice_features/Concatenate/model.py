import torch
from torch import nn

from src.CNN.model import VoicePathologyModel


class fc_model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(216,64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32,2)
        )
    def forward(self, x):
        x = self.classifier(x)
        return x

class ConcatenateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn_model = VoicePathologyModel()
        del self.cnn_model.classifier
        self.fc_model = fc_model()
    def forward(self,x1,x2):
        y1 = self.cnn_model.features(x1)
        y1 = self.cnn_model.gap(y1)
        y1 = torch.flatten(y1,start_dim=1)
        x2 = torch.cat((y1,x2),dim=1)
        y = self.fc_model(x2)
        return y

if __name__ == '__main__':

    model = ConcatenateModel()
    model.eval()

    dummy_input = torch.randn(1,1,128,188)
    dummy_features = torch.randn(1,88)

    with torch.no_grad():
        output = model(dummy_input,dummy_features)
        print(output.shape)