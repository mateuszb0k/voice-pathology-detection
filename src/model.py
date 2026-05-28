import torch.nn as nn
import torch
class VoicePathologyModel(nn.Module):
    def __init__(self):
        super(VoicePathologyModel, self).__init__()
        self.model = nn.Sequential(
            ConvBlock(1,16),
            ConvBlock(16,32),
            ConvBlock(32,64),
            ConvBlock(64,128),
            nn.Flatten(),
            LinearBlock(128*8*3,256)
        )
    def forward(self, x):
        return self.model(x)
class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
    def forward(self, x):
        return self.block(x)
class LinearBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(LinearBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channel,out_channel),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(out_channel,2)
        )
    def forward(self, x):
        return self.block(x)
model = VoicePathologyModel()
dummy_x = torch.randn(1,1,128,63)
output = model(dummy_x)
print(output.shape)