from torch.nn import MultiheadAttention
from torch import nn
from src.CNN.model import VoicePathologyModel
import torch

class eGeMAPSEncoder(nn.Module):
    def __init__(self):
        super(eGeMAPSEncoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(88, 64)
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.unsqueeze(1)
        return x


class SpectrogramEncoder(nn.Module):
    def __init__(self):
        super(SpectrogramEncoder, self).__init__()

        self.features = VoicePathologyModel().features

        self.encoder = nn.Linear(128, 64)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        x = self.encoder(x)
        return x


class CrossAttentionModel(nn.Module):
    def __init__(self):
        super(CrossAttentionModel, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(128, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 2)
        )

        self.spectrogram_encoder = SpectrogramEncoder()
        self.feature_encoder = eGeMAPSEncoder()

        self.multihead_attention = MultiheadAttention(64, 4, batch_first=True)

    def forward(self, features, spectrogram):
        spec_features = self.spectrogram_encoder(spectrogram)
        features_representation = self.feature_encoder(features)

        attn_output, attn_weights = self.multihead_attention(query=features_representation, key=spec_features,
                                                             value=spec_features)

        attn_output = attn_output.squeeze(1)
        features_representation = features_representation.squeeze(1)

        cat = torch.cat((features_representation, attn_output), dim=1)

        output = self.classifier(cat)

        return output, attn_weights


if __name__ == '__main__':
    dummy_spec = torch.randn(1,1,128,188)
    dummy_features = torch.randn(1,88)

    model = CrossAttentionModel()
    model.eval()
    with torch.no_grad():
        out = model(features = dummy_features, spectrogram = dummy_spec)
        print(out)