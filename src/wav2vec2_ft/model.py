import torch
from torchaudio.pipelines import WAV2VEC2_BASE as bundle
from torch import nn

class Wav2Vec2ForVoicePathology(nn.Module):
    def __init__(self):
        super(Wav2Vec2ForVoicePathology, self).__init__()
        self.backbone = bundle.get_model()
        self.backbone.requires_grad_(False)

        self.backbone.encoder.feature_projection.requires_grad_(True)

        for layer in self.backbone.encoder.transformer.layers[-4:]:
            layer.requires_grad_(True)

        self.head = nn.Sequential(
            nn.LayerNorm(768),
            nn.Dropout(0.3),
            nn.Linear(768, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, waveforms, valid_lengths):
        out, out_lengths = self.backbone.extract_features(waveforms, lengths = valid_lengths)
        x = out[-1]
        T = x.shape[1]
        mask = torch.arange(T, device = x.device).unsqueeze(0) < out_lengths.unsqueeze(1)
        mask = mask.unsqueeze(-1).to(x.dtype)

        pooled = (x * mask).sum(dim = 1) / mask.sum(dim = 1).clamp(1.0)
        pooled = self.head(pooled)
        return pooled

if __name__ == "__main__":
    from dataset import get_dataloaders
    model = Wav2Vec2ForVoicePathology()
    model.eval()
    t,_,_ = get_dataloaders()
    with torch.no_grad():
        for x,y,valid_lengths in t:
            model(x, valid_lengths)
            break