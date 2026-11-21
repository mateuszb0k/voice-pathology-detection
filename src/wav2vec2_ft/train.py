import torch
import tqdm
import os
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score
from dataset import get_dataloaders
from model import Wav2Vec2ForVoicePathology

def main():
    os.makedirs('../../models', exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    PATHOLOGICAL_COUNT = 10794
    DATASET_LENGTH = 16396
    PATHOLOGICAL_WEIGHT = DATASET_LENGTH / (2 * PATHOLOGICAL_COUNT)
    HEALTHY_WEIGHT = DATASET_LENGTH / (2 * (DATASET_LENGTH - PATHOLOGICAL_COUNT))
    CLASS_WEIGHTS = torch.tensor([HEALTHY_WEIGHT, PATHOLOGICAL_WEIGHT], dtype=torch.float32).to(device)

    EPOCHS = 20
    PATIENCE = 5

    train_dl, val_dl, test_dl = get_dataloaders()

    total_steps = len(train_dl) * EPOCHS
    warmup_steps = int(total_steps * 0.1)

    model = Wav2Vec2ForVoicePathology().to(device)

    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": 3e-5, "weight_decay": 0.01},
        {"params": model.head.parameters(), "lr": 3e-4, "weight_decay": 0.01},
    ])

    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])

    criterion = torch.nn.CrossEntropyLoss(weight=CLASS_WEIGHTS, label_smoothing=0.1)

    patience_counter = 0
    best_auc = 0.0

    for epoch in tqdm.tqdm(range(EPOCHS)):
        model.train()
        running_loss = 0.0
        val_loss = 0.0

        for x, y, valid_lengths in train_dl:
            x = x.to(device)
            y = y.to(device)
            valid_lengths = valid_lengths.to(device)

            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = model(x, valid_lengths)

                loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()

        loss_avg = running_loss / len(train_dl)

        model.eval()

        all_probs, all_labels = [], []
        val_loss = 0.0

        with torch.no_grad():
            for x, y, valid_lengths in val_dl:

                x = x.to(device)
                y = y.to(device)
                valid_lengths = valid_lengths.to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = model(x, valid_lengths)
                    loss = criterion(outputs, y)
                val_loss += loss.item()

                probs = nn.functional.softmax(outputs.float(), dim=-1)[:, 1]

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        val_loss = val_loss / len(val_dl)

        preds = (np.array(all_probs) > 0.5).astype(int)

        auc = roc_auc_score(all_labels, all_probs)
        sens = recall_score(all_labels, preds, pos_label=1)
        spec = recall_score(all_labels, preds, pos_label=0)
        uar = (sens + spec) / 2
        val_accuracy = accuracy_score(all_labels, preds)

        if auc > best_auc:
            best_auc = auc
            patience_counter = 0
            torch.save(model.state_dict(), '../../models/best_w2v2_ft.pt')
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping")
                break
        print(
            f"EPOCH {epoch} val_loss: {val_loss:.4f} | acc: {val_accuracy:.4f} | AUC: {auc:.4f} | UAR: {uar:.4f} | sens: {sens:.4f} | spec: {spec:.4f} | train_loss: {loss_avg:.4f}")
        print(f"Best auc score: {best_auc:.4f}")

if __name__ == "__main__":
    main()