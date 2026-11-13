import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from model import CrossAttentionModel
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score

from src.CNN_voice_features.Concatenate.dataset import get_dataloaders

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

EPOCHS  = 150
PATIENCE = 30

PATHOLOGICAL_COUNT = 10794
DATASET_LENGTH = 16396
PATHOLOGICAL_WEIGHT = DATASET_LENGTH/(2*PATHOLOGICAL_COUNT)
HEALTHY_WEIGHT = DATASET_LENGTH/(2*(DATASET_LENGTH-PATHOLOGICAL_COUNT))
CLASS_WEIGHTS = torch.tensor([HEALTHY_WEIGHT,PATHOLOGICAL_WEIGHT],dtype=torch.float32).to(device)

train_dataloader, val_dataloader, test_dataloader = get_dataloaders()

model = CrossAttentionModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

lr_reducer = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,T_max = EPOCHS,eta_min = 1e-6
)


criterion = torch.nn.CrossEntropyLoss(weight = CLASS_WEIGHTS,label_smoothing=0.1)

best_auc = 0.0
patience_counter = 0

for epoch in tqdm.tqdm(range(EPOCHS)):
    model.train()
    running_loss = 0.0
    val_loss = 0.0

    for x,feats,y in train_dataloader:
        x = x.to(device)
        feats = feats.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs, weights = model(features = feats, spectrogram = x)
        loss = criterion(outputs,y)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    loss_avg = running_loss / len(train_dataloader)
    print(f"Training loss: {loss_avg:.4f}")

    model.eval()

    all_probs, all_labels = [], []
    val_loss = 0.0

    with torch.no_grad():
        for x,feats,y in val_dataloader:
            x = x.to(device)
            feats = feats.to(device)
            y = y.to(device)

            outputs, weights = model(features = feats, spectrogram = x)

            loss = criterion(outputs,y)

            val_loss += loss.item()

            probs = F.softmax(outputs, dim=1)[:,1]

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    val_loss = val_loss / len(val_dataloader)

    preds = (np.array(all_probs) > 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_probs)
    sens = recall_score(all_labels, preds, pos_label=1)
    spec = recall_score(all_labels, preds, pos_label=0)
    uar = (sens + spec) / 2
    val_accuracy = accuracy_score(all_labels, preds)

    if auc > best_auc:
        best_auc = auc
        patience_counter = 0

        torch.save(model.state_dict(),'../../../models/new_cross_attention_best_model.pth')

    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping")
            break

    lr_reducer.step()
    print(f"val_loss: {val_loss:.4f} | acc: {val_accuracy:.4f} | AUC: {auc:.4f} | UAR: {uar:.4f} | sens: {sens:.4f} | spec: {spec:.4f} | train_loss: {loss_avg:.4f}")
    print(f"Best auc score: {best_auc:.4f}")
