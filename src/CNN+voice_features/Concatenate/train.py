import torch
from dataset import get_dataloaders
from model import ConcatenateModel
from sklearn.metrics import roc_auc_score, recall_score,accuracy_score
import torch.nn.functional as F
import numpy as np
import tqdm

PATHOLOGICAL_COUNT = 10794
DATASET_LENGTH = 16396
EPOCHS = 150
PATIENCE = 30

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

PATHOLOGICAL_WEIGHT = DATASET_LENGTH/(2*PATHOLOGICAL_COUNT)
HEALTHY_WEIGHT = DATASET_LENGTH/(2*(DATASET_LENGTH-PATHOLOGICAL_COUNT))
CLASS_WEIGHTS = torch.tensor([HEALTHY_WEIGHT,PATHOLOGICAL_WEIGHT],dtype=torch.float32).to(device)

model = ConcatenateModel().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

lr_reducer = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,T_max = EPOCHS,eta_min = 1e-6
)

train_loader, val_loader, test_loader = get_dataloaders()

criterion = torch.nn.CrossEntropyLoss(weight = CLASS_WEIGHTS,label_smoothing=0.1)

patience_counter = 0
best_auc = 0.0

for epoch in tqdm.tqdm(range(EPOCHS)):
    model.train()
    running_loss=0.0
    val_loss=0.0

    for x,feats,y in train_loader:
        x = x.to(device)
        feats = feats.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        outputs = model(x,feats)
        loss = criterion(outputs, y)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    loss_avg = running_loss / len(train_loader)

    model.eval()

    all_probs,all_labels = [],[]
    val_loss =0.0

    with torch.no_grad():
        for x,feats,y in val_loader:

            x,feats,y = x.to(device),feats.to(device),y.to(device)
            outputs = model(x,feats)

            loss = criterion(outputs, y)
            val_loss += loss.item()

            probs = F.softmax(outputs, dim=1)[:, 1]
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    val_loss /= len(val_loader)

    preds = (np.array(all_probs) > 0.5).astype(int)

    auc = roc_auc_score(all_labels, all_probs)
    sens = recall_score(all_labels, preds, pos_label=1)
    spec = recall_score(all_labels, preds, pos_label=0)
    uar = (sens + spec) / 2
    val_accuracy = accuracy_score(all_labels, preds)

    if auc>best_auc:
        best_auc = auc
        patience_counter = 0
        torch.save(model.state_dict(), f'../../../data/Concatenate_best.pth')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping")
            break
    lr_reducer.step()
    print(f"val_loss: {val_loss:.4f} | acc: {val_accuracy:.4f} | AUC: {auc:.4f} | UAR: {uar:.4f} | sens: {sens:.4f} | spec: {spec:.4f} | train_loss: {loss_avg:.4f}")
    print(f"Best auc score: {best_auc:.4f}")




