import torch
from dataset import get_data_loaders
from model import VoicePathologyModel,ModifiedResNet
from sklearn.metrics import roc_auc_score,recall_score
import torch.nn.functional as F
import numpy as np
import tqdm
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
TRAIN_X_PATH = '../data/processed/train_X.npy'
VAL_X_PATH = '../data/processed/val_X.npy'
TEST_X_PATH = '../data/processed/test_X.npy'
TRAIN_Y_PATH = '../data/processed/train_y.npy'
VAL_Y_PATH = '../data/processed/val_y.npy'
TEST_Y_PATH = '../data/processed/test_y.npy'
#fix
train_y = np.load('../data/processed/train_y.npy',allow_pickle=True)
pathological_count = np.count_nonzero(train_y)
pathological_weight = len(train_y)/(2*pathological_count)
healthy_weight = len(train_y)/(2*(len(train_y)-pathological_count))
class_weights = torch.tensor([healthy_weight,pathological_weight],dtype=torch.float32).to(device)
model = VoicePathologyModel().to(device)
optimizer= torch.optim.Adam(model.parameters(),lr=1e-4)
EPOCHS = 50
lr_reducer = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,T_max = EPOCHS,eta_min = 1e-6
)
train_loader, val_loader, test_loader = get_data_loaders()
criterion = torch.nn.CrossEntropyLoss(weight = class_weights,label_smoothing=0.1)

best_val_loss = float('inf')
patience_counter = 0
PATIENCE = 30
best_auc = 0.0
for epoch in tqdm.tqdm(range(EPOCHS)):
    model.train()
    running_loss=0.0
    val_loss=0.0
    correct = 0
    for x,y in train_loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    loss_avg = running_loss / len(train_loader)
    model.eval()
    all_probs,all_labels = [],[]
    val_loss =0.0
    with torch.no_grad():
        for x,y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
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
    val_accuracy = (np.array(preds) == np.array(all_labels)).mean()
    if auc>best_auc:
        best_auc = auc
        patience_counter = 0
        torch.save(model.state_dict(),f'../data/weightbest.pth')
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping")
            break
    lr_reducer.step()
    print(f"val_loss: {val_loss:.4f} | acc: {val_accuracy:.4f} | AUC: {auc:.4f} | UAR: {uar:.4f} | sens: {sens:.4f} | spec: {spec:.4f} | train_loss: {loss_avg:.4f}")
    print(f"Best auc score: {best_auc:.4f}")


