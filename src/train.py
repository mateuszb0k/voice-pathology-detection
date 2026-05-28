import torch
from dataset import get_data_loaders
from model import VoicePathologyModel
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
optimizer= torch.optim.Adam(model.parameters())
lr_reducer = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience =5,
    factor = 0.1,
)
train_loader, val_loader, test_loader = get_data_loaders()
criterion = torch.nn.CrossEntropyLoss(weight = class_weights)
EPOCHS = 50
best_val_loss = float('inf')
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
    with torch.no_grad():
        for x,y in val_loader:
            x = x.to(device)
            y = y.to(device)
            outputs = model(x)
            loss=criterion(outputs, y)
            outputs = torch.argmax(outputs, dim = 1)
            correct += (outputs == y).sum().item()
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_accuracy = correct / len(val_loader.dataset)
    lr_reducer.step(val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(),f'Epoch-{epoch}-best.pth')
    print(f"Validation loss: {val_loss} Validation accuracy: {val_accuracy} LR: {optimizer.param_groups[0]['lr']} train_loss: {loss_avg}")



