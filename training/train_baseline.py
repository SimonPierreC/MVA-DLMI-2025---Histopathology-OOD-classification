import h5py
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms
import torchmetrics
import numpy as np
import os
import pandas as pd
from utils.CustomDataset  import BaselineDataset, PrecomputedDataset
import yaml

# Set random seed for reproducibility
SEED = 0
torch.random.manual_seed(SEED)
random.seed(SEED)


with open("configs/baseline.yaml", "r") as file:
        config = yaml.safe_load(file)

train_path = config['train_path']
val_path = config['val_path']

def precompute(dataloader, model, device):
    xs, ys = [], []
    for x, y in tqdm(dataloader, leave=False):
        with torch.no_grad():
            xs.append(model(x.to(device)).detach().cpu().numpy())
        ys.append(y.numpy())
    xs = np.vstack(xs)
    ys = np.concatenate(ys, axis=0) if ys else np.array([])
    return torch.tensor(xs), torch.tensor(ys)

def precompute_dataset(path, model, device, batch_size = 16, mode = 'train'):
    preprocessing = transforms.Resize((98, 98))
    dataset = BaselineDataset(path, preprocessing, mode)
    dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size)
    model.eval()
    return PrecomputedDataset(*precompute(dataloader, model, device))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Working on {device}.')

print('Loading model...')
feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
feature_extractor.eval()

print('Loading datasets...')
train_dataset = precompute_dataset(train_path, feature_extractor, device)
val_dataset = precompute_dataset(val_path, feature_extractor, device)
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=config['batch_size'])
val_dataloader = DataLoader(val_dataset, shuffle=False, batch_size=config['batch_size'])

OPTIMIZER = config['optimizer']
OPTIMIZER_PARAMS = config['optimizer_params']
LOSS = config['loss']
METRIC = config['metric']
NUM_EPOCHS = config['num_epochs']
PATIENCE = config['patience']

linear_probing = torch.nn.Sequential(
    torch.nn.Linear(256, 128),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 64),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 1)
).to(device)

optimizer = getattr(torch.optim, OPTIMIZER)(linear_probing.parameters(), **OPTIMIZER_PARAMS)
criterion = getattr(torch.nn, LOSS)()
metric = getattr(torchmetrics, METRIC)('binary')
min_loss, best_epoch = float('inf'), 0

if not os.path.exists('/outputs'):
    os.makedirs('/outputs')

for epoch in range(NUM_EPOCHS):
    linear_probing.train()
    train_metrics, train_losses = [], []
    for train_x, train_y in train_dataloader:
        optimizer.zero_grad()
        train_pred = linear_probing(train_x.to(device))
        loss = criterion(train_pred, train_y.to(device))
        loss.backward()
        optimizer.step()
        train_losses.extend([loss.item()]*len(train_y))
        train_metric = metric(train_pred.cpu(), train_y.int().cpu())
        train_metrics.extend([train_metric.item()]*len(train_y))
    print(f'Epoch train [{epoch+1}/{NUM_EPOCHS}] | Loss {np.mean(train_losses):.4f} | Metric {np.mean(train_metrics):.4f}')

    linear_probing.eval()
    val_metrics, val_losses = [], []
    for val_x, val_y in val_dataloader:
        with torch.no_grad():
            val_pred = linear_probing(val_x.to(device))
        loss = criterion(val_pred, val_y.to(device))
        val_losses.extend([loss.item()]*len(val_y))
        val_metric = metric(val_pred.cpu(), val_y.int().cpu())
        val_metrics.extend([val_metric.item()]*len(val_y))
    print(f'Epoch valid [{epoch+1}/{NUM_EPOCHS}] | Loss {np.mean(val_losses):.4f} | Metric {np.mean(val_metrics):.4f}')

    if np.mean(val_losses) < min_loss:
        mean_val_loss = np.mean(val_losses)
        print(f'New best loss {min_loss:.4f} -> {mean_val_loss:.4f}')
        min_loss = mean_val_loss
        best_epoch = epoch
        torch.save(linear_probing.state_dict(), './outputs/best_model.pth')

    if epoch - best_epoch == PATIENCE:
        break
