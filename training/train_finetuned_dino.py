import torch
import random
import numpy as np
import pandas as pd
import torchmetrics
import h5py
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from utils.CustomDataset import BaselineDataset
from models.general_classifier import BinaryClassifier
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModel
import torch.nn as nn
import os
import yaml
import argparse
from models.general_classifier import BinaryClassifier
from utils.matched_histo import compute_average_histogram

with open("configs/finetuned_dino.yaml", "r") as file:
    config = yaml.safe_load(file)

SEED = 0
torch.random.manual_seed(SEED)
random.seed(SEED)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Working on {device}.')

dino_model = AutoModel.from_pretrained("facebook/dinov2-small")
dino_model.to(device)
# LoRA Configuration
lora_config = LoraConfig(
    r=16,  # Rank
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"]  # Target attention layers
)

# Apply LoRA to DINOv2 Model
dino_lora = get_peft_model(dino_model, lora_config)

# Wrap Model
model = BinaryClassifier(dino_lora).to(device)



parser = argparse.ArgumentParser()
parser.add_argument("--histogram", type=bool, default=False)
args = parser.parse_args()

if args.histogram:
    mean_hists_list_train = []
    with h5py.File(config['train_path'], "r") as f:
        for img_id in f.keys():
            mean_hists_list_train.append(compute_average_histogram(f[img_id]['img']))

    mean_hists_train = np.mean(np.array(mean_hists_list_train), axis = 0)
    
    mean_hists_list_val = []
    with h5py.File(config['val_path'], "r") as f:
        for img_id in f.keys():
            mean_hists_list_val.append(compute_average_histogram(f[img_id]['img']))
        
    mean_hists_val = np.mean(np.array(mean_hists_list_val), axis = 0)
    transform_train = transforms.Compose([
        transforms.Resize((98, 98)),
        transforms.Lambda(lambda x: compute_average_histogram(x, mean_hists_train))
    ]) 
    
    transform_val = transforms.Compose([
        transforms.Resize((98, 98)),
        transforms.Lambda(lambda x: compute_average_histogram(x, mean_hists_val))
    ])
    
    train_dataset = BaselineDataset(config['train_path'], preprocessing=transform_train, mode='train')
    val_datatset = BaselineDataset(config['val_path'], preprocessing=transform_val, mode='train')
    
else:
    transform = transforms.Compose([
        transforms.Resize((98, 98))
    ])
    
    train_dataset = BaselineDataset(config['train_path'], preprocessing=transform, mode='train')
    val_datatset = BaselineDataset(config['val_path'], preprocessing=transform, mode='train')


train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_dataloader = DataLoader(val_datatset, batch_size=config['batch_size'], shuffle=False)

OPTIMIZER = config['optimizer']
OPTIMIZER_PARAMS = config['optimizer_params']
LOSS = config['loss']
METRIC = config['metric']
NUM_EPOCHS = int(config['num_epochs'])
PATIENCE = int(config['patience'])
optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), **OPTIMIZER_PARAMS)
criterion = getattr(torch.nn, LOSS)()
metric = getattr(torchmetrics, METRIC)('binary')
min_loss, best_epoch = float('inf'), 0

if not os.path.exists('/outputs'):
    os.makedirs('/outputs')
    
for epoch in range(NUM_EPOCHS):  # Train for 3 epochs
    model.train()
    train_metrics, train_losses = [], []
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device, dtype=torch.float32)
        labels = labels.unsqueeze(-1)  # Add an extra dimension to labels
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_losses.extend([loss.item()]*len(labels))
        train_metric = metric(outputs.cpu(), labels.int().cpu())
        train_metrics.extend([train_metric.item()]*len(labels))
    print(f'Epoch train [{epoch+1}/{NUM_EPOCHS}] | Loss {np.mean(train_losses):.4f} | Metric {np.mean(train_metrics):.4f}')
    
    model.eval()
    val_metrics, val_losses = [], []
    for val_x, val_y in val_dataloader:
        with torch.no_grad():
            val_pred = model(val_x.to(device)) # alpha=0 disables domain adaptation
            
        val_y = val_y.unsqueeze(-1).to(device)  # Add an extra dimension to labels
        loss = criterion(val_pred, val_y.to(device).float())
        val_losses.extend([loss.item()] * len(val_y))
        val_metric = metric(val_pred.cpu(), val_y.int().cpu())
        val_metrics.extend([val_metric.item()] * len(val_y)) 
    print(f'Epoch valid [{epoch+1}/{NUM_EPOCHS}] | Loss {np.mean(val_losses):.4f} | Metric {np.mean(val_metrics):.4f}')
    
    if np.mean(val_losses) < min_loss:
        mean_val_loss = np.mean(val_losses)
        print(f'New best loss {min_loss:.4f} -> {mean_val_loss:.4f}')
        min_loss = mean_val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), './outputs/best_model_finetuned.pth')

    if epoch - best_epoch == PATIENCE:
        break