import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from peft import LoraConfig, get_peft_model
from transformers import AutoModel
import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F
from PIL import Image
import torchmetrics
import random
import os
import argparse
from utils.CustomDataset import CenterAwareDataset
import yaml
from models.adversarial import AdversarialClassifier


with open("configs/adv_dino.yaml", "r") as file:
    config = yaml.safe_load(file)
# Random for reproducibility
torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("--lbda", type=float, default=0.1)
args = parser.parse_args()

    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Working on {device}.')

preprocessing = transforms.Compose([
        transforms.Resize((98, 98))
    ])
train_dataset = CenterAwareDataset(config['train_path'], preprocessing, mode='train')
val_dataset = CenterAwareDataset(config['val_path'], preprocessing, mode='train')


train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)

dino_model = AutoModel.from_pretrained("facebook/dinov2-small")

dino_model.to(device)
# LoRA Configuration
lora_config = LoraConfig(
    r=4,  # Rank
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"]  # Target attention layers
)

# Apply LoRA to DINOv2 Model
dino_lora = get_peft_model(dino_model, lora_config)

model = AdversarialClassifier(dino_lora, 384, 2, 3, lambda_=0).to(device)

OPTIMIZER = config['optimizer']
OPTIMIZER_PARAMS = config['optimizer_params']
TASK_LOSS = config['task_loss']
DOMAIN_LOSS = config['domain_loss']
METRIC = config['metric']
NUM_EPOCHS = int(config['num_epochs'])
PATIENCE = int(config['patience'])
optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), **OPTIMIZER_PARAMS)
criterion_class  = getattr(torch.nn, TASK_LOSS)()
criterion_domain = getattr(torch.nn, DOMAIN_LOSS)()
metric = getattr(torchmetrics, METRIC)('binary')
min_loss, best_epoch = float('inf'), 0
p = np.linspace(0, 1, NUM_EPOCHS)
lambdas_list = args.lbda * np.ones(NUM_EPOCHS)
gamma = 10
if args.lbda != 0.1:
    lmbda = 2 / (1 + np.exp(-gamma * p)) - 1
    
if not os.path.exists('/outputs'):
    os.makedirs('/outputs')

for epoch in range(NUM_EPOCHS):  # Train for 3 epochs
    model.train()
    model.lambda_ = lambdas_list[epoch]
    train_metrics, train_losses = [], []
    for images, labels, centers in train_loader:
        images, labels, centers = images.to(device), labels.to(device), centers.to(device)
        optimizer.zero_grad()
        class_out, center_out = model(images)
        centers = centers // 2
        labels = labels.view(-1, 1).float()
        loss_class = criterion_class(class_out, labels)
        loss_adv = criterion_domain(center_out, centers)
        loss = loss_class + loss_adv
        loss.backward()
        optimizer.step()
        train_losses.extend([loss.item()]*len(labels))
        train_metric = metric(class_out.cpu(), labels.int().cpu())
        train_metrics.extend([train_metric.item()]*len(labels))
    print(f'Epoch train [{epoch+1}/{NUM_EPOCHS}] | Loss {np.mean(train_losses):.4f} | Metric {np.mean(train_metrics):.4f}')

    model.eval()
    val_metrics, val_losses = [], []
    for val_x, val_y, centers in val_loader:
        with torch.no_grad():
            val_pred, centers_out = model(val_x.to(device))  # alpha=0 disables domain adaptation

        val_y = val_y.view(-1, 1).float()
        loss = criterion_class(val_pred, val_y.to(device))
        val_losses.extend([loss.item()] * len(val_y))
        val_metric = metric(val_pred.cpu(), val_y.int().cpu())
        val_metrics.extend([val_metric.item()] * len(val_y))
    print(f'Epoch valid [{epoch+1}/{NUM_EPOCHS}] | Loss {np.mean(val_losses):.4f} | Metric {np.mean(val_metrics):.4f}')

    if np.mean(val_losses) < min_loss:
        mean_val_loss = np.mean(val_losses)
        print(f'New best loss {min_loss:.4f} -> {mean_val_loss:.4f}')
        min_loss = mean_val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), './outputs/dino_lora_adversarial.pth')

    if epoch - best_epoch == PATIENCE:
        break
        
print("Fine-tuning complete! Model saved as dino_lora_adversarial.pth")