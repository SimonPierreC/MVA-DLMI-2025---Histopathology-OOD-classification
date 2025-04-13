import torch
import random
import numpy as np
import pandas as pd
import torchmetrics
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from utils.CustomDataset import BaselineDataset
from models.general_classifier import BinaryClassifier
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModel
import torch.nn as nn
import os

# Set random seed for reproducibility
SEED = 0
torch.random.manual_seed(SEED)
random.seed(SEED)

cutmix = v2.CutMix(num_classes=2)
mixup = v2.MixUp(num_classes=2)
cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])


with open("configs/cutmix.yaml", "r") as file:
        config = yaml.safe_load(file)
        
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
dino_model = AutoModel.from_pretrained("facebook/dinov2-small")
dino_model.to(device)
# LoRA Configuration
lora_config = LoraConfig(
    r=8,  # Rank
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["query", "value"]  # Target attention layers
)

# Apply LoRA to DINOv2 Model
dino_lora = get_peft_model(dino_model, lora_config)

model = BinaryClassifier(dino_lora, output_dim=2).to(device)

def get_dataloader(batch_size, path_to_train):
    transform = transforms.Compose([
        transforms.Resize((98, 98))
    ])
    dataset = BaselineDataset(
        path_to_train,
        preprocessing=transform, mode='train'
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

train_dataloader = get_dataloader(config['batch_size'], config['train_path'])
val_dataloader = get_dataloader(config['batch_size'], config['val_path'])


OPTIMIZER = config['optimizer']
OPTIMIZER_PARAMS = config['optimizer_params']
LOSS = config['loss']
METRIC = config['metric']
NUM_EPOCHS = int(config['num_epochs'])
PATIENCE = int(config['patience'])
batch_size = config['batch_size']
optimizer = getattr(torch.optim, OPTIMIZER)(model.parameters(), **OPTIMIZER_PARAMS)
criterion = getattr(torch.nn, LOSS)()
metric = getattr(torchmetrics, METRIC)('binary')
min_loss, best_epoch = float('inf'), 0

if not os.path.exists('./outputs'):
    os.makedirs('./outputs')
    
# Training Loop
for epoch in range(NUM_EPOCHS):
    model.train()
    train_metrics, train_losses = [], []
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        output_1 = model(images)
        images_augmented, labels_augmented = cutmix_or_mixup(images, labels)
        optimizer.zero_grad()
        output_2 = model(images_augmented)
        outputs = torch.stack((output_1, output_2)).view(2*batch_size, 2)
        labels = nn.functional.one_hot(labels, num_classes=2)
        y = torch.stack((labels, labels_augmented)).view(2*batch_size, 2)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    
        # Track metrics
        train_losses.extend([loss.item()] * len(y))
        train_metric = metric(outputs.cpu(), y.int().cpu())
        train_metrics.extend([train_metric.item()] * len(y))
    
    print(f'Epoch train [{epoch+1}/{NUM_EPOCHS}] | Loss {np.mean(train_losses):.4f} | Metric {np.mean(train_metrics):.4f}')

     # Validation (without domain adaptation)
    model.eval()
    val_metrics, val_losses = [], []
    for val_x, val_y in val_dataloader:
        with torch.no_grad():
            val_pred = model(val_x.to(device))[:, 1] # alpha=0 disables domain adaptation
            
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
        torch.save(model.state_dict(), './outputs/best_model_cutmix.pth')

    if epoch - best_epoch == PATIENCE:
        break

print("Fine-tuning complete! Model saved as best_model_cutmix.pth")

