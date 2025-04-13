import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from utils.CustomDataset import BaselineDataset, PrecomputedDataset, CenterAwareDataset
from models.general_classifier import BinaryClassifier
from models.adversarial import AdversarialClassifier
from peft import LoraConfig, get_peft_model
from transformers import AutoModel
import numpy as np
import pandas as pd
import argparse

SEED = 0

torch.random.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Working on {device}.")

test_path = "../data/test.h5"
batch_size = 32

parser = argparse.ArgumentParser()
parser.add_argument("--test_param", type=str, default="baseline")
args = parser.parse_args()
    
transform = transforms.Compose([
    transforms.Resize((98, 98))
])
if args.test_param == "baseline":
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

    print('Loading model...')
    feature_extractor = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
    feature_extractor.eval()

    model = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.Dropout(0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        ).to(device)
    
    model.load_state_dict(torch.load("../outputs/best_model.pth"))
    
    TEST_IMAGES_PATH = test_data_path
    with h5py.File(TEST_IMAGES_PATH, 'r') as hdf:
        test_ids = list(hdf.keys())

    solutions_data = {'ID': [], 'Pred': []}
    with h5py.File(TEST_IMAGES_PATH, 'r') as hdf:
        for test_id in tqdm(test_ids):
            img = transform(torch.tensor(np.array(hdf.get(test_id).get('img'))).unsqueeze(0).float())
            pred = model.fc(model.feature_extractor(img.to(device)).last_hidden_state[:, 0, :]).detach().cpu()
            solutions_data['ID'].append(int(test_id))
            solutions_data['Pred'].append(int(pred.item() > 0.5))
    solutions_data = pd.DataFrame(solutions_data).set_index('ID')
    solutions_data.to_csv('baseline_test.csv')
    

else:
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
    
    if args.test_param == "fine-tuned_dino":
        model = BinaryClassifier(dino_lora).to(device)
        model.load_state_dict(torch.load("../outputs/best_model_finetuned.pth"))
    
    if args.test_param == "fine-tuned_dino_adversarial":
        model = AdversarialClassifier(dino_lora).to(device)
        model.load_state_dict(torch.load("../outputs/dino_lora_adversarial.pth"))
    
    
    TEST_IMAGES_PATH = test_data_path
    with h5py.File(TEST_IMAGES_PATH, 'r') as hdf:
        test_ids = list(hdf.keys())

    solutions_data = {'ID': [], 'Pred': []}
    with h5py.File(TEST_IMAGES_PATH, 'r') as hdf:
        for test_id in tqdm(test_ids):
            img = transform(torch.tensor(np.array(hdf.get(test_id).get('img'))).unsqueeze(0).float())
            pred = model.fc((img.to(device))).detach().cpu()
            solutions_data['ID'].append(int(test_id))
            solutions_data['Pred'].append(int(pred.item() > 0.5))
    solutions_data = pd.DataFrame(solutions_data).set_index('ID')
    solutions_data.to_csv('baseline_test.csv')


