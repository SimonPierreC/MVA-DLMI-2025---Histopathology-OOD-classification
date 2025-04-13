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

test_data_path = "../data_test/test.h5"
batch_size = 32

parser = argparse.ArgumentParser()
parser.add_argument("--test_param", type=str, default="baseline")
args = parser.parse_args()
    
transform = transforms.Compose([
    transforms.Resize((98, 98))
])

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

if args.test_param == "finetuned" or args.test_param == "cutmix":
    model = BinaryClassifier(dino_lora).to(device)
    if args.test_param == "cutmix":
        model = BinaryClassifier(dino_lora, 2).to(device)
    model.load_state_dict(torch.load("./outputs/best_model_finetuned.pth"))

if args.test_param == "adversarial":
    model = AdversarialClassifier(dino_lora, 384, 2, 3).to(device)
    model.load_state_dict(torch.load("./outputs/dino_lora_adversarial.pth"))

if args.test_param == "histogram":
    mean_hists_list_test = []
    with h5py.File(test_data_path, "r") as f:
        for img_id in f.keys():
            mean_hists_list_test.append(compute_average_histogram(f[img_id]['img']))
        
    mean_hists_test = np.mean(np.array(mean_hists_list_val), axis = 0)
    transform_train = transforms.Compose([
        transforms.Resize((98, 98)),
        transforms.Lambda(lambda x: compute_average_histogram(x, mean_hists_test))
    ]) 
        
    model = BinaryClassifier(dino_lora).to(device)
    model.load_state_dict(torch.load("../outputs/best_model_finetuned.pth"))
        
    
TEST_IMAGES_PATH = test_data_path
with h5py.File(TEST_IMAGES_PATH, 'r') as hdf:
    test_ids = list(hdf.keys())

solutions_data = {'ID': [], 'Pred': []}
with h5py.File(TEST_IMAGES_PATH, 'r') as hdf:
    model.eval()
    for test_id in test_ids:
        # Have to do a forward differently in function of the model
        img = transform(torch.tensor(np.array(hdf.get(test_id).get('img'))).unsqueeze(0).float().to(device))
        if args.test_param == "histogram" or args.test_param == "finetuned" or args.test_param == "cutmix":
            if args.test_param == 'cutmix':
                pred = torch.nn.Softmax(dim=2)(model(img)).detach().cpu()[1]
            else:
                pred = torch.nn.Sigmoid()(model(img)).detach().cpu()
        elif args.test_param == "adversarial":
            pred, = model(img)
            pred = torch.nn.Sigmoid()(pred).detach().cpu()
        solutions_data['ID'].append(int(test_id))
        solutions_data['Pred'].append(int(pred.item() > 0.5))
solutions_data = pd.DataFrame(solutions_data).set_index('ID')
solutions_data.to_csv(f'./outputs/test_{args.test_param}.csv')


