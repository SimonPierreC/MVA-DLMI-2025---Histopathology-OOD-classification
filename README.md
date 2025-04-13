# MVA-DLMI-2025---Histopathology-OOD-classification

## Installation
1. Aside the repository, create a virtual environnement
   ```bash
   python -m venv ./dlmi
3. Install dependencies:
   ```bash
   source /dlim/bin/python.exe
   pip install -r requirements.txt
   ```
   For windows, change the firt line :
   ```bash
   source /dlmi/Scripts/activate
   ```
5. Ensure all required libraries are available.

# Datasets

The dataset has to be of the following shape :

<pre> data/
        -train.h5
        -val.h5
        -test.h5
</pre>

You can change this setup in the configs/\*.yaml file

## Running and testing the Models

1. ```bash
    cd ~/MVA-DLMI-2025---Histopathology-OOD-classification
    py main.py
   ```
2. select in the commande line you the model you want to run and test (see report for more details):
<pre>Select a model to train:
3. BaseLine
4. CutMix
5. Fine-Tuned Dino
6. Adversarial DINO (0.1)
7. Adversarial DINO (lambda_p)
8. Histogram Adaptation
</pre>

## Model Files and Outputs

• The models follow the configurations as mentionned in the report
• Pretrained models are in the outputs directory.
• Test files for submissions are in the outputs directory.
