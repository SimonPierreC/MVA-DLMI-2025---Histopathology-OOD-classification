import subprocess
import os
import sys

def main():

    models = {
    "BaseLine": "train_baseline",
    "CutMix": "train_cutmix",
    "Fine-Tuned Dino": "train_finetuned_dino",
    "Adversarial DINO (0.1)": "train_adv_dino",
    "Adversarial DINO (lambda_p)": "train_adv_dino_lambda",
    "Histogram Adaptation": "train_histogram"
    }
    print("Select a model to train:")
    for i, name in enumerate(models.keys(), 1):
        print(f"{i}. {name}")

    choice = int(input("Enter the number of the model you want to train: "))
    if choice < 1 or choice > len(models):
        print("Invalid choice.")
        return
    
    venv_python = "../dlmi/Scripts/python.exe"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.path.abspath(os.path.dirname(__file__))

    selected_model = list(models.items())[choice - 1][1]
       
    if selected_model == "train_adv_dino_lambda":
        subprocess.run(
        [venv_python, f"training/train_adv_dino.py", "--lbda", "0.0"],
        env=env)
        test_param = 'adversarial'
        
    elif selected_model == "train_histogram":
        subprocess.run(
            [venv_python, f"training/train_finetuned_dino.py", "--histogram", "True"],
            env=env
        )
        test_param = 'histogram'
    else:
        subprocess.run(
            [venv_python, f"training/{selected_model}.py"],
            env=env
        )
    
    if selected_model == "train_adv_dino":
        test_param = 'adversarial'
    elif selected_model == "fine-tuned_dino":
        test_param = 'finetuned'
    elif test_param == "cutmix":
        test_param = 'cutmix'
    
    subprocess.run(
        [venv_python, f"testing/testing_model.py", "--test_param", test_param],
        env=env
    )

if __name__ == "__main__":
    main()