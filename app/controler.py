import os
from clients.train import post_train as client_train
from central.train import post_train as central_train
from analyses.get_confusion_map import get_confusion_map
from analyses.view_acc import get_accuracy
from torchvision import datasets, transforms
from modelNet import netTransform, Net
from torch.utils.data import random_split
from sysvars import SysVars as svar
import pandas as pd
import torch

def post_client_train():
    """
    Simulates local (client-side) training sessions.

    This function calls the `post_train` method from `clients/train.py` twice, 
    using different datasets: one benign and one poisoned. 

    It emulates the independent training of multiple clients in a 
    federated learning environment.

    Returns:
        None
    """
    client_train(epochs=100, load_data=True, data_path=svar.PATH_BASE_DATASET.value)
    client_train(epochs=100, load_data=True, data_path="./datasets/poisoned_data_set_1/")



def post_central_train(selected_indice_models: list = [-1]):
    """
    Simulates the central (server-side) aggregation training process.

    This function loads the weights of client models and performs 
    central training by calling `central/train.py`.

    Args:
        selected_indice_models (list[int], optional): 
            A list of integers representing the indices of the client models 
            to be aggregated.
            - If empty, all models will be used.
            - If [-1], only the most recent model will be used.

    Returns:
        None
    """
    new_model = get_weights(isCentral=False, selected_indice_models=selected_indice_models)
    new_model_dict = new_model[list(new_model.keys())[0]]
    central_train(new_model=new_model_dict)



def get_weights(isCentral = True,selected_indice_models: list = []):
    """
    Loads one or more saved model state_dicts from disk.

    Can load either central or client models depending on the `isCentral` flag.

    Args:
        isCentral (bool, optional): 
            Indicates whether to load models from the central server (True) 
            or from clients (False). Default is True.
        selected_indice_models (list[int], optional): 
            List of model indices to load. 
            - If empty, all models will be loaded.
            - If [-1], only the last model will be loaded.

    Returns:
        dict[str, dict]: 
            A dictionary mapping model filenames to their corresponding 
            PyTorch state_dict objects.

    Raises:
        FileNotFoundError: If the specified model directory does not exist.
        ValueError: If no valid model files are found.
    """
    # Carregar pesos de um ou mais modelos vindos do cliente
    path = svar.PATH_CLIENT_MODELS.value if not isCentral else svar.PATH_CENTRAL_MODELS.value

    if not os.path.exists(svar.PATH_CLIENT_MODELS.value):
        raise FileNotFoundError(f"The specified path {path} does not exist.")
    
    models_path = sorted(os.listdir(path))

    if len(models_path) == 0:
        raise ValueError(f"No valid model files found in {path}.")
    
    models_indices = []
    for model_name in models_path:
        model_idx = model_name.split("_")[-1]
        model_idx = model_idx.split(".")[0]
        models_indices.append(int(model_idx))

    selected_models_dict = {}
    if selected_indice_models == []:
        for model_name in models_path:
            model_path = path + model_name
            state_dict = torch.load(model_path)
            selected_models_dict[model_name] = state_dict

    elif selected_indice_models == [-1]:
        model_path = path + model_path[-1]
        model_static_dict = torch.load(model_path)
        selected_models_dict[model_path[-1]] = model_static_dict
    
    else:
        for sel_model in selected_indice_models:
            if sel_model in models_indices:
                model_name = "model_" + str(sel_model) + ".pt"
                model_path = path + model_name
                state_dict = torch.load(model_path)
                selected_models_dict[model_name] = state_dict

    return selected_models_dict
    


def get_analyses():
    """
    Performs model performance analyses and exports results to CSV files.

    This function evaluates trained models (e.g., accuracy and confusion matrix),
    logs the results, and saves them as CSV files in the analysis directory 
    specified by `SysVars`.

    Additionally, it prints the accuracy of benign and malicious client models
    to the console.

    Returns:
        None
    """
    #if not os.path.exists(svar.PATH_ANALYSES_CVS.value + "analyses.csv"):
    #    train_labels = {
    #        "train_id": [],
    #        "accuracy": [],
    #        "benign_clients": [],
    #        "malignant_clients": [],
    #        "poisoning": []
    #    }
    #    train_table = pd.DataFrame(train_labels)
    
    #else: train_table = pd.read_csv(svar.PATH_ANALYSES_CVS.value + "analyses.csv")

    if not os.path.exists(svar.PATH_ANALYSES_CVS.value + "map.csv"):
        map_labels = {
            "train_id": [],
            "0": [],
            "1": [],
            "2": [],
            "3": [],
            "4": [],
            "5": [],
            "6": [],
            "7": [],
            "8": [],
            "9": []
        }
        map_table = pd.DataFrame(map_labels)

    else: 
        map_table = pd.read_csv(svar.PATH_ANALYSES_CVS.value + "map.csv")

    models = get_weights(isCentral=False, selected_indice_models=[4, 5])

    #get_confusion_map(models["model_4.pt"], "model_4")
    #get_confusion_map(models["model_5.pt"], "model_5")
    acc_belign = get_accuracy(models["model_4.pt"])
    acc_malign = get_accuracy(models["model_5.pt"])

    print("Acc benign client model 4: ", acc_belign)
    print("Acc malign client model 5: ", acc_malign)

    #train_table.to_csv(svar.PATH_ANALYSES_CVS.value + "analyses.csv", index=False, encoding="utf-8")
    #map_table.to_csv(svar.PATH_ANALYSES_CVS.value + "map.csv", index=False, encoding="utf-8")

def get_graphics():
    # Ler dados salvos em tabelas CSV e gerar analises graficas    
    ...



def gradCAM():
    ...



def lime():
    ...



def manipule_data():
    """
    Manipulates the MNIST dataset to create a poisoned version.

    This function:
        - Loads the original MNIST training dataset.
        - Reassigns all label values `7` to `1` (data poisoning step).
        - Splits the dataset into training and testing subsets.
        - Saves the new poisoned dataset to 
          `./datasets/poisoned_data_set_1/MNIST/processed/`.

    Returns:
        None
    """

    mnist_trainset = datasets.MNIST(root=svar.PATH_BASE_DATASET.value, train=True, download=False, transform=netTransform)

    #Change all labels 7 to 1
    for i in range(len(mnist_trainset)):
        t, c = mnist_trainset[i]
        if c == 7:
            mnist_trainset.targets[i] = 1


    total_size = len(mnist_trainset)
    train_size = int(0.7 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(mnist_trainset, [train_size, test_size])

    train_indices = train_dataset.indices
    test_indices = test_dataset.indices
    
    train_data = mnist_trainset.data[train_indices].clone()
    train_targets = mnist_trainset.targets[train_indices].clone()

    test_data = mnist_trainset.data[test_indices].clone()
    test_targets = mnist_trainset.targets[test_indices].clone()

    torch.save((train_data, train_targets), './datasets/poisoned_data_set_1/MNIST/processed/training.pt')
    torch.save((test_data, test_targets), './datasets/poisoned_data_set_1/MNIST/processed/test.pt')