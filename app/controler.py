import os
from clients.train import post_train as client_train
from central.train import post_train as central_train
from analyses.get_confusion_map import get_confusion_map
from xai.gradcam.gradcam import mean_gradCAM
from xai.distance_algorithms.sad import sad
from xai.lime.lime import analyze_with_lime
from analyses.view_acc import get_accuracy
from torchvision import datasets, transforms
from torchvision.utils import save_image
from modelNet import netTransform, Net
from torch.utils.data import random_split
from sysvars import SysVars as svar
import pandas as pd
import torch
import cv2
import numpy as np
from datasets.manipule_data import save_mnist_examples, create_dataset, create_poisoned_dataset_x_to_y

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

    models_central = get_weights(isCentral=False, selected_indice_models=[4])
    mean_central = mean_gradCAM(models=models_central)

    # models_4 = get_weights(isCentral=False, selected_indice_models=[4])
    # mean_cam4 = mean_gradCAM(models=models_4)
    
    # models_5 = get_weights(isCentral=False, selected_indice_models=[5])
    # mean_cam5 = mean_gradCAM(models=models_5)
    
    # models_6 = get_weights(isCentral=False, selected_indice_models=[6])
    # mean_cam6 = mean_gradCAM(models=models_6)
    
    # models_7 = get_weights(isCentral=False, selected_indice_models=[7])
    # mean_cam7 = mean_gradCAM(models=models_7)
    
    # models_8 = get_weights(isCentral=False, selected_indice_models=[8])
    # mean_cam8 = mean_gradCAM(models=models_8)


    client_train(epochs=20, load_data=True, data_path=svar.PATH_BASE_DATASET.value)
    mean_cam4 = mean_gradCAM(models=get_weights(isCentral=False, selected_indice_models=[1]))

    # # create_dataset(0.3, 0.7, path="./datasets/belign_data_set_2/")
    client_train(epochs=20, load_data=True, data_path="./datasets/belign_data_set_2/")
    mean_cam6 = mean_gradCAM(models=get_weights(isCentral=False, selected_indice_models=[2]))

    # create_dataset(0.7, 1.0, path="./datasets/belign_data_set_3/")
    client_train(epochs=20, load_data=True, data_path="./datasets/belign_data_set_3/")
    mean_cam7 = mean_gradCAM(models=get_weights(isCentral=False, selected_indice_models=[3]))
    #ERRO AQUI

    client_train(epochs=20, load_data=True, data_path="./datasets/poisoned_data_set_1/")
    mean_cam5 = mean_gradCAM(models=get_weights(isCentral=False, selected_indice_models=[9]))


    # create_poisoned_dataset_x_to_y(6, 9, path="./datasets/poisoned_data_set_2/")
    client_train(epochs=20, load_data=True, data_path="./datasets/poisoned_data_set_2/")
    mean_cam8 = mean_gradCAM(models=get_weights(isCentral=False, selected_indice_models=[10]))

    abs_scores4 = []
    abs_scores5 = []
    abs_scores6 = []
    abs_scores7 = []
    abs_scores8 = []

    # for i in range(min(len(mean_central), len(mean_cam4), len(mean_cam5), 
    #                   len(mean_cam6), len(mean_cam7), len(mean_cam8))):
    for i in range(10):
        print(f"Calculating distances for sample {i}...")
        # print(mean_central)
        central_cam = mean_central[i]
        cam4 = mean_cam4[i]
        cam5 = mean_cam5[i] 
        cam6 = mean_cam6[i]
        cam7 = mean_cam7[i]
        cam8 = mean_cam8[i]
        # mean_cam4 = mean_cam4[i].cpu()
        # mean_cam5 = mean_cam5[i].cpu()
        # mean_cam6 = mean_cam6[i].cpu()
        # mean_cam7 = mean_cam7[i].cpu()
        # mean_cam8 = mean_cam8[i].cpu()

        abs_scores4.append(sad(central_cam, cam4)[1])
        abs_scores5.append(sad(central_cam, cam5)[1])
        abs_scores6.append(sad(central_cam, cam6)[1])
        abs_scores7.append(sad(central_cam, cam7)[1])
        abs_scores8.append(sad(central_cam, cam8)[1])


    print("\nDistance scores for client_4:\n")
    sum4 = 0
    for e in abs_scores4: sum4 += e
    print(sum4/10)

    print("\nDistance scores for client_5:\n")
    sum5 = 0
    for e in abs_scores5: sum5 += e
    print(sum5/10)
    
    print("\nDistance scores for client_6:\n")
    sum6 = 0
    for e in abs_scores6: sum6 += e
    print(sum6/10)

    print("\nDistance scores for client_7:\n")
    sum7 = 0
    for e in abs_scores7: sum7 += e
    print(sum7/10)

    print("\nDistance scores for client_8:\n")
    sum8 = 0
    for e in abs_scores8: sum8 += e
    print(sum8/10)
    # client_train(epochs=100, load_data=True, data_path=svar.PATH_BASE_DATASET.value)
    # client_train(epochs=100, load_data=True, data_path=svar.PATH_BASE_DATASET.value)
    # client_train(epochs=100, load_data=True, data_path="./datasets/poisoned_data_set_1/")



def post_central_train():
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

    mean_mask_central = mean_gradCAM(selected_indice_models=[1])

    mean_mask_client_1 = mean_gradCAM(selected_indice_models=[4])
    mean_mask_client_2 = mean_gradCAM(selected_indice_models=[6], isCentral=False)
    mean_mask_client_3 = mean_gradCAM(selected_indice_models=[7], isCentral=False)
    mean_mask_poisoned_1 = mean_gradCAM(selected_indice_models=[5], isCentral=False)
    mean_mask_poisoned_2 = mean_gradCAM(selected_indice_models=[8], isCentral=False)


    # score_client_1 = get_distance_scores([mean_mask_client_1], ref=mean_mask_central)
    # score_client_2 = get_distance_scores([mean_mask_client_2], ref=mean_mask_central)
    # score_client_3 = get_distance_scores([mean_mask_client_3], ref=mean_mask_central)
    # score_poisoned_1 = get_distance_scores([mean_mask_poisoned_1], ref=mean_mask_central)
    # score_poisoned_2 = get_distance_scores([mean_mask_poisoned_2], ref=mean_mask_central)


    print("\nDistance scores for benign client model 1:", score_client_1)
    print("\nDistance scores for benign client model 2:", score_client_2)
    print("\nDistance scores for benign client model 3:", score_client_3)
    print("\nDistance scores for malign client model 1:", score_poisoned_1)
    print("\nDistance scores for malign client model 2:", score_poisoned_2)


    #for index in selected_indice_models:
    #    new_model = get_weights(isCentral=False, selected_indice_models=[index])
    #    new_model_dict = new_model[list(new_model.keys())[0]]
    #    central_train(new_model=new_model_dict)



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

    models = get_weights(isCentral=True, selected_indice_models=[1])

    #get_confusion_map(models["model_5.pt"], "model_5")
    get_confusion_map(models["model_1.pt"], "central_model_1")
    acc_belign = get_accuracy(models["model_1.pt"])
    mean_gradCAM(selected_indice_models=[1])
    print("Acc benign central model 1: ", acc_belign)

    #train_table.to_csv(svar.PATH_ANALYSES_CVS.value + "analyses.csv", index=False, encoding="utf-8")
    #map_table.to_csv(svar.PATH_ANALYSES_CVS.value + "map.csv", index=False, encoding="utf-8")
   


def lime():
    
    test_dataset = datasets.MNIST(root=svar.PATH_BASE_DATASET.value, train=False, download=False, transform=netTransform)
    client_1 = get_weights(selected_indice_models=[4], isCentral=False)

    model = Net().to(svar.DEFAULT_DEVICE.value)
    model.load_state_dict(client_1["model_4.pt"])

    results = analyze_with_lime(model, test_dataset, num_samples=50)

        #    lime_results.append({
        #        'index': idx,
        #        'true_label': true_label,
        #        'explanation': explanation,
        #        'mask': mask,
        #        'image': image_np
        #    })


    def show_ascii_image(img):
        img = img.mean(axis=2)
        chars = " .:-=+*#%@"
        img_norm = (img - img.min()) / (img.max() - img.min())
        
        for row in img_norm:
            line = "".join(chars[int(val * (len(chars) - 1))] for val in row)
            print(line)

    for e in results:

        weights = e['explanation'].top_labels[0]
        weights = e['explanation'].local_exp[weights]

        print('\n\n\n\n==================')
        print(f"True label: {e['true_label']}")
        print(f"Explanation: {weights}")
        show_ascii_image(e['temp'])
        print(f"Mask: {e['mask']}")
        print(f"Image: {e['image']}")




def data_manipulations():
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

    save_mnist_examples()
    create_poisoned_dataset_x_to_y(7, 1)
    create_dataset(0.0, 1.0)