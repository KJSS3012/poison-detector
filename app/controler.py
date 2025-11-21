import os
from clients.train import post_train as client_train
from central.train import post_train as central_train
from analyses.get_confusion_map import get_confusion_map
from xai.gradcam.gradcam import gradcam as generate_gradcam
from xai.gradcam.utils import save_cam_mask
from xai.gradcam.distancetest import get_distance_scores
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
    client_train(epochs=100, load_data=True, data_path=svar.PATH_BASE_DATASET.value)
    client_train(epochs=100, load_data=True, data_path="./datasets/poisoned_data_set_1/")



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

    mean_mask_central = gradCAM(selected_indice_models=[1])

    mean_mask_client_1 = gradCAM(selected_indice_models=[4], isCentral=False)
    mean_mask_client_2 = gradCAM(selected_indice_models=[6], isCentral=False)
    mean_mask_client_3 = gradCAM(selected_indice_models=[7], isCentral=False)
    mean_mask_poisoned_1 = gradCAM(selected_indice_models=[5], isCentral=False)
    mean_mask_poisoned_2 = gradCAM(selected_indice_models=[8], isCentral=False)


    score_client_1 = get_distance_scores([mean_mask_client_1], ref=mean_mask_central)
    score_client_2 = get_distance_scores([mean_mask_client_2], ref=mean_mask_central)
    score_client_3 = get_distance_scores([mean_mask_client_3], ref=mean_mask_central)
    score_poisoned_1 = get_distance_scores([mean_mask_poisoned_1], ref=mean_mask_central)
    score_poisoned_2 = get_distance_scores([mean_mask_poisoned_2], ref=mean_mask_central)


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
    gradCAM(selected_indice_models=[1])
    print("Acc benign central model 1: ", acc_belign)

    #train_table.to_csv(svar.PATH_ANALYSES_CVS.value + "analyses.csv", index=False, encoding="utf-8")
    #map_table.to_csv(svar.PATH_ANALYSES_CVS.value + "map.csv", index=False, encoding="utf-8")



def gradCAM(selected_indice_models: list = [-1], isCentral: bool = True):

    models = get_weights(isCentral=isCentral, selected_indice_models=selected_indice_models)

    cams = {}

    samples = os.listdir("./datasets/sample_images/")
    numbers = {
        0:[],
        1:[],
        2:[],
        3:[],
        4:[],
        5:[],
        6:[],
        7:[],
        8:[],
        9:[]
    }

    for f in samples:
        num = f.split(".")[0]
        num = f.split("_")
        num = int(num[1])
        numbers[num].append(f)


    for model_name in models.keys():
        model = model_name


    j = 1
    path = './analyses/gradcams/cams_means/' + model.split(".")[0] + f'_{j}/'
    while os.path.exists(path):

        j += 1
        path = './analyses/gradcams/cams_means/' + model.split(".")[0] + f'_{j}/'
    
    os.makedirs(path)

    for num in numbers.keys():

        model = ''
        cams = {}
        cams_to_stack = []

        for f in numbers[num]:

            cams[f] = []
            for i in range(10):

                for model_name, model_dict in models.items():

                    cams[f].append(generate_gradcam(
                        img_path = f"./datasets/sample_images/{f}",
                        model_dict = model_dict,
                        model_name = f"number_{num}_belign",
                        class_index = None,
                        save = True
                    ))

                    #save_cam_mask(cams[f][i].detach().cpu().numpy(), f'./analyses/gradcams/cams_means/solid_cams/{num}_{i}.png')


                    cam = cams[f][i].detach().cpu().numpy().squeeze()
                    cam_min, cam_max = np.min(cam), np.max(cam)


                    cam = (cam - cam_min) / (cam_max - cam_min) if cam_max > cam_min else np.zeros_like(cam)

                    cam = cv2.resize(cam, (28, 28)) if cam.shape != (28, 28) else cam


                    cams_to_stack.append(torch.from_numpy(cam))



        stack = torch.stack(cams_to_stack)
        mean_cam = torch.mean(stack, axis=0)
        mean_cam = mean_cam.detach().cpu().numpy()

        return mean_cam
        #save_cam_mask(mean_cam, f'{path}mean_cam_{num}.png')
#
#
        #for f, masks in cams.items():
        #    print(f"\n\nDistance scores for image {f}:")
        #    get_distance_scores(masks)

            


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

    # Change all labels 7 to 1
    #for i in range(len(mnist_trainset)):
    #    t, c = mnist_trainset[i]
    #    if c == 7:
    #        mnist_trainset.targets[i] = 1

    numbers = {
        0:0,
        1:0,
        2:0,
        3:0,
        4:0,
        5:0,
        6:0,
        7:0,
        8:0,
        9:0
    }
    for i in range(len(mnist_trainset)):
        t, c = mnist_trainset[i]
        if numbers[c] < 5:
            numbers[c] += 1
            save_image(t, f'./datasets/sample_images/sample_{c}_{numbers[c]}.png')


    #total_size = len(mnist_trainset)
    #train_size = int(0.7 * total_size)
    #test_size = total_size - train_size
    #train_dataset, test_dataset = random_split(mnist_trainset, [train_size, test_size])

    #train_indices = train_dataset.indices
    #test_indices = test_dataset.indices
    
    #train_data = mnist_trainset.data[train_indices].clone()
    #train_targets = mnist_trainset.targets[train_indices].clone()

    #test_data = mnist_trainset.data[test_indices].clone()
    #test_targets = mnist_trainset.targets[test_indices].clone()

    #torch.save((train_data, train_targets), './datasets/poisoned_data_set_1/MNIST/processed/training.pt')
    #torch.save((test_data, test_targets), './datasets/poisoned_data_set_1/MNIST/processed/test.pt')