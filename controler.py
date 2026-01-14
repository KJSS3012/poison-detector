import os
from services.trains.train import post_train, control_models_info, merge_models, merge_n_models
from services.trains.view_acc import get_accuracy
# from analyses.get_confusion_map import get_confusion_map
# from xai.lime.lime import analyze_with_lime
# from analyses.view_acc import get_accuracy
from services.xai.gradcam.gradcam import mean_gradCAM, mean_plusplusCAM
from services.xai.gradcam.scorecam import mean_scorecam
from services.xai.gradcam.utils import save_sad_mask
from services.xai.distance_algorithms.sad import sad
from services.xai.distance_algorithms.ssim import ssim_algorithm as ssim
from services.xai.distance_algorithms.com import com_algorithm as com
from services.xai.distance_algorithms.was import wassertein_algorithm as was
from services.xai.treshold_algorithms.iqr import interquartile_range_treshold as iqr
from torchvision import datasets, transforms
from torchvision.utils import save_image
from services.trains.modelNet import netTransform, Net
from sysvars import SysVars as svar
import pandas as pd
import matplotlib.pyplot as plt
import torch
import cv2
import numpy as np
from services.datasets.manipule_data import save_mnist_examples, create_dataset, create_poisoned_dataset_x_to_y, create_randomized_dataset
from services.datasets.load_data import PTDataset
from services.trains.train import post_train as convergence_train

def experiment_001():

    epochs = 40
    central = post_train(save_model = True, epochs = epochs, model_path = svar.EXPERIMENT_ROOT / "models" / "central_model.pt")
    control_models_info(model_name="central_model.pt", epochs=epochs, poison="None", dataset_name="mnist", dataset_interval="0.0 - 1.0", isCentral=True)

    model1 = post_train(save_model = True, epochs = epochs, model_path = svar.EXPERIMENT_ROOT / "models" / "belign_1.pt")
    control_models_info(model_name="belign_1.pt", epochs=epochs, poison="None", dataset_name="mnist", dataset_interval="0.0 - 1.0", isCentral=False)
    
    model2 = post_train(save_model = True, epochs = epochs, model_path = svar.EXPERIMENT_ROOT / "models" / "belign_2.pt")
    control_models_info(model_name="belign_2.pt", epochs=epochs, poison="None", dataset_name="mnist", dataset_interval="0.0 - 1.0", isCentral=False)
    
    model3 = post_train(save_model = True, epochs = epochs, model_path = svar.EXPERIMENT_ROOT / "models" / "belign_3.pt")
    control_models_info(model_name="belign_3.pt", epochs=epochs, poison="None", dataset_name="mnist", dataset_interval="0.0 - 1.0", isCentral=False)
    
    model4 = post_train(save_model = True, epochs = epochs, model_path = svar.EXPERIMENT_ROOT / "models" / "belign_4.pt")
    control_models_info(model_name="belign_4.pt", epochs=epochs, poison="None", dataset_name="mnist", dataset_interval="0.0 - 1.0", isCentral=False)
    
    model5 = post_train(save_model = True, epochs = epochs, model_path = svar.EXPERIMENT_ROOT / "models" / "belign_5.pt")
    control_models_info(model_name="belign_5.pt", epochs=epochs, poison="None", dataset_name="mnist", dataset_interval="0.0 - 1.0", isCentral=False)
    
    model6 = post_train(save_model = True, epochs = epochs, model_path = svar.EXPERIMENT_ROOT / "models" / "belign_6.pt")
    control_models_info(model_name="belign_6.pt", epochs=epochs, poison="None", dataset_name="mnist", dataset_interval="0.0 - 1.0", isCentral=False)
    
    model7 = post_train(save_model = True, epochs = epochs, model_path = svar.EXPERIMENT_ROOT / "models" / "malign_1.pt", data_path = svar.EXPERIMENT_DATASETS / "poisoned_data_set_1")
    control_models_info(model_name="malign_1.pt", epochs=epochs, poison="7 > 1", dataset_name="poisoned_data_set_1", dataset_interval="0.0 - 1.0", isCentral=False)

    model8 = post_train(save_model = True, epochs = epochs, model_path = svar.EXPERIMENT_ROOT / "models" / "malign_2.pt", data_path = svar.EXPERIMENT_DATASETS / "poisoned_data_set_2")
    control_models_info(model_name="malign_2.pt", epochs=epochs, poison="3 > 9", dataset_name="poisoned_data_set_2", dataset_interval="0.0 - 1.0", isCentral=False)
    
    model9 = post_train(save_model = True, epochs = epochs, model_path = svar.EXPERIMENT_ROOT / "models" / "malign_3.pt", data_path = svar.EXPERIMENT_DATASETS / "poisoned_data_set_3")
    control_models_info(model_name="malign_3.pt", epochs=epochs, poison="8 > 2", dataset_name="poisoned_data_set_3", dataset_interval="0.0 - 1.0", isCentral=False)
    
    model10 = post_train(save_model = True, epochs = epochs, model_path = svar.EXPERIMENT_ROOT / "models" / "malign_4.pt", data_path = svar.EXPERIMENT_DATASETS / "poisoned_data_set_4")
    control_models_info(model_name="malign_4.pt", epochs=epochs, poison="6 > 4", dataset_name="poisoned_data_set_4", dataset_interval="0.0 - 1.0", isCentral=False)
    
    clients = [model1, model2, model3, model4, model5, model6, model7, model8, model9, model10]

    i = 1
    scores = []
    for client in clients:
        print("="*40)
        print(f"XAI Analyses for Client {i}")
        scores.append(xai_analyses(central, client))
        i += 1

       # Nomes para identificar cada combinação de score
    xai_names = ["plusplusCAM", "gradCAM", "scorecam"]
    class_index_names = ["class_true", "class_false"]
    distance_names = ["sad", "ssim", "com", "was"]
    
    score_names = []
    for xai in xai_names:
        for ci in class_index_names:
            for dist in distance_names:
                score_names.append(f"{xai}_{ci}_{dist}")

    # ...existing code de XAI analyses...

    # SEM FILTRO
    sf_central = dict(central)
    sf_central = merge_n_models([sf_central] + [dict(map) for map in clients])
    
    print("\n" + "="*60)
    print("ACURÁCIA SEM FILTRO (todos os clientes agregados):")
    sf_accuracy = get_accuracy(sf_central)
    print("="*60)

    # COM FILTRO - armazena resultados para comparação
    results = []
    
    for i in range(len(scores[0])):
        cf_central = dict(central)
        scores_data = []
        for j in range(len(scores)):
            scores_data.append(scores[j][i])

        treshold = iqr(scores_data)
        
        # Verifica se o threshold é válido (não é NaN ou infinito)
        if np.isnan(treshold) or np.isinf(treshold):
            print("\n" + "="*60)
            print(f"SCORE TYPE: {score_names[i]}")
            print(f"Threshold inválido (NaN ou Inf): {treshold}")
            print("Acurácia definida como 0")
            results.append({
                "score_type": score_names[i],
                "threshold": treshold,
                "clients_included": [],
                "accuracy": 0.0
            })
            print("="*60)
            continue

        # Identifica quais clientes passaram no filtro
        filtered_clients = []
        filtered_dicts = []
        for j in range(len(clients)):
            if scores[j][i] < treshold:
                filtered_dicts.append(dict(clients[j]))
                filtered_clients.append(j + 1)  # +1 para índice legível
        
        cf_central = merge_n_models([dict(central)] + filtered_dicts)
        
        print("\n" + "="*60)
        print(f"SCORE TYPE: {score_names[i]}")
        print(f"Threshold: {treshold:.4f}")
        print(f"Clientes incluídos: {filtered_clients}")
        print(f"Clientes filtrados: {[j+1 for j in range(len(clients)) if j+1 not in filtered_clients]}")
        
        accuracy = get_accuracy(cf_central)
        results.append({
            "score_type": score_names[i],
            "threshold": treshold,
            "clients_included": filtered_clients,
            "accuracy": accuracy
        })
        print("="*60)

    # Salvar resultados em CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv(svar.EXPERIMENT_CSV / "accuracy_by_score_type.csv", index=False)
    
    # Gráfico comparativo
    plt.figure(figsize=(14, 6))
    plt.bar(range(len(results)), [r["accuracy"] for r in results])
    plt.axhline(y=sf_accuracy, color='r', linestyle='--', label=f'Sem Filtro: {sf_accuracy:.2%}')
    plt.xticks(range(len(results)), [r["score_type"] for r in results], rotation=90)
    plt.xlabel("Tipo de Score")
    plt.ylabel("Acurácia")
    plt.title("Acurácia do Modelo Central por Tipo de Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(svar.EXPERIMENT_GRAPHICS / "accuracy_comparison.png", dpi=300)
    plt.close()
    
    print("\n" + "="*60)
    print("RESUMO DOS RESULTADOS:")
    print(f"Acurácia sem filtro: {sf_accuracy:.2%}")
    print(f"Melhor score: {max(results, key=lambda x: x['accuracy'])}")
    print("="*60)

        


def xai_analyses(central_model, client_model):

    xai_algorithms = [
        mean_plusplusCAM,
        mean_gradCAM,
        mean_scorecam
    ]

    distance_algorithms = [
        sad,
        ssim,
        com,
        was
    ]
    
    scores = []
    for xai_algo in xai_algorithms:
        for class_index in [True, False]:
            central_mask = xai_algo(central_model, save_path = "", class_index=class_index)
            client_mask = xai_algo(client_model, save_path = "", class_index=class_index)
            for dis_algo in distance_algorithms:
                sum = 0
                for i in range(10):
                    sum += dis_algo(central_mask[i], client_mask[i])
                scores.append(sum/10)
    
    return scores

def post_client_train():
    pass

def get_diffs(static_output_path):
    """
    Simulates local (client-side) training sessions.

    This function calls the `post_train` method from `clients/train.py` twice, 
    using different datasets: one benign and one poisoned. 

    It emulates the independent training of multiple clients in a 
    federated learning environment.

    Returns:
        None
    """

    central = get_weights(isCentral=False, selected_indice_models=[6])
    central_masks = mean_gradCAM(models=central, save_path="./teste/central_")

    #COM O PATHLIB VOCÊ PODE PASSAR ASSIM: svar.BASEROOTEXEMPLO / "algumaCOisa"
    # create_dataset(0.0, 0.3, "./datasets/emnist_data_set_30/", "./datasets/emnist_data_set/")
    # create_poisoned_dataset_x_to_y(1, 7, "./datasets/emnist_poisoned_17/", "./datasets/emnist_data_set_30/")
    # create_poisoned_dataset_x_to_y(7, 1, "./datasets/emnist_poisoned_71/", "./datasets/emnist_data_set_30/")
    # create_poisoned_dataset_x_to_y(7, 1, "./datasets/emnist_poisoned_71/", "./datasets/emnist_data_set_30/")
    # create_poisoned_dataset_x_to_y(5, 6, "./datasets/emnist_poisoned_56/", "./datasets/emnist_data_set_30/")


    # client_train(epochs=30, dataset_interval="0.0 - 0.3", data_path="./datasets/emnist_data_set_30/")
    # client_train(epochs=30, dataset_interval="0.0 - 0.3", data_path="./datasets/emnist_data_set_30/")
    # client_train(epochs=30, dataset_interval="0.0 - 0.3", data_path="./datasets/emnist_data_set_30/")
    # client_train(epochs=30, dataset_interval="0.0 - 0.3", data_path="./datasets/emnist_data_set_30/")
    # client_train(epochs=30, dataset_interval="0.0 - 0.3", data_path="./datasets/emnist_poisoned_17/", poison="1 to 7")
    # client_train(epochs=30, dataset_interval="0.0 - 0.3", data_path="./datasets/emnist_poisoned_71/", poison="7 to 1")
    # client_train(epochs=30, dataset_interval="0.0 - 0.3", data_path="./datasets/emnist_poisoned_71/", poison="7 to 1")
    # client_train(epochs=30, dataset_interval="0.0 - 0.3", data_path="./datasets/emnist_poisoned_56/", poison="5 to 6")

    sums_arr = []

    for i in range(8, 18):
        print(f"Calculating distances for model {i}...")
        
        client = get_weights(isCentral=False, selected_indice_models=[i])
        client_masks = mean_gradCAM(models=client, save_path=f"./teste/{i}_")

        sum = 0
        for j in range(10):
            # sum += ssim(central_masks[j], client_masks[j])
            sum += com(central_masks[j], client_masks[j])

        print(f"\nDistance scores for central model {i}:\n")
        print(sum/10)
        sums_arr.append(sum/10)

    models = [f"model_{i}" for i in range(8, 18)]

    df = pd.DataFrame({
        "model": models,
        "valor": sums_arr
    })
    df.to_csv("client_to_central_scores.csv", index=False)

    plt.figure()
    plt.plot(models, sums_arr, marker='o')
    plt.xlabel("Modelo")
    plt.ylabel("Valor")
    plt.title("Resultados por modelo")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(static_output_path, dpi=300)
    plt.close() 




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

    pass



def get_weights(path, isCentral = True,selected_indice_models: list = []):
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
    # path = svar.PATH_CLIENT_MODELS if not isCentral else svar.PATH_CENTRAL_MODELS

    if not os.path.exists(svar.PATH_CLIENT_MODELS):
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
    pass


def lime():
    
    test_dataset = PTDataset(pt_file = svar.EMNIST_TEST_PATH)
    client_1 = get_weights(selected_indice_models=[4], isCentral=False)

    model = Net().to(svar.DEFAULT_DEVICE)
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
          `./datasets/poisoned_data_set_1/`.

    Returns:
        None
    """

    # save_mnist_examples()
    create_poisoned_dataset_x_to_y(8, 2, path = svar.EXPERIMENT_DATASETS / "poisoned_data_set_3")
    create_poisoned_dataset_x_to_y(6, 4, path = svar.EXPERIMENT_DATASETS / "poisoned_data_set_4")
    

    with open(svar.EXPERIMENT_DATASETS / "datasets_info.json", "r") as f:
        import json
        datasets_info = json.load(f)
    datasets_info = []
    datasets_info.append({
        "dataset_name": "poisoned_data_set_3",
        "poison": "8 > 2",
        "base": "MNIST",
        "dataset_interval": "0.0 - 1.0"
    })

    datasets_info.append({
        "dataset_name": "poisoned_data_set_4",
        "poison": "6 > 4",
        "base": "MNIST",
        "dataset_interval": "0.0 - 1.0"
    })

    with open(svar.EXPERIMENT_DATASETS / "datasets_info.json", "w") as f:
        import json
        json.dump(datasets_info, f, indent=4)

    
    # create_dataset(0.0, 1.0)

    # train = datasets.EMNIST(root="data", split="digits", train=True, download=True)
    # test  = datasets.EMNIST(root="data", split="digits", train=False, download=True)

    # train_data  = train.data.clone()
    # train_labels = train.targets.clone()

    # test_data = test.data.clone()
    # test_labels = test.targets.clone()

    # torch.save((train_data, train_labels), "./datasets/emnist_data_set/training.pt")
    # torch.save((test_data, test_labels), "./datasets/emnist_data_set/test.pt")