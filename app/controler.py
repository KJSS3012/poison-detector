import os
from clients.train import post_train as client_train
from central.train import post_train as central_train
from torchvision import datasets, transforms
from modelNet import netTransform
from torch.utils.data import random_split
from syspaths import SysPaths as spath
import pandas as pd
import torch



def post_client_train():
    """
    Method to simulate a client training session.
    It's call the post_train method from clients/train.py with some default parameters.
    """
    client_train(epochs=5, use_cuda=True, load_data=True, data_path=spath.PATH_BASE_DATASET)



def post_central_train(selected_indice_models: list = [-1]):
    """
    Method to simulate a central training session.
    It's call the post_train method from central/train.py with some default parameters.

    args:
    - selected_indice_models: list of integers with the indices of the client models to be used in the training. If empty, all models will be used. If [-1], only the last model will be used.
    """
    new_model = get_weights(isCentral=False, selected_indice_models=selected_indice_models)
    new_model_dict = new_model[list(new_model.keys())[0]]
    central_train(new_model=new_model_dict, use_cuda=True)



def get_weights(isCentral = True,selected_indice_models: list = []):
    """
    Method to load one or more static dict models saved from clients or central.

    args:
    - isCentral: boolean to indicate if the models are from central or clients.
    - selected_indice_models: list of integers with the indices of the models to be loaded. If empty, all models will be loaded. If [-1], only the last model will be loaded.
    """
    # Carregar pesos de um ou mais modelos vindos do cliente
    path = spath.PATH_CLIENT_MODELS.value if not isCentral else spath.PATH_CENTRAL_MODELS.value

    if not os.path.exists(spath.PATH_CLIENT_MODELS.value):
        return "There arent models saved from clients."
    
    models_path = sorted(os.listdir(path))

    if len(models_path) == 0:
        return "There arent models saved from clients."
    
    models_indices = []
    for model_name in models_path:
        model_idx = model_name.split("_")[-1]
        model_idx = model_idx.split(".")[0]
        models_indices.append(int(model_idx))

    selected_models_dict = {}
    if selected_indice_models == []:
        for model_name in models_path:
            model_path = path + model_name
            model_static_dict = torch.load(model_path)
            selected_models_dict[model_name] = model_static_dict

    elif selected_indice_models == [-1]:
        model_path = path + model_path[-1]
        model_static_dict = torch.load(model_path)
        selected_models_dict[model_path[-1]] = model_static_dict
    
    else:
        for sel_model in selected_indice_models:
            if sel_model in models_indices:
                model_name = "model_" + str(sel_model) + ".pt"
                model_path = path + model_name
                model_static_dict = torch.load(model_path)
                selected_models_dict[model_name] = model_static_dict

    return selected_models_dict
    



def get_analyses():
    # Carregar modelo (ou modelos) e fazer testes
    # Salvar testes em CSV
    #Seguindo esse modelo
    data = {
        "Index": ["Value1", "Value2"]
    }
    df = pd.DataFrame(data)
    df.to_csv("data.csv", index=False, encoding="utf-8")
    ...



def get_graphics():
    # Ler dados salvos em tabelas CSV e gerar analises graficas    
    ...



def gradCAM():
    ...



def lime():
    ...



def manipule_data():
    # Pegue o diretorio ./data, troque os labels de 7 para 1
    # aumente os dados de ./data duplicando o dataset e invertendo as cores dos dados duplicados
    # Carregue apenas N% dos dados
    # 

    mnist_trainset = datasets.MNIST(root='./data', train=True, download=False, transform=netTransform)


    # AQUI POSSO ALTERAR O MNIST TRAINSET


    total_size = len(mnist_trainset)
    train_size = int(0.7 * total_size)
    test_size = total_size - train_size
    train_dataset, test_dataset = random_split(mnist_trainset, [train_size, test_size])

    train_indices = train_dataset.indices
    test_indices = test_dataset.indices
    
    train_images = mnist_trainset.data[train_indices].clone()
    train_images[0]
    train_labels = mnist_trainset.targets[train_indices].clone()
    train_labels[0]

    test_images = mnist_trainset.data[test_indices].clone()
    test_labels = mnist_trainset.targets[test_indices].clone()


    #os.mkdir("./data/MNIST/processed")
    torch.save((train_images,train_labels), './data/MNIST/processed/training.pt')
    torch.save((test_images,test_labels), './data/MNIST/processed/test.pt')
    ...
