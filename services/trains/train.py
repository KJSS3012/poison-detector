import torch
import torch.optim as optim
from torchvision import datasets, transforms
from services.trains.modelNet import Net, netTransform
from services.trains.model_mnist import train, test
from services.datasets.load_data import PTDataset
from sysvars import SysVars as svar
import os


# Torch configs to allow custom classes in serialization
torch.serialization.add_safe_globals([datasets.mnist.MNIST])
torch.serialization.add_safe_globals([transforms.transforms.Compose])
torch.serialization.add_safe_globals([transforms.transforms.ToTensor])
torch.serialization.add_safe_globals([transforms.transforms.Normalize])
torch.serialization.add_safe_globals([datasets.vision.StandardTransform])

def post_train(**kwargs: dict):
    """
    Method to train a model with MNIST dataset, the idea is to call this method passing the features you want to customize in training.

    Args:
    - batch_size: Integer of the batchs count to train (default int 64).
    - test_batch_size: Integer of the batchs count to test (default int 1000).
    - epochs: Integer of the epochs count to train (default int 20).
    - lr: Float to learning rate (default float 0.01).
    - momentum: Float to SGD momentum (default float 0.5).
    - seed: Random id seed (default int 1).
    - model_path: String with the path to saved model (default str "model_" + (n) + ".pt"). Note that the default model path is dynamic to prevent overwriting.
    - model_static_dict: n-darray with the weights of a trained model to fine tuning (default is a empty dictionary).
    - load_data: Boolean if you load a existing data (default bool False).
    - data_path: String with the path to load data (default str "./base_data_set").
    - log_interval: (default 10).
    - save_model: Boolean to save the model after training (default bool True).
    - dataset_interval: String to identify the dataset interval used in training (default "0.0 - 1.0").
    - poison: String to identify the poison used in training (default "no poison").
    
    Returns:
        state_dict (n-darray): The state dict of the trained model.
    """
    args = {
        "batch_size" : kwargs.get("batch_size", 64),
        "test_batch_size" : kwargs.get("test_batch_size", 1000),
        "epochs" : kwargs.get("epochs", 20),
        "lr" : kwargs.get("lr", 0.01),
        "momentum" : kwargs.get("momentum", 0.5),
        "seed" : kwargs.get("seed", 1),
        "model_path" : kwargs.get("model_path", ""),
        "model_static_dict" : kwargs.get("model_static_dict", {}),
        "load_data" : kwargs.get("load_data", False),
        "data_path" : kwargs.get("data_path", svar.MNIST_ROOT_PATH),
        "log_interval" : kwargs.get("log_interval", 10),
        "save_model" : kwargs.get("save_model", True),
        "dataset_interval" : kwargs.get("dataset_interval", "0.0 - 1.0"),
        "poison" : kwargs.get("poison", "no poison"),
    }

    device = svar.DEFAULT_DEVICE

    torch.manual_seed(args["seed"])
    kwargs = {'num_workers': 8, 'pin_memory': True} if device == 'cuda' else {}

    try:
        train_dataset = PTDataset(pt_file=args["data_path"] / "training.pt")

        test_dataset = PTDataset(pt_file=args["data_path"] / "test.pt")

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args["batch_size"], shuffle=True, **kwargs)
        

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args["test_batch_size"], shuffle=True, **kwargs)

    except Exception as e:
        print("Fail in load data!")
        print(e)
        return False

    model = Net().to(device)
    if args["model_static_dict"] != {}:
        try:
            model.load_state_dict(args["model_static_dict"])
        except:
            print("Fail to load the static dict!")
            return False

    try:
        optimizer = optim.SGD(model.parameters(), lr=args["lr"], momentum=args["momentum"])

        for epoch in range(1, args["epochs"] + 1):
            train(args, model, train_loader, optimizer, epoch)
            test(args, model, test_loader)

        # if model_path == "":
            
        #     control = 1
        #     path = svar.PATH_CLIENT_MODELS + "model_" + str(control) + ".pt"

        #     while os.path.exists(path):
        #         control += 1
        #         path = svar.PATH_CLIENT_MODELS + "model_" + str(control) + ".pt"

        #     model_path = path
        
        state_dict = model.state_dict()

        print("Train completed!")
        if args["save_model"]: 
            model_path = args["model_path"]
            torch.save(state_dict, model_path)
            # control_models_info(model_path, args["epochs"], args["poison"], args["dataset_interval"])
            
        return state_dict
    
    except Exception as e:
        print("Fail in train!")
        print(e)
        return False


def control_models_info(model_name: str, epochs: int, poison: str, dataset_name: str ,dataset_interval: str, save_path: str = None, isCentral: bool = False):
    """
    Method to save the models info in a json file.

    Args:
        model_name (str): The name of the model file.
        epochs (int): The number of epochs the model was trained.
        poison (str): The type of poison used in the dataset.
        dataset_name (str): The name of the dataset used.
        dataset_interval (str): The dataset interval used for training.
        save_path (str): The path to save the json
    Returns:
        None
    """
    
    save_path = save_path if save_path is not None else svar.EXPERIMENT_MODELS / "models_info.json"

    with open(save_path, "r") as f:
        import json
        models_info = json.load(f)
    
    models_info.append({
        "model_name": model_name,
        "epcohs": epochs,
        "poison": poison,
        "dataset_name": dataset_name,
        "dataset_interval": dataset_interval
    })

    with open(save_path, "w") as f:
        import json
        json.dump(models_info, f, indent=4)

def merge_models(new_model: dict, old_model: dict = None, old_mpath: str = None, alpha: float = 0.4):
    """
    Method to receive models from clients, merge them with the central model using average of the weights and save the updated model as central model.
    For this, send a compatible static dict model, please. If you have questions about compatibility, check the modelNet.py documentation.
    
    args:
    - new_model: static dict of the new model received from a client.
    - old_model: static dict of the old central model, if None it will be loaded from disk.
    - old_mpath: path to load/save the central model.
    - alpha: float value to weight the new model in the merging process.
    """

    if not is_compatible(new_model):
        print("\n=====================\n",f"The model is incompatible!\n=====================\n")
        return False
    
    device = svar.DEFAULT_DEVICE

    if old_model is None:
        if old_mpath and os.path.exists(old_mpath):
            old_model = torch.load(old_mpath, map_location=device)
        else:
            print("\n=========================\n",f"No old model found in {old_mpath}. A new model will be created.\n=========================\n")
            old_model = new_model
            alpha = 1.0

    try:
        updated_model = {}
        for k in old_model.keys():
            
            if k in new_model.keys() and old_model[k].shape == new_model[k].shape:
                updated_model[k] = (1 - alpha)*old_model[k] + alpha*new_model[k]
            
            else:
                updated_model[k] = old_model[k]
        
        if (old_mpath):
            torch.save(updated_model, old_mpath)
            print(f"Updated model saved in {old_mpath}")
        return updated_model

    except Exception as e:
        print("\n=========================\nMerged is failed!\nException: \n")
        print(e)
        print("\n=========================\n")
        return False

def merge_n_models(models: list, save_path: str = None):
    """
    Method to merge multiple models using simple arithmetic mean of the weights.
    Incompatible models are automatically removed from the list.
    
    Args:
        models (list): List of state_dicts to be merged.
        save_path (str | None): Path to save the merged model. If None, the model is not saved.
    
    Returns:
        dict | bool: The state_dict of the merged model, or False if something goes wrong.
    """
    
    # Filtra apenas modelos compatíveis
    compatible_models = [m for m in models if is_compatible(m)]
    
    if len(compatible_models) == 0:
        print("\n=========================\n")
        print("No compatible models found in the list!")
        print("\n=========================\n")
        return False
    
    removed_count = len(models) - len(compatible_models)
    if removed_count > 0:
        print(f"\n[INFO] {removed_count} incompatible model(s) removed from the list.\n")
    
    try:
        # Usa o primeiro modelo como base para as keys
        base_model = compatible_models[0]
        merged_model = {}
        n = len(compatible_models)
        
        for k in base_model.keys():
            # Soma os pesos de todos os modelos compatíveis
            weight_sum = torch.zeros_like(base_model[k], dtype=torch.float32)
            
            for model in compatible_models:
                weight_sum += model[k].float()
            
            # Calcula a média aritmética simples
            merged_model[k] = weight_sum / n
        
        # Salva o modelo se um caminho foi fornecido
        if save_path is not None:
            torch.save(merged_model, save_path)
            print(f"Merged model saved in {save_path}")
        
        print(f"Successfully merged {n} models.")
        return merged_model
    
    except Exception as e:
        print("\n=========================\n")
        print("Merge failed! Exception:")
        print(e)
        print("\n=========================\n")
        return False


def is_compatible(static_dict):
    """
    Method to check if a static dict model is compatible with the base model defined in modelNet.py

    args:
    - static_dict: static dict of the model to be checked.
    """
    base_model = Net()
    model_dict = base_model.state_dict()

    missing = model_dict.keys() - static_dict.keys()
    unexpected = static_dict.keys() - model_dict.keys()

    if missing or unexpected:
        return False
    
    for k in model_dict.keys():
        if model_dict[k].shape != static_dict[k].shape:
            return False
        
    return True