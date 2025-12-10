import torch
from modelNet import Net, netTransform
import torch.optim as optim
from torchvision import datasets, transforms
from clients.model_mnist import train, test
from sysvars import SysVars as svar
import os

from datasets.load_data import PTDataset

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
        "data_path" : kwargs.get("data_path", svar.PATH_BASE_DATASET.value),
        "log_interval" : kwargs.get("log_interval", 10),
        "save_model" : kwargs.get("save_model", True),
        "dataset_interval" : kwargs.get("dataset_interval", "0.0 - 1.0"),
        "poison" : kwargs.get("poison", "no poison"),
    }

    device = svar.DEFAULT_DEVICE.value

    torch.manual_seed(args["seed"])
    kwargs = {'num_workers': 8, 'pin_memory': True} if device == 'cuda' else {}

    try:
        train_dataset = PTDataset(pt_file=args["data_path"] + "training.pt")

        test_dataset = PTDataset(pt_file=args["data_path"] + "test.pt")

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

        model_path = args["model_path"]
        if model_path == "":
            
            control = 1
            path = svar.PATH_CLIENT_MODELS.value + "model_" + str(control) + ".pt"

            while os.path.exists(path):
                control += 1
                path = svar.PATH_CLIENT_MODELS.value + "model_" + str(control) + ".pt"

            model_path = path
        
        state_dict = model.state_dict()

        print("Train completed!")
        if args["save_model"]: 
            torch.save(state_dict, model_path)
            control_models_info(model_path, args["epochs"], args["poison"], args["dataset_interval"])
            
        return state_dict
    
    except Exception as e:
        print("Fail in train!")
        print(e)
        return False


def control_models_info(model_name: str, epochs: int, poison: str, dataset_interval: str):
    """
    Method to save the models info in a json file.

    Args:
        model_name (str): The name of the model file.
        epochs (int): The number of epochs the model was trained.
        poison (str): The type of poison used in the dataset.
        dataset_interval (str): The dataset interval used for training.
    Returns:
        None
    """
    
    with open(svar.MODELS_INFO.value, "r") as f:
        import json
        models_info = json.load(f)
    
    models_info.append({
        "model_name": model_name,
        "epcohs": epochs,
        "poison": poison,
        "dataset_interval": dataset_interval
    })

    with open(svar.MODELS_INFO.value, "w") as f:
        import json
        json.dump(models_info, f, indent=4)
