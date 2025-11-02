import torch
from modelNet import Net
import torch.optim as optim
from torchvision import datasets, transforms
from clients.model_mnist import train, test
import os
# import requests

def post_train(**kwargs: dict):
    """
    args:
    - batch_size: Integer of the batchs count to train (default int 64).
    - test_batch_size: Integer of the batchs count to test (default int 1000).
    - epochs: Integer of the epochs count to train (default int 20).
    - lr: Float to learning rate (default float 0.01).
    - momentum: Float to SGD momentum (default float 0.5).
    - seed: Random id seed (default int 1).
    - use_cuda: Boolean if you use Cuda Service (default bool False).
    - model_path: String with the path to saved model (default str "model_" + (n) + ".pt"). Note that the default model path is dynamic to prevent overwriting.
    - model_static_dict: n-darray with the weights of a trained model to fine tuning (default is a empty dictionary).
    - load_data: Boolean if you load a existing data (default bool False).
    - data_path: String with the path to load data (default str "./data").
    - log_interval: (default 10).
    """
    args = {
        "batch_size" : kwargs.get("batch_size", 64),
        "test_batch_size" : kwargs.get("test_batch_size", 1000),
        "epochs" : kwargs.get("epochs", 20),
        "lr" : kwargs.get("lr", 0.01),
        "momentum" : kwargs.get("momentum", 0.5),
        "seed" : kwargs.get("seed", 1),
        "use_cuda" : kwargs.get("use_cuda", False),
        "device" : "cuda" if kwargs.get("use_cuda", False) else "cpu",
        "model_path" : kwargs.get("model_path", ""),
        "model_static_dict" : kwargs.get("model_static_dict", {}),
        "load_data" : kwargs.get("load_data", False),
        "data_path" : kwargs.get("data_path", "./data"),
        "log_interval" : kwargs.get("log_interval", 10),
    }

    torch.manual_seed(args["seed"])
    kwargs = {'num_workers': 8, 'pin_memory': True} if args["use_cuda"] else {}

    try:
        mnist_trainset = datasets.MNIST(args["data_path"], train=True, download=False,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))

        train_loader = torch.utils.data.DataLoader(
            mnist_trainset,
            batch_size=args["batch_size"], shuffle=True, **kwargs)
        

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(args["data_path"], train=False, transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
            batch_size=args["test_batch_size"], shuffle=True, **kwargs)

    except:
        print("Fail in load data!")
        return False

    model = Net()
    if args["model_static_dict"] != {}:
        try:
            model.load_state_dict(args["model_static_dict"])
        except:
            print("Fail to load the static dict!")
            return False

    try:
        optimizer = optim.SGD(model.parameters(), lr=args["lr"], momentum=args["momentum"])

        for epoch in range(1, args["epochs"] + 1):
            train(args, model, args["device"], train_loader, optimizer, epoch)
            test(args, model, args["device"], test_loader)

        model_path = args["model_path"]
        if model_path == "":
            models_path = sorted(os.listdir("./clients/models/"))

            if len(models_path) != 0:
                last_idx = models_path[-1].split("_")[-1]
                model_path = "./clients/models/model_" + (int(last_idx) + 1) + ".pt"
            
            else:
                model_path = "./clients/models/model_1.pt"
        
        torch.save(model.state_dict(), model_path)
    
    except Exception as e:
        print("Fail in train!")
        print(e)
        return False

    return True

