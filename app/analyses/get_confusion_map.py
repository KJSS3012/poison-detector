from modelNet import Net, netTransform
import torch
import torch.nn.functional as F
from torchvision import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sysvars import SysVars as svar

def get_confusion_map(state_dict, model_id, data_path=svar.PATH_BASE_DATASET.value):
    """
    Method to generate and plot the confusion matrix of a model with MNIST dataset.
    This method returns the confusion matrix as a nested dictionary.

    args:
    - model: the weights in static dict format of the model to be evaluated.
    - model_id: string with the identifier of the model, used to save the confusion map.
    - data_path: path to the dataset location (default value is extracted from system config).

    returns:
    - predicts: nested dictionary representing the confusion matrix.
    """
    device = svar.DEFAULT_DEVICE.value
    model = Net().to(device)
    model.load_state_dict(state_dict)
    model.eval()

    mnist_trainset = torch.utils.data.DataLoader(
        datasets.MNIST(data_path, train=False, transform=netTransform),
        batch_size=1, shuffle=True, **{})

    predicts = {
        0: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        },
        1: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        },
        2: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        },
        3: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        },
        4: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        },
        5: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        },
        6: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        },
        7: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        },
        8: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        },
        9: {
            0: 0,
            1: 0,
            2: 0,
            3: 0,
            4: 0,
            5: 0,
            6: 0,
            7: 0,
            8: 0,
            9: 0
        }
    }

    with torch.no_grad():

        for data, target in mnist_trainset:
            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities = F.softmax(output, dim=1)
            predicted = output.argmax(dim=1, keepdim=True)
            predicted_class = predicted[0].item()
            predicts[target[0].item()][predicted_class] += 1

    df = pd.DataFrame(predicts).T
    print(df)
    plt.figure(figsize=(10, 7))
    plt.title("Confusion Map")
    sns.heatmap(df, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Real")
    plt.savefig(svar.PATH_ANALYSES_GRAPHICS.value + model_id)
    plt.show()
    print(f"Confusion map saved in '{svar.PATH_ANALYSES_GRAPHICS.value + model_id}'")

    return predicts