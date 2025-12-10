from sysvars import SysVars as svar
from torchvision import datasets
from torchvision.utils import save_image
from torch.utils.data import random_split
from modelNet import Net, netTransform
import torch
import os
from datasets.load_data import PTDataset

def save_mnist_examples(base_path='./datasets/sample_images/'):
    """
    Saves sample images from the MNIST dataset to the specified directory.
    This function loads the MNIST training dataset and saves the first five
    images of each digit (0-9) to the specified directory.

    Args:
        base_path (str): The directory where the sample images will be saved.
                         Default is './datasets/sample_images/'.
    Returns:
        None
    """

    # mnist_trainset = PTDataset(pt_file = svar.PATH_BASE_DATASET.value + "training.pt")
    mnist_trainset = PTDataset(pt_file = svar.PATH_BASE_DATASET.value + "test.pt")

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
            save_image(t, f'{base_path}sample_{c}_{numbers[c]}.png')

def create_poisoned_dataset_x_to_y(x: int, y: int, path: str = None, path_to_load: str = None):
    """
    Manipulates the MNIST dataset to create a poisoned version.

    This function:
        - Loads the original MNIST training dataset.
        - Reassigns all label values `x` to `y` (data poisoning step).
        - Splits the dataset into training and testing subsets.
        - Saves the new poisoned dataset.

    Args:
        x (int): The original label value to be changed.
        y (int): The new label value to assign.
        path (str, optional): The directory where the dataset will be saved. Default is None.
        path_to_load (str, optional): The directory from which to load the original dataset. Default is None.
    Returns:
        None
    """

    path_to_load = svar.PATH_BASE_DATASET.value if path_to_load is None else path_to_load

    # mnist_trainset = datasets.MNIST(root=path_to_load, train=True, download=False, transform=netTransform)
    # mnist_testset = datasets.MNIST(root=path_to_load, train=False, download=False, transform=netTransform)

    mnist_trainset = PTDataset(pt_file = path, root=path_to_load, train=True, download=False, transform=netTransform)
    mnist_testset = PTDataset(pt_file = path, root=path_to_load, train=False, download=False, transform=netTransform)
    
    train_targets = mnist_trainset.targets.clone()
    train_targets[train_targets == x] = y
    train_data = mnist_trainset.data.clone()

    test_targets = mnist_testset.targets.clone()
    test_targets[test_targets == x] = y
    test_data = mnist_testset.data.clone()


    if path is None:
        control = 1
        path = f"./datasets/poisoned_data_set_{control}/"
        while os.path.exists(path):
            control += 1
            path = f"./datasets/poisoned_data_set_{control}/"

    os.makedirs(f"{path}", exist_ok=True)
        


    torch.save((train_data, train_targets), f'{path}training.pt')
    torch.save((test_data, test_targets), f'{path}test.pt')




def create_dataset(init_tax_interval: float, end_tax_interval: float, path: str = None, path_to_load: str = None):
    """
    Manipulates the MNIST dataset to create a non poisoned version.
    This function can be create a dataset with a specific tax interval (like 0.1 to 0.5 datas).

    Args:
        init_tax_interval (float): The initial tax interval (between 0 and 1).
        end_tax_interval (float): The end tax interval (between 0 and 1).
        path (str, optional): The directory where the dataset will be saved. Default is None.
        path_to_load (str, optional): The directory from which to load the original dataset. Default is None.
    Returns:
        None
    """

    if path_to_load is None:
        path_to_load = svar.PATH_BASE_DATASET.value

    mnist_trainset = PTDataset(pt_file = path_to_load + "training.pt")
    mnist_testset = PTDataset(pt_file = path_to_load + "test.pt")
    
    train_size = len(mnist_trainset)
    test_size = len(mnist_testset)

    init_idx_train = int(train_size * init_tax_interval)
    end_idx_train = int(train_size * end_tax_interval)
    init_idx_test = int(test_size * init_tax_interval)
    end_idx_test = int(test_size * end_tax_interval)

    train_targets = mnist_trainset.targets[init_idx_train:end_idx_train].clone()
    train_data = mnist_trainset.data[init_idx_train:end_idx_train].clone()
    test_targets = mnist_testset.targets[init_idx_test:end_idx_test].clone()
    test_data = mnist_testset.data[init_idx_test:end_idx_test].clone()

    if path is None:
        control = 1
        path = f"./datasets/belign_data_set_{control}/"
        while os.path.exists(path):
            control += 1
            path = f"./datasets/belign_data_set_{control}/"

    os.makedirs(f"{path}", exist_ok=True)
        


    torch.save((train_data, train_targets), f'{path}training.pt')
    torch.save((test_data, test_targets), f'{path}test.pt')

def create_randomized_dataset(size:float = 0.2, path_to_load: str = None, path_to_save: str = None):
    """
    Manipulates the MNIST dataset to create a randomized version.
    This function shuffles the MNIST dataset and saves it.

    Args:
        size (float): The fraction of the dataset to include in the randomized set (between 0 and 1).
        path_to_load (str, optional): The directory from which the dataset will be loaded. Default is None.
        path_to_save (str, optional): The directory where the dataset will be saved. Default is None.
    Returns:
        None
    """

    path_to_load = svar.PATH_BASE_DATASET.value + "training.pt" if path_to_load is None else path_to_load

    dataset = PTDataset(pt_file = path_to_load + "training.pt")
    total_size = len(dataset)
    subset_size = int(total_size * size)

    train_dataset, test_dataset  = random_split(dataset, [subset_size, total_size - subset_size])

    if path_to_save is None:
        control = 1
        path_to_save = f"./datasets/randomized_data_set_{control}/"
        while os.path.exists(path_to_save):
            control += 1
            path_to_save = f"./datasets/randomized_data_set_{control}/"
    os.makedirs(f"{path_to_save}", exist_ok=True)

    train_data = train_dataset.dataset.data[train_dataset.indices].clone()
    train_targets = train_dataset.dataset.targets[train_dataset.indices].clone()
    test_data = test_dataset.dataset.data[test_dataset.indices].clone()
    test_targets = test_dataset.dataset.targets[test_dataset.indices].clone()

    torch.save((train_data, train_targets), f'{path_to_save}training.pt')
    torch.save((test_data, test_targets), f'{path_to_save}test.pt')