import torch
import torch.nn.functional as F
from torchvision import datasets
from modelNet import Net, netTransform
from sysvars import SysVars as svar
from datasets.load_data import PTDataset

def get_accuracy(state_dict, data_path=svar.PATH_BASE_DATASET.value):
    """
    Method to test a model with MNIST dataset.
    This method returns the accuracy of the model on the test dataset.

    args:
    - model: the weights in static dict format of the model to be tested.
    - device: the device to run the testing on ('cpu' or 'cuda').
    - data_path: path to the dataset location.

    returns:
    - accuracy: float representing the accuracy of the model on the test dataset.
    """

    device = svar.DEFAULT_DEVICE.value
    test_loader = torch.utils.data.DataLoader(
        
    PTDataset(pt_file = data_path, download=False, train=False, transform=netTransform),
        batch_size=64, shuffle=True, **{})
    
    model = Net().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return correct / len(test_loader.dataset)
    
