import torch
import torch.nn.functional as F
from sysvars import SysVars as svar

def train(args, model, train_loader, optimizer, epoch):
    """
    Receive a model and train it with the provided data loader and optimizer for one epoch.

    args:
    - args: dict with training parameters.
    - model: the neural network model to be trained in static dict format.
    - device: the device to run the training on ('cpu' or 'cuda').
    - train_loader: DataLoader providing the training data. 
    """
    device = svar.DEFAULT_DEVICE.value
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args["log_interval"] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, test_loader):
    """
    Receive a model and test it with the provided data loader.

    args:
    - args: dict with testing parameters.
    - model: the neural network model to be tested in static dict format.
    - device: the device to run the testing on ('cpu' or 'cuda').
    - test_loader: DataLoader providing the testing data.
    """
    device = svar.DEFAULT_DEVICE.value
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

