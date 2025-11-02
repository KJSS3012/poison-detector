import os
from clients.train import post_train
from torchvision import datasets, transforms
from torch.utils.data import random_split
import torch
from modelNet import netTransform

def post_client_train():
    post_train(epochs=5, use_cuda=True, load_data=True)

def get_client_weights():
    # Carregar pesos de um ou mais modelos vindos do cliente
    ...

def post_central_train():
    # Enviar novos pesos para a central
    ...

def get_central_weights():
    # Carregar pesos de um ou mais modelos vindos da central
    ...

def get_analyses():
    # Carregar modelo (ou modelos) e fazer testes
    # Salvar testes em CSV
    ...

def get_graphics():
    # Ler dados salvos em tabelas CSV e gerar analises graficas    
    ...

def gradCAM():
    ...

def lime():
    ...

def poison_data():
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
    train_labels = mnist_trainset.targets[train_indices].clone()

    test_images = mnist_trainset.data[test_indices].clone()
    test_labels = mnist_trainset.targets[test_indices].clone()


    #os.mkdir("./data/MNIST/processed")
    torch.save((train_images,train_labels), './data/MNIST/processed/training.pt')
    torch.save((test_images,test_labels), './data/MNIST/processed/test.pt')
    ...

def main():
    # post_client_train()
    poison_data()

if __name__ == "__main__":
    main()