import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from sysvars import SysVars as svar
import os

def load_image(path):
    transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),    # Ensure single channel
            transforms.Resize((28, 28)),                    # Standardize size
            transforms.ToTensor(),                          # Convert to [0,1] range
            transforms.Normalize((0.1307,), (0.3081,))      # MNIST dataset statistics
        ])
    
    img = cv2.imread(path, 1)
    img = cv2.resize(img, (28, 28))
    img = np.float32(img) / 255
    img = torch.from_numpy(img).float()
    
    if len(img.shape) == 2:
        img = img.reshape(1, 28, 28)
    elif len(img.shape) == 3:
        img = img.mean(axis=2).reshape(1, 28, 28)

    img = torch.FloatTensor(img).cuda().unsqueeze(0) if svar.DEFAULT_DEVICE.value == 'cuda' else torch.FloatTensor(img).cpu().unsqueeze(0)

    return img

def preprocess_image(img, mean = 0.1307, std = 0.3081):
    """
    Preprocess the input image, in numpy array format, for the model.

    args:
        - img: input image in numpy array format
        - mean: mean for normalization  ( default: 0.1307 for MNIST )
        - std: standard deviation for normalization ( default: 0.3081 for MNIST )
    
    returns:
        - tensor_img: preprocessed image in tensor format
    """
    
    if len(img.shape) == 2:
        img = img.reshape(1, 28, 28)
    elif len(img.shape) == 3:
        img = img.mean(axis=2).reshape(1, 28, 28)

    tensor_img = torch.FloatTensor(img).unsqueeze(0)
    tensor_img = (tensor_img - mean) / std

    return tensor_img

def save_cam(mask, img, img_path, model_name):
    """
    Saves the Grad-CAM heatmap overlayed on the original image in the configured directory.

    args:
        - mask: gradcam in ndarray format
        - img: original image ndarray format
        - img_path: path of original image
    
    returns:
        None
    """

    mask = (mask - np.min(mask)) / np.max(mask)

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    img = img.squeeze()  # (28, 28)
    img = np.stack([img, img, img], axis=-1)  # (28, 28, 3)

    heatmap = np.float32(heatmap) / 255
    gradcam = 1.0 * heatmap + img
    gradcam = gradcam / np.max(gradcam)

    index = img_path.find('/')
    index2 = img_path.find('.')

    path = svar.PATH_GRADCAMS.value + 'result/' + model_name + '/' + img_path[index + 1:index2]
    if not (os.path.isdir(path)):
        os.makedirs(path)

    gradcam_path = path + "/gradcam.png"
    n = 1
    while True:
        gradcam_path = path + "/gradcam_" + str(n) + ".png" 
        if os.path.exists(gradcam_path):
            n += 1
            continue

        cv2.imwrite(gradcam_path, np.uint8(255 * gradcam))
        break

   
def save_cam_mask(mask, save_path):
    """
    Salva apenas o Grad-CAM (heatmap) como imagem.
    mask: ndarray 2D (ex: 28x28)
    save_path: caminho para salvar o arquivo
    """
    # Normaliza para 0–1
    mask = cv2.resize(mask, (28,28))
    mask = (mask - np.min(mask)) / np.max(mask)

    # Aplica colormap JET (azul → vermelho)

    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    #img = np.zeros(heatmap.shape, dtype=np.float32)

    heatmap = np.float32(heatmap) / 255

    #gradcam = 1.0 * heatmap + img
    gradcam = heatmap / np.max(heatmap)

    # Salva a imagem colorida
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path,  np.uint8(255 * gradcam))
    print(f"Heatmap salvo em: {save_path}")