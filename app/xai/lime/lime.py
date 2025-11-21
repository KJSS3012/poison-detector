import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import torch
from torchvision import transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
from torch.nn import functional as F
from sysvars import SysVars as svar

def analyze_with_lime(model, test_dataset, num_samples=100):
    """
    Generates LIME explanations for randomly selected samples in a dataset.

    This function applies the LIME (Local Interpretable Model-agnostic Explanations) 
    technique to identify important image regions that influence model predictions. 
    It is particularly useful for analyzing and detecting anomalies, such as 
    data poisoning or class-based bias in a trained model.

    The process includes:
        1. Randomly selecting `num_samples` images from `test_dataset`.
        2. Applying LIME to generate explanations for each image's predicted class.
        3. Gathering and returning a list containing explanation masks and metadata.

    Args:
        model (torch.nn.Module):
            The trained PyTorch model to be analyzed.
        test_dataset (torch.utils.data.Dataset):
            Dataset containing samples to be explained. Must return 
            `(image, label)` tuples where `image` is a tensor.
        num_samples (int, optional):
            Number of random samples to analyze. Defaults to 50.

    Returns:
        list[dict]:
            A list of dictionaries, each containing:
                - 'index' (int): Index of the sample in the dataset.
                - 'true_label' (int): Ground truth label of the image.
                - 'explanation' (lime.explanation.ImageExplanation): 
                    The full LIME explanation object.
                - 'mask' (numpy.ndarray): Binary mask highlighting important regions.
                - 'image' (numpy.ndarray): Original image data in numpy format.

    Raises:
        Exception:
            If the LIME analysis of a sample fails, the function prints 
            the error and continues with the next sample.
    """

    lime_explainer = lime_image.LimeImageExplainer()
    device = svar.DEFAULT_DEVICE.value
    model.eval()
    lime_results = []
    
    print(f"\nAnalisando {num_samples} amostras com LIME...")
    
    indices = np.random.choice(len(test_dataset), num_samples, replace=False)

    indices = {
        0 : [],
        1 : [],
        2 : [],
        3 : [],
        4 : [],
        5 : [],
        6 : [],
        7 : [],
        8 : [],
        9 : []
    }
    

    explain_id = 0
    path = svar.PATH_LIMES.value + f'explain_{explain_id}/'
    while(os.path.exists(path)):
        explain_id += 1
        path = svar.PATH_LIMES.value + f'explain_{explain_id}/'
    os.makedirs(path)


    for i in test_dataset:
        image, label = i
        if (len(indices[label]) <= 10):
            indices[label].append((image, label))
    
    for i, elements in indices.items():
        print(f"Analisando amostras do dígito {i}...")


        for image, true_label in elements:
    
            image_np = image.squeeze().numpy()
        
            def predict_fn(images):
                tensor_images = []
                for img in images:
                    # LIME envia (28,28,3)
                    # Converter para cinza, pois seu modelo é MNIST com 1 canal
                    img_gray = img.mean(axis=2)  # (28,28)
                    
                    # Transformar em tensor e adicionar canal
                    img_tensor = torch.FloatTensor(img_gray).unsqueeze(0)  # (1,28,28)
                    
                    # Normalizar como no treinamento
                    img_tensor = (img_tensor - 0.1307) / 0.3081
                    
                    tensor_images.append(img_tensor)
                
                batch = torch.stack(tensor_images).to(device)
                
                with torch.no_grad():
                    outputs = model(batch)
                    probs = F.softmax(outputs, dim=1)
                
                #return probs.cpu().numpy() if (svar.DEFAULT_DEVICE.value == 'cpu') else probs.cuda().numpy()
                return probs.cpu().numpy()
            
            try:
                explanation = lime_explainer.explain_instance(
                    image_np, predict_fn, top_labels=1, hide_color=0, num_samples=1000
                )
                
                temp, mask = explanation.get_image_and_mask(
                    explanation.top_labels[0], positive_only=True, 
                    num_features=5, hide_rest=False, min_weight=0.1
                )

                lime_img = mark_boundaries(temp, mask)
                lime_img = np.clip(lime_img, 0, 1)
                # Salva em arquivo
                label_id = 0
                label_path = path + f"lime_result_{true_label}_id{label_id}.png"
                while(os.path.exists(label_path)):
                    label_id += 1
                    label_path = path + f"lime_result_{true_label}_id{label_id}.png"

                plt.imsave(label_path, lime_img)

                lime_results.append({
                    'true_label': true_label,
                    'explanation': explanation,
                    'temp': temp,
                    'mask': mask,
                    'image': image_np
                })
                
            except Exception as e:
                print(f"Erro na análise LIME da amostra: {e}")
                continue
    
    return lime_results



def load_image(img):
    
    img = cv2.resize(img, (28, 28))
    img = np.float32(img) / 255
    img = torch.from_numpy(img).float()
    
    if len(img.shape) == 2:
        img = img.reshape(1, 28, 28)
    elif len(img.shape) == 3:
        img = img.mean(axis=2).reshape(1, 28, 28)

    img = torch.FloatTensor(img).cuda().unsqueeze(0) if svar.DEFAULT_DEVICE.value == 'cuda' else torch.FloatTensor(img).cpu().unsqueeze(0)

    return img

def slic_segmenter(image):
    return slic(
        image,
        n_segments=50,      # mais regiões
        compactness=0.1,
        sigma=1
    )
