import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torch.nn import functional as F
from sysvars import SysVars as svar

def analyze_with_lime(model, test_dataset, num_samples=50):
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


    #########
#
#
#   Fazer lógica para preparar os samples
#
#   for j in range(num_classes):
#       for i in range(samples_per_class):
#
    #########
    
    for i, idx in enumerate(indices):
        if i % 10 == 0:
            print(f"Analisando amostra {i+1}/{num_samples}")
        
        image, true_label = test_dataset[idx]
        image_np = image.squeeze().numpy()
    
        def predict_fn(images):
            tensor_images = []
            for img in images:
                img_tensor = torch.FloatTensor(img).unsqueeze(0)
                img_tensor = (img_tensor - 0.1307) / 0.3081
                tensor_images.append(img_tensor)
            
            batch = torch.stack(tensor_images).to(device)
            
            with torch.no_grad():
                outputs = model(batch)
                probs = F.softmax(outputs, dim=1)
            
            return probs.cpu().numpy()
        
        try:
            explanation = lime_explainer.explain_instance(
                image_np, predict_fn, top_labels=3, hide_color=0, num_samples=100
            )
            
            temp, mask = explanation.get_image_and_mask(
                explanation.top_labels[0], positive_only=True, 
                num_features=10, hide_rest=False
            )
            
            lime_results.append({
                'index': idx,
                'true_label': true_label,
                'explanation': explanation,
                'mask': mask,
                'image': image_np
            })
            
        except Exception as e:
            print(f"Erro na análise LIME da amostra {idx}: {e}")
            continue
    
    return lime_results