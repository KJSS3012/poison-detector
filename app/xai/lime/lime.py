import numpy as np
import torch
from lime import lime_image
from skimage.segmentation import mark_boundaries
from torch.nn import functional as F
from sysvars import SysVars as svar


'''
    1 - Pegar n amostras para cada classe
    2 - Gerar explicações LIME para cada amostra
    3 - Guardar máscaras e explicações por classe
    4 - Comparar máscaras entre modelos (central e clientes) por média de classe 
'''
def analyze_with_lime(model, test_dataset, num_samples=50):
    """
    Analisa amostras usando LIME para detectar padrões de envenenamento
    """
    lime_explainer = lime_image.LimeImageExplainer()
    device = svar.DEFAULT_DEVICE.value
    model.eval()
    lime_results = []
    
    print(f"\nAnalisando {num_samples} amostras com LIME...")
    
    # Selecionar amostras aleatórias
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
        
        # Converter para formato numpy (para LIME)
        image_np = image.squeeze().numpy()
        
        # Função de predição para LIME
        def predict_fn(images):
            # Converter numpy arrays para tensors
            tensor_images = []
            for img in images:
                # Normalizar como no treinamento
                img_tensor = torch.FloatTensor(img).unsqueeze(0)  # Add channel dim
                img_tensor = (img_tensor - 0.1307) / 0.3081  # Normalização MNIST
                tensor_images.append(img_tensor)
            
            batch = torch.stack(tensor_images).to(device)
            
            with torch.no_grad():
                outputs = model(batch)
                probs = F.softmax(outputs, dim=1)
            
            return probs.cpu().numpy()
        
        # Gerar explicação LIME
        try:
            explanation = lime_explainer.explain_instance(
                image_np, predict_fn, top_labels=3, hide_color=0, num_samples=100
            )
            
            # Coletar informações da explicação
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