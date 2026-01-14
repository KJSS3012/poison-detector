import numpy as np
import cv2
import torch
import os

import torch 
from torch.nn import functional as F

from sysvars import SysVars as svar
from services.trains.modelNet import Net
from services.xai.gradcam.utils import load_image, preprocess_image, save_cam, save_sad_mask


def generate_scorecam(
    img_path,
    model_dict,
    model_name="model.pt",
    class_index=None,
    save=False
) -> torch.Tensor:

    device = svar.DEFAULT_DEVICE

    model = Net().to(device)
    model.load_state_dict(model_dict)
    model.eval()

    activations = {}

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    t_layer = model.conv1
    hook = t_layer.register_forward_hook(forward_hook)

    img = load_image(img_path).to(device)
    b, c, h, w = img.size()

    output = model(img)
    class_index = output.argmax(dim=1).item() if class_index is None else class_index

    activation_maps = activations['value']
    _, num_channels, _, _ = activation_maps.shape

    scores = torch.zeros(num_channels, device=device)
    baseline = img.mean()

    with torch.no_grad():
        for i in range(num_channels):

            saliency_map = activation_maps[:, i].unsqueeze(1)
            saliency_map = F.interpolate(
                saliency_map, size=(h, w), mode='bilinear', align_corners=False
            )

            if saliency_map.max() == saliency_map.min():
                continue

            saliency_map = (saliency_map - saliency_map.min()) / \
                        (saliency_map.max() - saliency_map.min() + 1e-8)

            masked_img = img * saliency_map + (1 - saliency_map) * baseline

            logits = model.forward_logits(masked_img)
            score = logits[0, class_index]

            scores[i] = score

    scores = scores.clamp(min=0)
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

    weights = scores.view(1, num_channels, 1, 1)
    scorecam = torch.sum(activation_maps * weights, dim=1).squeeze()
    scorecam = F.relu(scorecam)

    if save:
        mask = cv2.resize(scorecam.cpu().numpy(), (28, 28))
        save_cam(mask, img.cpu().numpy(), img_path, model_name)

    hook.remove()
    return scorecam



def mean_scorecam(model: dict, save_path: str = "", class_index = True):
    """
    Generate mean Score-CAM signature for each digit class (0-9).
    
    Unlike mean_gradCAM, Score-CAM is deterministic (same input = same output),
    so we don't need multiple iterations per image. Instead, we average across
    DIFFERENT images to create a robust signature for each digit class.
    
    This signature represents the typical activation pattern a model has for 
    each digit, which can be used to compare models and detect anomalies.
    
    Args:
        models (dict): Dictionary mapping model names to their state_dicts
        save_path (str): Path to save the mean CAMs. If empty, won't save
    
    Returns:
        list: List of 10 mean CAMs (one per digit 0-9), each of shape (28, 28)
    """

    # Load sample images and organize by digit class
    samples = os.listdir(svar.SAMPLE_IMAGES_PATH)
    numbers = {i: [] for i in range(10)}

    # Parse filenames to extract digit labels
    # Expected format: "number_X_Y.png" where X is the digit (0-9)
    for f in samples:
        num = f.split(".")[0]
        num = f.split("_")
        num = int(num[1])
        numbers[num].append(f)

    means_cams = []
    
    # Process each digit class (0-9)
    for num in range(10):
        cams = []

        # Generate Score-CAM for each sample image of this digit
        # Note: We only run once per image because Score-CAM is deterministic
        for f in numbers[num]:

            # Generate Score-CAM (single execution - no loops needed)
            x = generate_scorecam(
                img_path = svar.SAMPLE_IMAGES_PATH / f,
                model_dict = model,
                model_name = f"number_{num}_scorecam",
                class_index = num if class_index else None,
                save = False
            )
            x = x.detach().cpu().float()

            # Ensure CAM is 2D
            while x.ndim > 2:
                x = x.squeeze(0)

            assert x.ndim == 2, f"CAM must be 2D, but got shape {x.shape}"

            # Resize to standard size if needed (but don't normalize yet)

            if x.shape != (28, 28):
                x = F.interpolate(
                    x.unsqueeze(0).unsqueeze(0),
                    size=(28, 28),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
            
            cams.append(x)

        # Average across all images to create a signature for this digit
        # This signature represents the typical activation pattern
        mean_cam = torch.mean(torch.stack(cams), dim=0).detach().cpu().numpy()
        
        cam = mean_cam.squeeze()
        cam = cv2.resize(cam, (28, 28)) if cam.shape != (28, 28) else cam
        
        means_cams.append(cam)

        # Save if path is provided
        if save_path != "":
            # os.makedirs(save_path, exist_ok=True)
            save_sad_mask(mean_cam, f'{save_path}mean_cam_{num}.png')

    return means_cams
