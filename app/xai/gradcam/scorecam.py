import numpy as np
import cv2
import torch
import os

import torch 
from torch.nn import functional as F

from sysvars import SysVars as svar
from modelNet import Net
from xai.gradcam.utils import load_image, preprocess_image, save_cam, save_sad_mask


def generate_scorecam(
            img_path, 
            model_dict, 
            model_name = "model.pt",
            class_index = None,
            save = False
            ) -> torch.Tensor:
    """
    Score-CAM: A deterministic approach to generate class activation maps.
    
    Unlike GradCAM, Score-CAM does not use gradients (no backpropagation).
    Instead, it evaluates the importance of each activation channel by:
    1. Using each channel as a mask on the input image
    2. Measuring how much each masked input increases the target class score
    3. Combining channels weighted by their scores
    
    This makes Score-CAM deterministic: same input always produces same output.
    
    Args:
        img_path (str): Path to the input image
        model_dict (dict): State dictionary of the trained model
        model_name (str): Name of the model for saving purposes
        class_index (int, optional): Target class index. If None, uses predicted class
        save (bool): Whether to save the visualization
    
    Returns:
        torch.Tensor: The Score-CAM activation map
    """
    
    # Save outputs of forward hooking
    activations = dict()
    device = svar.DEFAULT_DEVICE.value

    # Load model and set to evaluation mode
    model = Net().to(device)
    model.load_state_dict(model_dict)
    model.eval()  # Important: disables dropout and sets batchnorm to eval mode
    
    def forward_hook(module, input, output):
        activations['value'] = output.detach()
        return None
    
    # Register hook on target convolutional layer
    t_layer = model.conv2
    hook = t_layer.register_forward_hook(forward_hook)

    # Load and preprocess image
    img = load_image(img_path)

    # Initial forward pass to get activations and determine target class
    with torch.no_grad():  # Score-CAM doesn't need gradients
        output = model(img)
        class_index = torch.argmax(output).item() if class_index is None else class_index

    # Get activation maps from the target layer
    activation_maps = activations['value']  # Shape: (batch, channels, height, width)
    batch_size, num_channels, h, w = activation_maps.shape
    
    # Normalize each activation channel to [0, 1] range
    # This is necessary to use them as masks
    normalized_activations = torch.zeros_like(activation_maps)
    for i in range(num_channels):
        act = activation_maps[0, i, :, :]
        act_min, act_max = act.min(), act.max()
        if act_max > act_min:
            normalized_activations[0, i, :, :] = (act - act_min) / (act_max - act_min)
    
    # Upsample activation maps to input image size (28x28)
    # This allows us to use them as masks on the input
    upsampled_activations = F.interpolate(
        normalized_activations, 
        size=(28, 28), 
        mode='bilinear', 
        align_corners=False
    )  # Shape: (batch, channels, 28, 28)
    
    # Calculate importance score for each channel
    # Score = how much the channel contributes to the target class prediction
    scores = torch.zeros(num_channels, device=device)
    
    with torch.no_grad():  # No gradients needed
        for i in range(num_channels):
            # Extract the i-th channel as a mask
            mask = upsampled_activations[0, i, :, :].unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, 28, 28)
            
            # Apply mask to the input image
            # Higher activation values preserve more of the original image
            masked_img = img * mask
            
            # Forward pass with masked input
            masked_output = model(masked_img)
            
            # Score = confidence in the target class
            # Higher score means this channel is more important for the prediction
            scores[i] = F.softmax(masked_output, dim=1)[0, class_index]
    
    # Normalize scores to [0, 1] range
    scores = scores - scores.min()
    if scores.max() > 0:
        scores = scores / scores.max()
    
    # Combine activation maps weighted by their importance scores
    # Channels with higher scores contribute more to the final CAM
    weights = scores.view(1, num_channels, 1, 1)
    scorecam = torch.sum(activation_maps * weights, dim=1).squeeze()  # Shape: (height, width)
    
    # Apply ReLU to remove negative values
    scorecam = F.relu(scorecam)
    
    # Save visualization if requested
    if save: 
        mask = cv2.resize(scorecam.data.cpu().numpy(), (28,28))
        save_cam(mask, img.cpu().numpy(), img_path, model_name)
    
    # Remove hook to free memory
    hook.remove()

    return scorecam



def mean_scorecam(models: dict, save_path: str = ""):
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
    samples = os.listdir("./datasets/sample_images/")
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
            for model_name, model_dict in models.items():

                # Generate Score-CAM (single execution - no loops needed)
                x = generate_scorecam(
                    img_path = f"./datasets/sample_images/{f}",
                    model_dict = model_dict,
                    model_name = f"number_{num}_scorecam",
                    class_index = None,
                    save = False
                )
                x = x.detach().cpu().float()

                # Ensure CAM is 2D
                while x.ndim > 2:
                    x = x.squeeze(0)

                assert x.ndim == 2, f"CAM must be 2D, but got shape {x.shape}"

                # Normalize to [0, 1] range for consistency
                xmin, xmax = x.min(), x.max()
                if xmax > xmin:
                    x = (x - xmin) / (xmax - xmin)
                else:
                    x = torch.zeros_like(x)
                
                # Resize to standard size if needed
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
        
        # Final normalization
        cam_min, cam_max = mean_cam.min(), mean_cam.max()
        mean_cam = (mean_cam - cam_min) / (cam_max - cam_min) if cam_max > cam_min else np.zeros_like(mean_cam)
        
        means_cams.append(mean_cam)

        # Save if path is provided
        if save_path != "":
            os.makedirs(save_path, exist_ok=True)
            save_sad_mask(mean_cam, f'{save_path}mean_cam_{num}.png')

    return means_cams
