import numpy as np
import cv2
import torch
import os

import torch 
from torch.nn import functional as F
from torch.autograd import Variable

from sysvars import SysVars as svar
from modelNet import Net
from xai.gradcam.utils import load_image, preprocess_image, save_cam, save_sad_mask


def generate_gradcam(
            img_path, 
            model_dict, 
            model_name = "model.pt",
            class_index = None,
            save = False,
            plusplus = False
            ) -> torch.Tensor:

    gradients = dict()
    activations = dict()
    device = svar.DEFAULT_DEVICE.value

    model = Net().to(device)
    model.load_state_dict(model_dict)
    model.eval()

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()
        return None
    def forward_hook(module, input, output):
        activations['value'] = output.detach()
        return None
    
    t_layer = model.conv2
    hook1 = t_layer.register_forward_hook(forward_hook)
    hook2 = t_layer.register_full_backward_hook(backward_hook)

    img = load_image(img_path)


    output = model(img)
    class_index = torch.argmax(output).item() if class_index is None else class_index

    one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
    one_hot[0][class_index] = 1
    one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
    one_hot = torch.sum(one_hot * output) if device == 'cpu' else torch.sum(one_hot.cuda() * output)

    model.zero_grad()
    one_hot.backward(retain_graph = True)
    
    gradients = gradients['value']
    activations = activations['value']
    
    if plusplus:
        grads_power_2 = gradients**2
        grads_power_3 = grads_power_2 * gradients

        # Equation 19 in https://arxiv.org/abs/1710.11063
        eps = 1e-6
        sum_activations = activations.sum(dim=(2, 3))
        aij = grads_power_2 / (
            2 * grads_power_2
            + sum_activations[:, :, None, None] * grads_power_3
            + eps
        )

        # Now bring back the ReLU from eq.7 in the paper,
        # And zero out aijs where the activations are 0
        aij = torch.where(gradients != 0, aij, torch.zeros_like(aij))

        weights = torch.relu(gradients) * aij
        weights = weights.sum(dim=(2, 3))  # (N, C)

    else:
        #reshaping
        weights = torch.mean(torch.mean(gradients, dim=2), dim=2)
          
    #Get gradcam
    weights = weights[0][:, None, None]
    activationMap = activations[0]
    gradcam = F.relu((weights * activationMap).sum(dim=0))

    if save: 
        mask = cv2.resize(gradcam.data.cpu().numpy(), (28,28))
        save_cam(mask, img.cpu().numpy(), img_path, model_name)
    
    hook1.remove()
    hook2.remove()

    return gradcam



def mean_gradCAM(models: dict, save_path: str = "", scale=10):

    samples = os.listdir("./datasets/sample_images/")
    numbers = {i: [] for i in range(10)}

    for f in samples:
        num = f.split(".")[0]
        num = f.split("_")
        num = int(num[1])
        numbers[num].append(f)

    means_cams = []

    for num in range(10):
        cams = []

        for f in numbers[num]:

            for i in range(scale):

                for model_name, model_dict in models.items():

                    x = generate_gradcam(
                        img_path = f"./datasets/sample_images/{f}",
                        model_dict = model_dict,
                        model_name = f"number_{num}_belign",
                        class_index = None,
                        save = False,
                        plusplus=True
                    )
                    x = x.detach().cpu().float()

                    while x.ndim > 2:
                        x = x.squeeze(0)

                    assert x.ndim == 2, f"CAM deve ser 2D, mas veio {x.shape}"

                    xmin, xmax = x.min(), x.max()
                    if xmax > xmin:
                        x = (x - xmin) / (xmax - xmin)
                    else:
                        x = torch.zeros_like(x)
                    if x.shape != (28, 28):
                        x = torch.nn.functional.interpolate(
                            x.unsqueeze(0).unsqueeze(0),
                            size=(28, 28),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze()
                    cams.append(x)

                    #save_cam_mask(cams[f][i].detach().cpu().numpy(), f'./analyses/gradcams/cams_means/solid_cams/{num}_{i}.png')


        stack = torch.stack(cams)
        stack = stack / (scale * 5)
        mean_cam = torch.mean(stack, axis=0).detach().cpu().numpy()
        
        cam = mean_cam.squeeze()
        cam_min, cam_max = np.min(cam), np.max(cam)
        cam = (cam - cam_min) / (cam_max - cam_min) if cam_max > cam_min else np.zeros_like(cam)
        cam = cv2.resize(cam, (28, 28)) if cam.shape != (28, 28) else cam

        means_cams.append(cam)

        if save_path != "":
            os.makedirs(save_path, exist_ok=True)
            save_sad_mask(cam, f'{save_path}mean_cam_{num}.png')

    return means_cams
#
#
        #for f, masks in cams.items():
        #    print(f"\n\nDistance scores for image {f}:")
        #    get_distance_scores(masks)
