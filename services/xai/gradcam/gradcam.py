import numpy as np
import cv2
import torch
import os

import torch 
from torch.nn import functional as F
from torch.autograd import Variable

from sysvars import SysVars as svar
from services.trains.modelNet import Net
from services.xai.gradcam.utils import load_image, preprocess_image, save_cam, save_sad_mask


def generate_gradcam(
            img_path, 
            model_dict, 
            model_name = "model.pt",
            class_index = None,
            save = False,
            plusplus = False
            ) -> torch.Tensor:
    
    activations = {}
    gradients = {}
    device = svar.DEFAULT_DEVICE

    model = Net().to(device)
    model.load_state_dict(model_dict)
    model.eval()

    def forward_hook(module, input, output):
        activations['value'] = output.detach()

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0].detach()

    # 🔑 camada correta para MNIST
    t_layer = model.conv1
    hook1 = t_layer.register_forward_hook(forward_hook)
    hook2 = t_layer.register_full_backward_hook(backward_hook)

    img = load_image(img_path)

    # 🔑 usar LOGITS
    output = model.forward_logits(img)
    class_index = output.argmax(dim=1).item() if class_index is None else class_index

    score = output[0, class_index]

    model.zero_grad()
    score.backward()

    grads = gradients['value']
    acts = activations['value']

    if plusplus:
        grads2 = grads ** 2
        grads3 = grads2 * grads
        sum_acts = acts.sum(dim=(2, 3), keepdim=True)

        eps = 1e-8
        aij = grads2 / (2 * grads2 + sum_acts * grads3 + eps)
        aij = torch.where(grads != 0, aij, torch.zeros_like(aij))

        weights = (torch.relu(grads) * aij).sum(dim=(2, 3))
    else:
        weights = grads.mean(dim=(2, 3))

    weights = weights[0][:, None, None]
    cam = F.relu((weights * acts[0]).sum(dim=0))

    hook1.remove()
    hook2.remove()

    if save:
        mask = cv2.resize(cam.data.cpu().numpy(), (28,28))
        save_cam(mask, img.cpu().numpy(), img_path, model_name)

    return cam



def mean_gradCAM(model: dict, save_path: str = "", scale=10, plusplus = False, class_index=True):

    samples = os.listdir(svar.SAMPLE_IMAGES_PATH)
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

                x = generate_gradcam(
                    img_path = svar.SAMPLE_IMAGES_PATH / f,
                    model_dict = model,
                    model_name = f"number_{num}_belign",
                    class_index = num if class_index else None,
                    save = False,
                    plusplus=plusplus
                )
                x = x.detach().cpu().float()

                while x.ndim > 2:
                    x = x.squeeze(0)

                assert x.ndim == 2, f"CAM deve ser 2D, mas veio {x.shape}"

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
        cam = cv2.resize(cam, (28, 28)) if cam.shape != (28, 28) else cam

        means_cams.append(cam)

        if save_path != "":
            os.makedirs(save_path, exist_ok=True)
            save_sad_mask(cam, f'{save_path}mean_cam_{num}.png')

    return means_cams

def mean_plusplusCAM(model: dict, save_path: str = "", scale=10, class_index=True):

    return mean_gradCAM(model, save_path, scale, plusplus=True, class_index=class_index)
#
#
        #for f, masks in cams.items():
        #    print(f"\n\nDistance scores for image {f}:")
        #    get_distance_scores(masks)
