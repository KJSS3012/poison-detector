import numpy as np
import cv2
import torch
import os

import torch 
from torch.nn import functional as F
from torch.autograd import Variable

from sysvars import SysVars as svar
from modelNet import Net
from xai.gradcam.utils import load_image, preprocess_image, save_cam


def generate_gradcam(
            img_path, 
            model_dict, 
            model_name = "model.pt",
            class_index = None,
            save = False
            ) -> torch.Tensor:

    # Save outputs of forward and backward hooking
    gradients = dict()
    activations = dict()
    device = svar.DEFAULT_DEVICE.value

    model = Net().to(device)
    model.load_state_dict(model_dict)

    def backward_hook(module, grad_input, grad_output):
        gradients['value'] = grad_output[0]
        return None
    def forward_hook(module, input, output):
        activations['value'] = output
        return None
    
    t_layer = model.conv2
    t_layer.register_forward_hook(forward_hook)
    t_layer.register_full_backward_hook(backward_hook)

    img = load_image(img_path)

    #numpy to tensor and normalize
    #input = preprocess_image(img)

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
    
    #reshaping
    weights = torch.mean(torch.mean(gradients, dim=2), dim=2)
    weights = weights.reshape(weights.shape[1], 1, 1)
    activationMap = torch.squeeze(activations[0])
          
    #Get gradcam
    gradcam = F.relu((weights*activationMap).sum(0))

    if save: 
        mask = cv2.resize(gradcam.data.cpu().numpy(), (28,28))
        save_cam(mask, img.cpu().numpy(), img_path, model_name)
    
    return gradcam



def mean_gradCAM(models: dict):

    cams = {}

    samples = os.listdir("./datasets/sample_images/")
    numbers = {
        0:[],
        1:[],
        2:[],
        3:[],
        4:[],
        5:[],
        6:[],
        7:[],
        8:[],
        9:[]
    }

    for f in samples:
        num = f.split(".")[0]
        num = f.split("_")
        num = int(num[1])
        numbers[num].append(f)


    model = ''
    for model_name in models.keys():
        model = model_name

    j = 1
    path = './analyses/gradcams/cams_means/' + model.split(".")[0] + f'_{j}/'
    while os.path.exists(path):

        j += 1
        path = './analyses/gradcams/cams_means/' + model.split(".")[0] + f'_{j}/'
    
    # os.makedirs(path)

    means_cams = []
    for num in range(10):

        model = ''
        cams = []

        for f in numbers[num]:

            for i in range(10):

                for model_name, model_dict in models.items():

                    cams.append(generate_gradcam(
                        img_path = f"./datasets/sample_images/{f}",
                        model_dict = model_dict,
                        model_name = f"number_{num}_belign",
                        class_index = None,
                        save = False
                    ))

                    #save_cam_mask(cams[f][i].detach().cpu().numpy(), f'./analyses/gradcams/cams_means/solid_cams/{num}_{i}.png')


                    cam = cams[i].detach().cpu().numpy().squeeze()
                    cam_min, cam_max = np.min(cam), np.max(cam)


                    cam = (cam - cam_min) / (cam_max - cam_min) if cam_max > cam_min else np.zeros_like(cam)

                    cam = cv2.resize(cam, (28, 28)) if cam.shape != (28, 28) else cam


        stack = torch.stack(cams)
        mean_cam = torch.mean(stack, axis=0).detach().cpu().numpy()
        means_cams.append(mean_cam)

    return means_cams
        #save_cam_mask(mean_cam, f'{path}mean_cam_{num}.png')
#
#
        #for f, masks in cams.items():
        #    print(f"\n\nDistance scores for image {f}:")
        #    get_distance_scores(masks)
