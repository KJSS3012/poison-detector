import numpy as np
import cv2
import torch

import torch 
from torch.nn import functional as F
from torch.autograd import Variable

from sysvars import SysVars as svar
from modelNet import Net
from xai.gradcam.utils import load_image, preprocess_image, save_cam


def gradcam(
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

    print('\nGradCAM start ... ')

    img = load_image(img_path)

    #numpy to tensor and normalize
    #input = preprocess_image(img)

    output = model(img)
    class_index = np.argmax(output.cpu().data.numpy()) if device == 'cpu' else np.argmax(output.cuda().data.numpy())

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

    if not save: 
        mask = cv2.resize(gradcam.data.cpu().numpy(), (28,28)) if device == 'cpu' else cv2.resize(gradcam.data.cuda().numpy(), (28,28))
        save_cam(mask, img, img_path, model_name)
    
    return gradcam



