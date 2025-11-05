import numpy as np
import cv2

import torch 
from torch.nn import functional as F
from torch.autograd import Variable
from sysvars import SysVars as svar

from xai.gradcam.utils import load_image, load_model, preprocess_image, save 

from modelNet import Net

class GradCAM():
    def __init__(self,
                img_path, 
                model_dict, 
                class_index = None):

        self.img_path = img_path
        self.model_dict = model_dict
        self.class_index = class_index
        
        # Save outputs of forward and backward hooking
        self.gradients = dict()
        self.activations = dict()
        self.device = svar.DEFAULT_DEVICE.value

        m = Net().to(self.device)
        m.load_state_dict(self.model_dict)
        self.model = m

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]
            return None
        def forward_hook(module, input, output):
            self.activations['value'] = output
            return None
        
        self.t_layer = self.model.conv2
        self.t_layer.register_forward_hook(forward_hook)
        self.t_layer.register_full_backward_hook(backward_hook)
        
    def __call__(self):

        print('\nGradCAM start ... ')

        self.img = load_image(self.img_path)

        #numpy to tensor and normalize
        self.input = preprocess_image(self.img)

        output = self.model(self.input)
        self.class_index = np.argmax(output.cpu().data.numpy()) if self.device == 'cpu' else np.argmax(output.data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype = np.float32)
        one_hot[0][self.class_index] = 1
        one_hot = Variable(torch.from_numpy(one_hot), requires_grad = True)
        one_hot = torch.sum(one_hot * output) if self.device == 'cpu' else torch.sum(one_hot.cuda() * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph = True)
        
        gradients = self.gradients['value']
        activations = self.activations['value']
        
        #reshaping
        weights = torch.mean(torch.mean(gradients, dim=2), dim=2)
        weights = weights.reshape(weights.shape[1], 1, 1)
        activationMap = torch.squeeze(activations[0])
           
        #Get gradcam
        gradcam = F.relu((weights*activationMap).sum(0))
        gradcam = cv2.resize(gradcam.data.cpu().numpy(), (28,28))
        save(gradcam, self.img, self.img_path, self.model_path)
        
        print('GradCAM end !!!\n')


