import torch
from modelNet import Net
from sysvars import SysVars as svar

def post_train(new_model: dict, old_model: dict = None, old_mpath: str = "central_model.pt", use_cuda: bool = False, alpha: float = 0.2):
    """
    Method to receive models from clients, merge them with the central model using average of the weights and save the updated model as central model.
    For this, send a compatible static dict model, please. If you have questions about compatibility, check the modelNet.py documentation.
    
    args:
    - new_model: static dict of the new model received from a client.
    - old_model: static dict of the old central model, if None it will be loaded from disk.
    - old_mpath: path to load/save the central model.
    - use_cuda: boolean to indicate if cuda is used.
    - alpha: float value to weight the new model in the merging process.
    """

    if not is_compatible(new_model):
        print("\n=====================\n",f"The model is incompatible!\n=====================\n")
        return False
    
    device = "cuda" if use_cuda else "cpu"

    if old_model is None:
        if os.path.exists(svar.PATH_CENTRAL_MODELS.value + old_mpath):
            old_model = torch.load(svar.PATH_CENTRAL_MODELS.value + old_mpath, map_location=device)
        else:
            old_model = Net().state_dict()
            alpha = 1.0

    try:
        updated_model = {}
        for k in old_model.keys():
            
            if k in new_model.keys() and old_model[k].shape == new_model[k].shape:
                updated_model[k] = (1 - alpha)*old_model[k] + alpha*new_model[k]
            
            else:
                updated_model[k] = old_model[k]
        
        torch.save(updated_model, old_mpath)
        print(f"Updated model saved in {old_mpath}")
        return True

    except Exception as e:
        print("\n=========================\nMerged is failed!\nException: \n")
        print(e)
        print("\n=========================\n")
        return False



def is_compatible(static_dict):
    """
    Method to check if a static dict model is compatible with the base model defined in modelNet.py

    args:
    - static_dict: static dict of the model to be checked.
    """
    base_model = Net()
    model_dict = base_model.state_dict()

    missing = model_dict.keys() - static_dict.keys()
    unexpected = static_dict.keys() - model_dict.keys()

    if missing or unexpected:
        return False
    
    for k in model_dict.keys():
        if model_dict[k].shape != static_dict[k].shape:
            return False
        
    return True