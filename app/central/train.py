import torch
from modelNet import Net
from syspaths import SysPaths as spath

def post_train(new_model: dict, old_model: dict = None, old_mpath: str = "central_model.pt", use_cuda: bool = False, alpha: float = 0.2):
    
    if not is_compatible(new_model):
        print("\n=====================\n",f"The model is incompatible!\n=====================\n")
        return False
    
    device = "cuda" if use_cuda else "cpu"

    if old_model is None:
        if os.path.exists(spath.PATH_CENTRAL_MODELS.value + old_mpath):
            old_model = torch.load(spath.PATH_CENTRAL_MODELS.value + old_mpath, map_location=device)
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