from enum import Enum

class SysVars(Enum):
    """
    System Variables used across the project
    """
    DEFAULT_DEVICE = "cuda"
    PATH_BASE_DATASET = "./datasets/base_data_set/"
    PATH_CLIENT_MODELS = "./clients/models/"
    PATH_CENTRAL_MODELS = "./central/models/"
    PATH_ANALYSES_GRAPHICS = "./analyses/graphics/"
    PATH_ANALYSES_CVS = "./analyses/csv/"
    PATH_GRADCAMS = "./analyses/gradcams/"
    PATH_LIMES = "./analyses/limes/"