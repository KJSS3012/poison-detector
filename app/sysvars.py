from enum import Enum

class SysVars(Enum):
    PATH_BASE_DATASET = "./base_data_set/"
    PATH_CLIENT_MODELS = "./clients/models/"
    PATH_CENTRAL_MODELS = "./central/models/"
    DEFAULT_DEVICE = "cuda"