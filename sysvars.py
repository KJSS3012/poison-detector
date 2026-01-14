from pathlib import Path

class SysVars:
    """
    Centralized experiment-aware paths
    """

    ROOT = Path.cwd()

    DEFAULT_DEVICE = "cuda"

    EXPERIMENT_NAME = "experiment_002"
    EXPERIMENT_ROOT = ROOT / "experiments" / EXPERIMENT_NAME

    EXPERIMENT_CSV = EXPERIMENT_ROOT / "csv"
    EXPERIMENT_DATASETS = EXPERIMENT_ROOT / "datasets"
    EXPERIMENT_GRAPHICS = EXPERIMENT_ROOT / "imgs" / "graphics"
    EXPERIMENT_GRAD_CAMS = EXPERIMENT_ROOT / "imgs" / "gradCams"
    EXPERIMENT_MODELS = EXPERIMENT_ROOT / "models"

    TESTE_ROOT_PATH = ROOT / "experiments" / "test"

    DATASETS_ROOT = ROOT / "datasets"

    MNIST_TRAIN_PATH = DATASETS_ROOT / "mnist" / "training.pt"
    MNIST_TEST_PATH = DATASETS_ROOT / "mnist" / "test.pt"
    MNIST_ROOT_PATH = DATASETS_ROOT / "mnist"

    EMNIST_TRAIN_PATH = DATASETS_ROOT / "emnist" / "training.pt"
    EMNIST_TEST_PATH = DATASETS_ROOT / "emnist" / "test.pt"
    EMNIST_ROOT_PATH = DATASETS_ROOT / "emnist"

    SAMPLE_IMAGES_PATH = ROOT / "services" / "xai" / "sample_images"

    
