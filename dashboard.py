from controler import *
from sysvars import SysVars

def main():
    #gradCAM(selected_indice_models=[4])
    #gradCAM(selected_indice_models=[5])
    #gradCAM(selected_indice_models=[6])
    #gradCAM(selected_indice_models=[7])
    #gradCAM(selected_indice_models=[8])
    data_manipulations()
    # get_diffs(f"grafico_scorecam.png")
    #post_central_train(selected_indice_models=[5,8])
    #get_analyses()
    #post_central_train()
    # lime()
    experiment_001()
    ...

if __name__ == "__main__":
    main()