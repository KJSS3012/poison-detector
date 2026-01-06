from controler import *

def main():
    #gradCAM(selected_indice_models=[4])
    #gradCAM(selected_indice_models=[5])
    #gradCAM(selected_indice_models=[6])
    #gradCAM(selected_indice_models=[7])
    #gradCAM(selected_indice_models=[8])
    # data_manipulations()
    for i in range(10):
        get_diffs(f"grafico_{i}.png")
    #post_central_train(selected_indice_models=[5,8])
    #get_analyses()
    #post_central_train()
    # lime()

if __name__ == "__main__":
    main()