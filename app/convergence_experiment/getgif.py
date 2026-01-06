import os
import cv2
import imageio
import numpy as np
import natsort
from tqdm import tqdm

def gerar_gifs_por_pasta(base_dir, saida_dir=None, fps=8):
    """
    Gera GIFs da evolução dos CAMs dentro de uma única pasta,
    adicionando uma barra de progresso desenhada no próprio GIF.
    """

    if saida_dir is None:
        saida_dir = base_dir.rstrip("/") + "_gifs"

    os.makedirs(saida_dir, exist_ok=True)

    epocas = natsort.natsorted([
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ])

    print(f"Pasta: {base_dir}  |  {len(epocas)} épocas")

    for num in range(10):
        arquivo = f"num_{num}.png" if base_dir[-1] == 'd' else f"mean_cam_{num}.png"
        frames = []

        print(f"Processando num{num}...")

        for i, ep in enumerate(tqdm(epocas, desc=f"num{num}", leave=True)):
            caminho_img = os.path.join(base_dir, ep, arquivo)

            if not os.path.exists(caminho_img):
                continue

            img = cv2.imread(caminho_img, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Dimensões
            h, w = img.shape[:2]

            # Criar espaço extra para a barra (20px de altura)
            barra_altura = 20
            frame = np.zeros((h + barra_altura, w, 3), dtype=np.uint8)

            # Copia a imagem original
            frame[:h, :, :] = img

            # Progresso (0–1)
            progresso = (i + 1) / len(epocas)
            largura_barra = int(progresso * w)

            # Barra preenchida (verde)
            cv2.rectangle(
                frame,
                (0, h), (largura_barra, h + barra_altura),
                (0, 255, 0),
                thickness=-1
            )

            # Borda da barra (branca)
            cv2.rectangle(
                frame,
                (0, h), (w, h + barra_altura),
                (255, 255, 255),
                thickness=2
            )

            frames.append(frame)

        # Salvar GIF
        if frames:
            gif_path = os.path.join(saida_dir, f"num{num}.gif")
            imageio.mimsave(gif_path, frames, fps=fps)
            print(f"GIF salvo em: {gif_path}")
        else:
            print(f"[ERRO] Nenhum frame para num{num}")



def gif_comparativo_3pastas(
    dirA, dirB, dirC,
    saida_dir="gifs_comparativos",
    fps=8
):
    """
    Cria GIFs sincronizados mostrando:
        [ dirA | dirB | dirC ]
    para todos os números (0..9) ao longo das épocas.
    """

    os.makedirs(saida_dir, exist_ok=True)

    # Listar épocas a partir da DIR A (assume que B e C são iguais)
    epocas = natsort.natsorted([
        d for d in os.listdir(dirA)
        if os.path.isdir(os.path.join(dirA, d))
    ])
    
    print(f"Épocas detectadas: {len(epocas)}")

    # Para cada número de 0 a 9
    for num in range(10):
        frames = []
        arq = f"num_{num}.png"
        arqb = f"mean_cam_{num}.png"

        print(f"\nGerando GIF comparativo para num{num}:")

        for i, ep in enumerate(tqdm(epocas, desc=f"num{num}", leave=True)):

            # Caminhos sincronizados
            imgA_path = os.path.join(dirA, ep, arqb)
            imgB_path = os.path.join(dirB, ep, arqb)
            imgC_path = os.path.join(dirC, f"epoch_{i+1}", arq)

            # Carregar
            imgA = cv2.imread(imgA_path, cv2.IMREAD_COLOR)
            imgB = cv2.imread(imgB_path, cv2.IMREAD_COLOR)
            imgC = cv2.imread(imgC_path, cv2.IMREAD_COLOR)

            if imgA is None or imgB is None or imgC is None:
                print(f"[AVISO] Frame faltando no epoch {i} para num{num}.")
                continue

            # Normalizar tamanhos (mesma altura/largura)
            h, w = imgA.shape[:2]
            imgB = cv2.resize(imgB, (w, h))
            imgC = cv2.resize(imgC, (w, h))

            # BGR → RGB
            imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2RGB)
            imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2RGB)
            imgC = cv2.cvtColor(imgC, cv2.COLOR_BGR2RGB)

            # Concat horizontal: A | B | C
            combinado = np.concatenate([imgA, imgB, imgC], axis=1)

            # --- BARRA DE PROGRESSO DENTRO DO FRAME ---
            barra_altura = 20
            h2, w2 = combinado.shape[0], combinado.shape[1]

            frame = np.zeros((h2 + barra_altura, w2, 3), dtype=np.uint8)
            frame[:h2] = combinado

            progresso = (i + 1) / len(epocas)
            largura = int(progresso * w2)

            # Barra preenchida
            cv2.rectangle(
                frame, (0, h2), (largura, h2 + barra_altura),
                (0, 255, 0), thickness=-1
            )

            # Borda da barra
            cv2.rectangle(
                frame, (0, h2), (w2, h2 + barra_altura),
                (255, 255, 255), thickness=2
            )

            frames.append(frame)

        # Salvar GIF final do número
        if frames:
            out_path = os.path.join(saida_dir, f"num{num}.gif")
            imageio.mimsave(out_path, frames, fps=fps)
            print(f"GIF salvo em: {out_path}")
        else:
            print(f"[ERRO] Nenhum frame para num{num}!")



if __name__ == "__main__":
    gerar_gifs_por_pasta("single_cams/mnist1", "gifs/mnist1")
    gerar_gifs_por_pasta("single_cams/mnist2", "gifs/mnist2")
    gerar_gifs_por_pasta("sad_cams/mxm_ssim", "gifs/mxm_ssim")
    gif_comparativo_3pastas("single_cams/mnist1", "single_cams/mnist2", "sad_cams/mxm_ssim", "gifs/all_3")
