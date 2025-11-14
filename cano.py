import cv2
import numpy as np
import argparse

# --- Funções de callback para os trackbars (não precisam fazer nada) ---
def nada(x):
    pass

# --- Configuração inicial ---
parser = argparse.ArgumentParser(description='Script interativo para ajustar parametros.')
parser.add_argument('-i', '--image', type=str, required=True, help='Caminho para a imagem.')
args = parser.parse_args()

# Carrega a imagem UMA VEZ
img_original = cv2.imread(args.image)
if img_original is None:
    print(f"Erro: Nao foi possivel carregar a imagem: {args.image}")
    exit()

gray = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)

# --- Cria a janela para os sliders ---
cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)
cv2.createTrackbar("Canny 1", "Trackbars", 30, 500, nada)
cv2.createTrackbar("Canny 2", "Trackbars", 150, 500, nada)
cv2.createTrackbar("Blur", "Trackbars", 7, 21, nada) # Kernel de blur (impar)

# --- NOVOS SLIDERS PARA LIMPEZA ---
cv2.createTrackbar("Erode Iter", "Trackbars", 1, 10, nada) # Iterações de Erosão
cv2.createTrackbar("Dilate Iter", "Trackbars", 1, 10, nada) # Iterações de Dilatação

cv2.createTrackbar("Min Area", "Trackbars", 500, 10000, nada)
cv2.createTrackbar("Max Area", "Trackbars", 20000, 50000, nada)
cv2.createTrackbar("Min Ratio x100", "Trackbars", 20, 100, nada) # (ex: 20 = 0.2)

print("\n--- INSTRUCOES (V2) ---")
print(f"Ajustando: {args.image}")
print("1. Ajuste 'Canny 1', 'Canny 2' e 'Blur' na janela 'Canny Edges'.")
print("2. Ajuste 'Erode' e 'Dilate' para limpar o ruido na janela 'Morphological Clean'.")
print("3. Finalmente, ajuste 'Area' e 'Ratio' para encaixar os circulos verdes.")
print("4. Pressione 'q' para sair.")
print("-----------------------\n")

# Kernel para morfologia (pode ser (3,3) ou (5,5))
kernel = np.ones((3,3), np.uint8)

while True:
    # --- Pega os valores atuais dos sliders ---
    canny1 = cv2.getTrackbarPos("Canny 1", "Trackbars")
    canny2 = cv2.getTrackbarPos("Canny 2", "Trackbars")
    k_blur = cv2.getTrackbarPos("Blur", "Trackbars")
    erode_iter = cv2.getTrackbarPos("Erode Iter", "Trackbars")
    dilate_iter = cv2.getTrackbarPos("Dilate Iter", "Trackbars")
    min_area = cv2.getTrackbarPos("Min Area", "Trackbars")
    max_area = cv2.getTrackbarPos("Max Area", "Trackbars")
    min_ratio = cv2.getTrackbarPos("Min Ratio x100", "Trackbars") / 100.0

    # Garante que o blur seja ímpar
    if k_blur % 2 == 0:
        k_blur += 1
        
    # --- Lógica de detecção ---
    img_copia = img_original.copy()
    
    blurred = cv2.GaussianBlur(gray, (k_blur, k_blur), 0)
    edges = cv2.Canny(blurred, canny1, canny2)
    
    # --- PASSO DE LIMPEZA MORFOLÓGICA (A GRANDE MUDANÇA) ---
    # 1. Erode para remover ruído
    edges_clean = cv2.erode(edges, kernel, iterations=erode_iter)
    # 2. Dilate para fechar buracos nos contornos dos tubos
    edges_clean = cv2.dilate(edges_clean, kernel, iterations=dilate_iter)
    # --------------------------------------------------------
    
    # IMPORTANTE: Usar a imagem limpa (edges_clean) para encontrar contornos
    # Mudei de RETR_EXTERNAL para RETR_LIST para pegar todos os contornos
    contours, _ = cv2.findContours(edges_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    pipe_count = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if area > min_area and area < max_area:
            if len(cnt) < 5:
                continue
            try:
                ellipse = cv2.fitEllipse(cnt)
                
                (ax1, ax2) = ellipse[1]
                if max(ax1, ax2) > 0:
                    aspect_ratio = min(ax1, ax2) / max(ax1, ax2)
                else:
                    aspect_ratio = 0
                    
                if aspect_ratio > min_ratio:
                    pipe_count += 1
                    cv2.ellipse(img_copia, ellipse, (0, 255, 0), 2)
            except cv2.error:
                continue
    
    # --- Mostra os resultados ---
    cv2.putText(img_copia, f"Contagem: {pipe_count}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    cv2.imshow("Deteccoes", img_copia)
    cv2.imshow("Canny Edges (Original)", edges)
    cv2.imshow("Morphological Clean (Limpa)", edges_clean) # A JANELA MAIS IMPORTANTE!

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()