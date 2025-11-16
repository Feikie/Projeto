# --- Importação das bibliotecas necessárias ---
import cv2  # A biblioteca principal do OpenCV, para processamento de imagem
import numpy as np  # Biblioteca para manipulação de arrays (o OpenCV usa)
import argparse  # Biblioteca para criar scripts de linha de comando (pegar o --image)

# --- Funções de callback para os trackbars ---
def nada(x):
    # Esta função é um truque. Os sliders (trackbars) do OpenCV EXIGEM
    # uma função para ser chamada toda vez que o slider é movido.
    # Como nós lemos os valores dentro do loop principal, não precisamos
    # que ela faça nada. 'pass' significa "não faça nada".
    pass

# --- Configuração inicial ---

# 1. Configura o "parser" que vai ler os argumentos da linha de comando
parser = argparse.ArgumentParser(description='Script interativo para ajustar parametros.')
# 2. Adiciona o argumento "-i" (ou "--image")
#    'required=True' significa que o script dará erro se não for fornecido.
parser.add_argument('-i', '--image', type=str, required=True, help='Caminho para a imagem.')
# 3. Executa o parser e armazena os argumentos
args = parser.parse_args()

# --- Carregamento da Imagem ---

# Carrega a imagem do caminho que o usuário forneceu (ex: "easy.jpg")
img_original = cv2.imread(args.image)
# Verificação de segurança: se a imagem não for encontrada, 'img_original' será 'None'
if img_original is None:
    # Imprime uma mensagem de erro e sai do script
    print(f"Erro: Nao foi possivel carregar a imagem: {args.image}")
    exit()

# Converte a imagem original (colorida, BGR) para escala de cinza
# A detecção de bordas (Canny) funciona em imagens de um canal (cinza)
gray = cv2.cvtColor(img_original, cv2.COLOR_BGR_GRAY)

# --- Criação da Interface Gráfica (Janelas e Sliders) ---

# Cria a janela principal para os sliders
# cv2.WINDOW_NORMAL permite que a janela seja redimensionada pelo usuário
cv2.namedWindow("Trackbars", cv2.WINDOW_NORMAL)

# Sintaxe: cv2.createTrackbar("Nome do Slider", "Janela", valor_inicial, valor_max, funcao_callback)
cv2.createTrackbar("Canny 1", "Trackbars", 30, 500, nada)   # Limiar 1 do Canny
cv2.createTrackbar("Canny 2", "Trackbars", 150, 500, nada)  # Limiar 2 do Canny
cv2.createTrackbar("Blur", "Trackbars", 7, 21, nada)        # Tamanho do "borrão" (blur)

# Sliders para limpeza morfológica
cv2.createTrackbar("Erode Iter", "Trackbars", 1, 10, nada)  # Nº de vezes que a erosão é aplicada
cv2.createTrackbar("Dilate Iter", "Trackbars", 1, 10, nada) # Nº de vezes que a dilatação é aplicada

# Sliders para filtrar os contornos
cv2.createTrackbar("Min Area", "Trackbars", 500, 10000, nada)   # Área mínima (em pixels)
cv2.createTrackbar("Max Area", "Trackbars", 20000, 50000, nada)  # Área máxima (em pixels)
cv2.createTrackbar("Min Ratio x100", "Trackbars", 20, 100, nada) # Proporção (0.0 a 1.0), multiplicada por 100

# Imprime as instruções no console para o usuário
print("\n--- INSTRUCOES (V2) ---")
print(f"Ajustando: {args.image}")
print("1. Ajuste 'Canny 1', 'Canny 2' e 'Blur' na janela 'Canny Edges'.")
print("2. Ajuste 'Erode' e 'Dilate' para limpar o ruido na janela 'Morphological Clean'.")
print("3. Finalmente, ajuste 'Area' e 'Ratio' para encaixar os circulos verdes.")
print("4. Pressione 'q' para sair.")
print("-----------------------\n")

# --- Loop Principal do Programa ---

# Define o "kernel", uma matriz 3x3 de 1s.
# É a "forma" usada para erodir e dilatar.
kernel = np.ones((3,3), np.uint8)

# Este loop roda "para sempre" (até o usuário apertar 'q')
# Isso permite que os sliders atualizem a imagem em tempo real
while True:
    # --- 1. Leitura dos Sliders ---
    # Pega o valor ATUAL de cada slider na janela "Trackbars"
    canny1 = cv2.getTrackbarPos("Canny 1", "Trackbars")
    canny2 = cv2.getTrackbarPos("Canny 2", "Trackbars")
    k_blur = cv2.getTrackbarPos("Blur", "Trackbars")
    erode_iter = cv2.getTrackbarPos("Erode Iter", "Trackbars")
    dilate_iter = cv2.getTrackbarPos("Dilate Iter", "Trackbars")
    min_area = cv2.getTrackbarPos("Min Area", "Trackbars")
    max_area = cv2.getTrackbarPos("Max Area", "Trackbars")
    # Converte o valor do slider (ex: 20) de volta para um float (ex: 0.20)
    min_ratio = cv2.getTrackbarPos("Min Ratio x100", "Trackbars") / 100.0

    # --- 2. Pré-processamento da Imagem ---

    # Garante que o valor do 'Blur' seja sempre ímpar
    # O GaussianBlur exige um tamanho de kernel ímpar (ex: 3, 5, 7...)
    if k_blur % 2 == 0:
        k_blur += 1
        
    # CRÍTICO: Sempre fazemos uma CÓPIA da imagem original
    # Se desenharmos na original, os círculos verdes serão permanentes
    img_copia = img_original.copy()
    
    # Aplica o "borrão" (blur) para suavizar a imagem e reduzir o ruído
    blurred = cv2.GaussianBlur(gray, (k_blur, k_blur), 0)
    
    # Aplica o detector de bordas Canny.
    # Ele encontra onde as intensidades dos pixels mudam bruscamente.
    edges = cv2.Canny(blurred, canny1, canny2)
    
    # --- 3. Limpeza Morfológica (O passo mais importante!) ---
    
    # 1. ERODE (Erosão): "Afina" as linhas brancas.
    #    Isso remove pequenos pontos de ruído (ruído "sal").
    edges_clean = cv2.erode(edges, kernel, iterations=erode_iter)
    
    # 2. DILATE (Dilatação): "Engrossa" as linhas brancas.
    #    Isso conecta bordas que estavam quebradas. (ex: transforma C em O)
    edges_clean = cv2.dilate(edges_clean, kernel, iterations=dilate_iter)
    
    # --- 4. Encontrar e Filtrar Contornos ---
    
    # Agora, procuramos os contornos (formas) na imagem JÁ LIMPA (edges_clean)
    # RETR_LIST: Pega todos os contornos, sem hierarquia.
    # CHAIN_APPROX_SIMPLE: Salva memória guardando só os pontos de "quina" da forma.
    contours, _ = cv2.findContours(edges_clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Zera a contagem a cada "frame" do loop
    pipe_count = 0

    # Itera sobre CADA contorno (forma) que foi encontrado
    for cnt in contours:
        # Calcula a área (em pixels) do contorno
        area = cv2.contourArea(cnt)
        
        # --- FILTRO 1: ÁREA ---
        # Se a forma for muito pequena (ruído) ou muito grande (uma borda), ignora.
        if area > min_area and area < max_area:
            
            # --- FILTRO 2: 5 PONTOS (Segurança) ---
            # O 'fitEllipse' precisa de pelo menos 5 pontos para calcular uma elipse.
            # Se o contorno tiver menos, ele dá crash. Isso previne o crash.
            if len(cnt) < 5:
                continue # Pula para o próximo contorno
                
            # 'try/except' é outro bloco de segurança.
            # Às vezes, 'fitEllipse' falha mesmo com 5+ pontos (ex: uma linha reta).
            try:
                # --- O CÁLCULO PRINCIPAL ---
                # Pega os pontos do contorno (cnt) e calcula a melhor
                # elipse que se encaixa neles.
                ellipse = cv2.fitEllipse(cnt)
                
                # Pega os eixos da elipse (ex: largura e altura)
                (ax1, ax2) = ellipse[1]
                
                # Previne divisão por zero
                if max(ax1, ax2) > 0:
                    # --- FILTRO 3: PROPORÇÃO (ASPECT RATIO) ---
                    # Divide o eixo menor pelo maior.
                    # Resultado: 1.0 (círculo perfeito), < 1.0 (elipse)
                    aspect_ratio = min(ax1, ax2) / max(ax1, ax2)
                else:
                    aspect_ratio = 0
                    
                # Se a forma for "circular" o suficiente (não uma linha fina)
                if aspect_ratio > min_ratio:
                    # Se passou em TODOS os filtros, conta como um tubo
                    pipe_count += 1
                    # Desenha a elipse verde na NOSSA CÓPIA da imagem
                    cv2.ellipse(img_copia, ellipse, (0, 255, 0), 2)
                    
            except cv2.error:
                # Se o 'fitEllipse' falhar, apenas ignora e continua
                continue
    
    # --- 5. Mostrar os Resultados ---
    
    # Coloca o texto da contagem na imagem (cor vermelha, BGR = 0,0,255)
    cv2.putText(img_copia, f"Contagem: {pipe_count}", (20, 70), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    # Mostra as 3 janelas de processamento
    cv2.imshow("Deteccoes", img_copia)                # Imagem final com elipses
    cv2.imshow("Canny Edges (Original)", edges)       # Saída do Canny (com ruído)
    cv2.imshow("Morphological Clean (Limpa)", edges_clean) # Imagem limpa (o que o 'findContours' vê)

    # --- 6. Condição de Saída ---
    
    # Espera 1ms por uma tecla.
    # `& 0xFF == ord('q')` checa se a tecla pressionada foi 'q'.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break # Quebra o loop 'while True'

# --- Limpeza Final ---
# Depois que o loop é quebrado, fecha todas as janelas do OpenCV.
cv2.destroyAllWindows()