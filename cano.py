import cv2
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")

args = ap.parse_args() 

img = cv2.imread(args.image) 

if img is None:
    print(f"Erro: Nao foi possivel carregar a imagem em: {args.image}")
    print("Verifique se o caminho esta correto.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (9, 9), 2)

circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=40,
    param1=80,
    param2=35,
    minRadius=27,
    maxRadius=40
)

if circles is not None:
    circles = np.uint16(np.around(circles))
    print("Number of circles:", len(circles[0]))
    for x, y, r in circles[0]:
        cv2.circle(img, (x, y), r, (0, 255, 0), 3)

cv2.imshow("Circles", img)
cv2.waitKey(0)
cv2.destroyAllWindows()