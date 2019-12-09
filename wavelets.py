import pywt
import numpy as np
import pywt.data
import cv2
import imutils as im
import argparse


def w2D(image, mode, level):
    # calcular los coeficientes
    coeffs = pywt.wavedec2(image, mode, level=level)

    # procesar los coeficientes
    coeffs_H = list(coeffs)
    coeffs_H[0] *= 0

    # reconstruccion de la imagen
    im_array_H = pywt.waverec2(coeffs_H, mode)
    im_array_H *= 255
    im_array_H = np.uint8(im_array_H)

    return im_array_H


# mostrar los resultados
def apply_wavelets(img, families, level):
    cv2.imshow('Original', img)
    for i, family in enumerate(families):
        for level in range(1, level + 1):
            image_dec = w2D(img, family, level)
            cv2.imwrite(f'code_images/{family}_{level}.jpg', image_dec)
            cv2.imshow(f'Family: {family}, Level: {level}', image_dec)


ap = argparse.ArgumentParser(description='Ingrese familia de wavelet y nivel.')
ap.add_argument('-f', '--families', type=str, nargs='+', help='wavelet families')
ap.add_argument('-l', '--level', type=int, default=1, help='levels')
args = ap.parse_args()

# Cargar la imagen
image = cv2.imread('img/Dario_Velez.jpg')
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# reducir tamaño
gray_image = im.resize(gray_image, 200)

# convertirlo a float
gray_image = np.float32(gray_image)
gray_image /= 255

apply_wavelets(gray_image, args.families, args.level)

cv2.waitKey(0)
cv2.destroyAllWindows()
