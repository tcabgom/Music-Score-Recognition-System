import cv2
import numpy as np


BINARY_TRANFORM_THRESHOLD = 127


def binary_transform(image):
    '''
    Realiza una transformación binaria en una imagen dada.

    Entradas:
        image (array): Una matriz que representa la imagen de entrada.

    Salidas:
        binary_image (array): La imagen transformada binariamente.
    '''
    _, binary_image = cv2.threshold(image, BINARY_TRANFORM_THRESHOLD, 255, cv2.THRESH_BINARY)
    return binary_image


def horizontal_projection(image):
    '''
    Calcula el histograma de proyección horizontal de una imagen dada.

    Parámetros:
        image (array): Una matriz que representa la imagen de entrada en escala de grises.

    Salidas:
        array: Un array que contiene el histograma de proyección horizontal.
    '''
    height, width = image.shape[:2]                                                     # Obtiene las dimensiones de la imagen
    horizontal_sum = np.sum(image, axis=1)                                              # Calcula la suma horizontal de los píxeles en la imagen
    max_value = np.max(horizontal_sum)                                                  # Encuentra el valor máximo de la suma horizontal
    if max_value == 0:                                                                  # Si la suma máxima es cero, devuelve una matriz de ceros con las mismas dimensiones que la imagen original
        return np.zeros((height, width), dtype=np.uint8)

    horizontal_sum_normalized = (horizontal_sum / max_value) * (width - 1)              # Normaliza la suma horizontal
    horizontal_image = np.zeros((height, width), dtype=np.uint8)                        # Crea una matriz de ceros con las mismas dimensiones que la imagen original
    for y, value in enumerate(horizontal_sum_normalized):                               # Itera sobre la suma horizontal para dibujar las líneas de la proyección horizontal
        cv2.line(horizontal_image, (width - int(value) - 1, y), (width, y), 255, 1)     # El -1 es para que no se dibuje una línea en la primera columna de la imagen

    return horizontal_image


def region_segmentation(image):
    pass


def test_binary_transform(resize = False):
    image_path = 'images/Test Sheet 8.png'
    image = cv2.imread(image_path, 0)
    binary_image = binary_transform(image)
    if resize:
        binary_image = cv2.resize(binary_image, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('Binary Image', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_horizontal_projection(resize = False):
    image_path = 'images/Test Sheet 8.png'
    image = cv2.imread(image_path, 0)
    binary_image = binary_transform(image)
    horizontal_sum = horizontal_projection(binary_image)
    if resize:
        horizontal_sum = cv2.resize(horizontal_sum, (0, 0), fx=0.5, fy=0.5)
    cv2.imshow('Horizontal Projection', horizontal_sum)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_binary_transform(True)
    test_horizontal_projection(True)
