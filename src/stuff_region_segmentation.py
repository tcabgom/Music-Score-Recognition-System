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

    horizontal_sum = np.sum(image, axis=1)
    max_value = np.max(horizontal_sum)
    if max_value == 0:
        return np.zeros((image.shape[0], 512), dtype=np.uint8)

    horizontal_sum_normalized = (horizontal_sum / max_value) * 255
    horizontal_image = np.zeros((image.shape[0], 512), dtype=np.uint8)
    for y, value in enumerate(horizontal_sum_normalized):
        cv2.line(horizontal_image, (0, y), (int(value), y), 255, 1)

    return horizontal_image


def region_segmentation(image):
    pass


def test_binary_transform():
    image_path = 'images/Test Sheet 1.png'
    image = cv2.imread(image_path, 0)
    binary_image = binary_transform(image)
    cv2.imshow('Binary Image', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_horizontal_projection():
    image_path = 'images/Test Sheet 2.png'
    image = cv2.imread(image_path, 0)
    binary_image = binary_transform(image)
    horizontal_sum = horizontal_projection(binary_image)
    cv2.imshow('Horizontal Projection', horizontal_sum)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_binary_transform()
    test_horizontal_projection()
