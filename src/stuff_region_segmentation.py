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
    _, binary_image = cv2.threshold(
        image, BINARY_TRANFORM_THRESHOLD, 255, cv2.THRESH_BINARY)
    return binary_image


def horizontal_proyection(binary_image):
    '''
    Realiza una proyección horizontal de una imagen binaria dada para obtener así su histograma de proyección horizontal.

    Entradas:
        image (array): Una matriz que representa la imagen binaria de entrada.

    Salidas:
        horizontal_hist (array): Un histograma de proyección horizontal.
    '''
    horizontal_hist = np.sum(binary_image, axis=1)
    return horizontal_hist


def region_segmentation(image):
    pass


def test_binary_transform():
    image_path = 'images/Test Sheet 1.png'
    image = cv2.imread(image_path, 0)
    binary_image = binary_transform(image)
    cv2.imshow('Binary Image', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_binary_transform()
