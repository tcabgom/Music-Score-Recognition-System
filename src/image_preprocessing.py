import numpy as np

import cv2


def staff_line_filtering(binary_image, staff_lines):
    '''
    Elimina las líneas de la partitura de una imagen binaria.

    Parámetros:
        binary_image (array): Una matriz que representa la imagen binaria.
        staff_lines (array): Una matriz que representa las líneas de la partitura.

    Salidas:
        image_without_lines (array): La imagen binaria sin las líneas de la partitura.
    '''

    inverted_staff_lines = cv2.bitwise_not(staff_lines)                         # Invertir las líneas del pentagrama (cambiar 0 por 255 y viceversa)
    image_without_lines = cv2.bitwise_and(binary_image, inverted_staff_lines)   # Aplicar una operación de AND para eliminar las áreas donde ambas imágenes tienen píxeles negros
    inverted_staff_lines = cv2.bitwise_not(inverted_staff_lines)                # Invertir nuevamente las líneas del pentagrama para obtener las áreas que son solo negras en esa imagen
    inverted_binary_image = cv2.bitwise_not(binary_image)                       # Invertir la imagen binaria para obtener las áreas que son negras en la imagen binaria
    black_areas = cv2.bitwise_and(inverted_binary_image, inverted_staff_lines)  # Aplicar una operación de AND para obtener las áreas que son negras en ambas imágenes
    black_areas = cv2.bitwise_not(black_areas)                                  # Invertir nuevamente la imagen para obtener las áreas que son blancas en la imagen binaria
    image_without_lines = cv2.bitwise_or(image_without_lines, black_areas)      # Combinar las áreas que son negras en ambas imágenes con las áreas que son blancas en la imagen binaria
    return image_without_lines


def morphological_processing(binary_image, kernel_size):
    '''
    Aplica el procesamiento morfológico para mejorar la conectividad y estructura de la imagen binaria.

    Parámetros:
        binary_image (array): Una matriz que representa la imagen binaria.
        kernel_size (tuple): Tamaño del elemento estructurante para el cierre morfológico.

    Salidas:
        processed_image (array): La imagen binaria procesada morfológicamente.
    '''
    pass


def connected_component_labeling(image):
    pass
