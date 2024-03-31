import numpy as np
import stuff_region_segmentation

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


def morphological_processing(image):
    pass


def connected_component_labeling(image):
    pass


def test_remove_staff_lines(resize=False):
    image_path = 'images/Test Sheet 5.png'
    image = cv2.imread(image_path, 0)
    
    # Obtener la imagen binaria y las líneas de la partitura
    binary_image = stuff_region_segmentation.binary_transform(image)
    horizontal_sum = stuff_region_segmentation.horizontal_projection(binary_image)
    staff_lines = stuff_region_segmentation.region_segmentation(horizontal_sum)
    
    # Eliminar las líneas de la partitura de la imagen binaria
    image_without_lines = staff_line_filtering(binary_image, staff_lines)
    
    if resize:
        # Redimensionar imágenes para una mejor visualización
        binary_image = cv2.resize(binary_image, (0, 0), fx=0.5, fy=0.5)
        staff_lines = cv2.resize(staff_lines, (0, 0), fx=0.5, fy=0.5)
        image_without_lines = cv2.resize(image_without_lines, (0, 0), fx=0.5, fy=0.5)
    
    # Visualizar las imágenes
    cv2.imshow('Image Without Staff Lines', image_without_lines)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_remove_staff_lines(True)