import numpy as np
from queue import Queue
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
    Aplica el procesamiento morfológico para mejorar la conectividad y estructura de la imagen binaria

    Parámetros:
        binary_image (array): Una matriz que representa la imagen binaria.
        kernel_size (tuple): Tamaño del elemento estructurante para el cierre morfológico.

    Salidas:
        processed_image (array): La imagen binaria procesada morfológicamente.
    '''
    kernel = np.ones(kernel_size, np.uint8)                             # Crear un kernel para el cierre morfológico
    dilated_image = cv2.erode(binary_image, kernel, iterations=1)       # Erosionar la imagen
    processed_image = cv2.dilate(dilated_image, kernel, iterations=1) # Dilatar la imagen binaria para cerrar
    return processed_image


def connected_component_labeling(binary_image):
    '''
    Aplica el algoritmo de relleno de semillas para etiquetar componentes conectados en una imagen binaria.

    Parámetros:
        binary_image (array): Una matriz que representa la imagen binaria.

    Salidas:
        labeled_image (array): La imagen con componentes conectados etiquetados.
        num_labels (int): El número total de etiquetas utilizadas.
    '''

    labeled_image = np.zeros_like(binary_image)
    label = 1  # Inicializar la etiqueta con 1
    stack = []

    # Recorrer la imagen binaria
    for y in range(binary_image.shape[0]):
        for x in range(binary_image.shape[1]):
            if binary_image[y, x] == 0 or labeled_image[y, x] != 0:
                continue

            # Inicializar la pila con la semilla actual
            stack.append((x, y))

            # Mientras haya elementos en la pila
            while stack:
                current_x, current_y = stack.pop()

                # Verificar si el píxel actual es válido y si ya ha sido etiquetado
                if current_x < 0 or current_y < 0 or current_x >= binary_image.shape[1] or current_y >= binary_image.shape[0]:
                    continue
                if labeled_image[current_y, current_x] != 0 or binary_image[current_y, current_x] == 0:
                    continue

                # Asignar la etiqueta al píxel actual
                labeled_image[current_y, current_x] = label

                # Agregar los píxeles adyacentes válidos y no etiquetados a la pila
                for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                    new_x, new_y = current_x + dx, current_y + dy
                    if 0 <= new_x < binary_image.shape[1] and 0 <= new_y < binary_image.shape[0] and labeled_image[new_y, new_x] == 0:
                        stack.append((new_x, new_y))

            label += 1
    
    # Contar el número total de etiquetas utilizadas
    num_labels = label - 1

    return labeled_image, num_labels
