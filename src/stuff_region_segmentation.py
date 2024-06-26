import cv2
import numpy as np


BINARY_TRANFORM_THRESHOLD = 127
REGION_SEGMENTATION_RATIO = 0.7


##############################################   FUNCIONES   ##############################################


def binary_transform(image):
    '''
    Realiza una transformación binaria en una imagen dada.

    Entradas:
        image (array): Una matriz que representa la imagen de entrada.

    Salidas:
        binary_image (array): La imagen transformada binariamente.
    '''
    _, binary_image = cv2.threshold(image, BINARY_TRANFORM_THRESHOLD, 255, cv2.THRESH_BINARY)
    return binary_image[:,:,1]


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
    
    for y, value in enumerate(horizontal_sum_normalized):                              # Itera sobre la suma horizontal para dibujar las líneas de la proyección horizontal
        if np.isscalar(value):
            value = int(value)
        else:
            value = int(value[0])
        cv2.line(horizontal_image, (width - value - 1, y), (width, y), 255, 1)     # El -1 es para que no se dibuje una línea en la primera columna de la imagen

    return horizontal_image


def region_segmentation(image):
    '''
    Realiza la segmentación de regiones para detectar las líneas de las partituras en una imagen binaria.

    Parámetros:
        image (array): Una histograma de proyección horizontal.

    Salidas:
        staff_lines: Un array que contiene las líneas de las partituras sin notas ni otros símbolos.
    '''
    threshold = round(image.shape[1] * REGION_SEGMENTATION_RATIO)       # Calcula el pixel umbral para la segmentación de regiones
    staff_lines = np.zeros_like(image)                                  # Crea una matriz de ceros con las mismas dimensiones que la imagen original
    
    for y in range(image.shape[0]):                                     # Itera sobre las filas de la imagen
        if image[y, threshold] == 255:                                  # Si el píxel en la posición (threshold, y) es blanco, no dibujar linea
            staff_lines[y, :] = 255
        else:                                                           # Si el píxel en la posición (threshold, y) es negro, dibujar linea
            staff_lines[y, :] = 0

    return staff_lines


def get_black_column_positions(image):
    '''
    Obtiene las posiciones de las columnas negras en la imagen.

    Parámetros:
        image (array): Una matriz que representa la imagen.

    Salidas:
        black_column_positions (list): Una lista que contiene las posiciones de las columnas negras.
    '''
    black_column_positions = []

    for y in range(image.shape[0]):
        is_black_column = image[y, 1] == 0
        if is_black_column:
            black_column_positions.append(y)

    return black_column_positions


def get_staff_lines_positions(black_column_positions):
    '''
    Obtiene las posiciones medias de cada línea de pentagrama en la imagen.

    Parámetros:
        black_column_positions (list): Una lista que contiene las posiciones de las columnas negras.

    Salidas:
        staffs (list[list]): Una lista de listas que contiene las posiciones medias de cada línea de pentagrama.
    '''
    staffs = []
    current_staff = []
    current_staff_line = 0

    started_position = black_column_positions[0]

    for i in range(1, len(black_column_positions)):
        if (black_column_positions[i] - black_column_positions[i - 1]) > 1 or i == len(black_column_positions) - 1:
            
            current_staff.append(round((started_position + black_column_positions[i - 1]) / 2))
            current_staff_line += 1

            if current_staff_line == 5:
                staffs.append(current_staff)
                current_staff = []
                current_staff_line = 0

            started_position = black_column_positions[i]

    return staffs


def get_staff_lines_positions_v2(black_column_positions):
    '''
    Una alternativa de la función alterior donde solo se retorna la lista de primeras lineas del pentagrama y la media de 
    distancia entre las lineas

    Parámetros:
        black_column_positions (list): Una lista que contiene las posiciones de las columnas negras.

    Salidas:
        staffs (list, int): Una lista de listas que contiene las posiciones medias de cada línea de pentagrama.
    '''
    pass
