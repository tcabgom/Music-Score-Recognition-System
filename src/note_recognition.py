import cv2
import numpy as np
import matplotlib as plt
from stuff_region_segmentation import get_histogram_image
from math import floor


notes_mapping = {
    0: "C",  # Do bajo
    1: "D",  # Re bajo
    2: "E",  # Mi bajo
    3: "F",  # Fa bajo
    4: "G",  # Sol bajo
    5: "A",  # La bajo
    6: "B",  # Si bajo
    7: "^C",  # Do alto
    8: "^D",  # Re alto
    9: "^E",  # Mi alto
    10: "^F",  # Fa alto
    11: "^G",  # Sol alto
    12: "^A",  # La alto
    13: "^B",  # Si alto
    14: "^^C",  # Do muy alto
    15: "^^D",  # Re muy alto
}


def size_filtering(staff_lines):
    staff_distance = []
    staff_lines_distance = [staff_lines[0][1]]
    staff_gap = []

    if len(staff_lines) == 1:
        return staff_lines[0][1], 0, staff_lines[0][1]*4
    
    for i in range(1, len(staff_lines)):
        distance = staff_lines[i][0] - staff_lines[i - 1][0]
        staff_distance.append(distance)
        staff_lines_distance.append(staff_lines[i][1])
    staff_distance = int(np.mean(staff_distance).round())
    lines_distance = int(np.mean(staff_lines_distance).round())
    staff_gap = int((staff_distance - lines_distance * 5)//2)
    return lines_distance, staff_distance, staff_gap


def vertical_projection(binary_image):
    normalized_image = binary_image / 255
    vertical_sum = np.sum(normalized_image, axis=0)

    return vertical_sum


def divide_staff(binary_image, staff_lines_positions):
    '''
    Recorre la lista de posiciones de líneas del pentagrama y genera una imagen para cada pentagrama recortando la imagen completa
    en las coordenadas precisas.

    Parámetros:
        image_without_lines (imagen): Una imagen binaria con pentagramas sin líneas horizontales.
        staff_lines_positions array((array)): Un vector de vectores que representa las posiciones de las líneas del pentagrama.

    Salidas:
        staff_images (array(imagen)): Un array con imágenes representando cada pentagrama.
    '''
    staff_images = []
    num_of_staff = len(staff_lines_positions)
    image_length = binary_image.shape[0]

    for stove_index in range(0,num_of_staff):
        current_stove = staff_lines_positions[stove_index]
        next_stove = (staff_lines_positions[stove_index + 1] if stove_index != num_of_staff - 1 else [image_length] * 5)
        previous_stove = staff_lines_positions[stove_index - 1] if stove_index != 0 else [0] * 5
        margin_next = int(abs(current_stove[4] - next_stove[0]) / 2) if stove_index != num_of_staff - 1 else abs(current_stove[4] - image_length)
        margin_previous = int(abs(current_stove[0] - previous_stove[4]) / 2 ) if stove_index != 0 else current_stove[0]
        current_stove_image = binary_image[current_stove[0] - margin_previous : current_stove[4] + margin_next,:]
        staff_images.append(current_stove_image)

    return staff_images


def divide_staff_v2(image_without_lines, staff_lines, staff_gap):
    staff_images = []
    staff_boundaries = []

    # Iterar sobre las posiciones de las líneas de los pentagramas
    for start_y, line_distance in staff_lines:
        # Calcular las coordenadas de inicio y fin del pentagrama con el margen adicional
        start_x = 0
        end_x = image_without_lines.shape[1]
        end_y = int(start_y + (line_distance * 5) + staff_gap)  # Convertir a entero

        # Ajustar start_y para incluir el margen superior
        start_y -= staff_gap
        
        # Asegurarse de que start_y no sea negativo
        start_y = int(max(0, start_y))

        # Recortar la región de interés de la imagen
        staff_image = image_without_lines[int(start_y):end_y, start_x:end_x]  # Convertir a entero

        # Agregar la imagen del pentagrama a la lista
        staff_images.append(staff_image)
        staff_boundaries.append((start_y, end_y))  # Guardar las coordenadas de inicio y fin

    return staff_images, staff_boundaries


def stem_filtering(staff_images):
    '''
    Recorre la lista de imágenes de pentagrama y realiza una segmentación por regiones donde genera una imagen con los tallos de las notas.

    Parámetros:
        staff_images (array(imagen)): Un array con imágenes representando cada pentagrama.

    Salidas:
        stem_lines (imagen): Una imagen de los tallos de las notas.
    '''
    stem_lines = []

    kernel = np.ones(3, np.uint8)  

    for staff_index in range(len(staff_images)):
        hist = vertical_projection(cv2.bitwise_not(staff_images[staff_index]))
        current_stem_lines = staff_images[staff_index].copy()  # Make a copy to avoid modifying the original image
        threshold = int(np.max(hist) * 0.6)
        for x in range(current_stem_lines.shape[1]):
            if hist[x] >= threshold:
                # Remove stem pixels in the current column
                current_stem_lines[:, x] = 255          # Crear un kernel para el cierre morfológico
        eroded_image_1 = cv2.erode(current_stem_lines, kernel, iterations=2)
        dilated_image_1 = cv2.dilate(eroded_image_1, kernel, iterations=2)
        dilated_image_2= cv2.dilate(dilated_image_1, kernel, iterations=2)
        eroded_image_2 = cv2.erode(dilated_image_2, kernel, iterations=2)
        stem_lines.append(eroded_image_2)
    return stem_lines


# En el paper se llama size filtering
def head_filtering_v1(image, head_size, staffs_positions):
    pass


def head_filtering_v2(image, head_size, staffs_positions):
    pass


def shape_filtering(note_head_size, binary_image):
    note_head_centers = []

    inverted_image = cv2.bitwise_not(binary_image)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted_image)

    # Iterate through connected components
    for label in range(1, num_labels):  # Start from 1 to exclude background label
        # Get the bounding box coordinates of the connected component
        x, y, w, h, _ = stats[label]

        # Calculate the center of the connected component
        center = (x + w // 2, y + h // 2)

        # Calculate the rate of symmetry
        sum_r = 0
        sum_s = 0
        for i in range(x, x + w):
            for j in range(y, y + h):
                if labels[j, i] == label:
                    # Calculate distance from center
                    d = np.sqrt((i - center[0])**2 + (j - center[1])**2)
                    # Calculate symmetric point
                    symmetric_point = (center[0] - (i - center[0]), center[1] - (j - center[1]))
                    # Check if symmetric point is within bounds
                    if symmetric_point[0] >= x and symmetric_point[0] < x + w and symmetric_point[1] >= y and symmetric_point[1] < y + h:
                        if labels[symmetric_point[1], symmetric_point[0]] == label:
                            # Increment sum_s if symmetric point is part of the connected component
                            sum_s += 1
                            # Increment sum_r if the point satisfies the symmetry condition
                            if abs(d - note_head_size) < 2:
                                sum_r += 1

        # Calculate rate of symmetry
        if sum_s != 0:
            rate_of_symmetry = sum_r / sum_s
        else:
            rate_of_symmetry = 0

        # Threshold for considering as a note head
        if rate_of_symmetry > 0.5:
            note_head_centers.append(center)

    return note_head_centers


def pitch_analysis_v1(note_head_positions, staff_lines_positions):
    """
    Analiza las posiciones de las cabezas de notas en un pentagrama y determina las notas asociadas.

    Entradas:
        note_head_positions (list[tuple]): Una lista de tuplas que representan las posiciones (x, y) de las cabezas de las notas en el pentagrama.
        staff_lines_positions (list[tuple]): Una lista de tuplas que representan las posiciones (y, distancia) de las líneas del pentagrama, empezando desde la línea más baja.

    Salida:
        notes_pitch_dict (map): Un diccionario que a cada nota de note_head_positions le asigna una nota musical.
    """
    notes_pitch = {}

    staff_lines_expected_area = []
    for staff_line in staff_lines_positions:
        y, distance = staff_line
        # Añadimos la posición minima y maxima de la nota en el pentagrama para poder detectar a cual de ellos pertenece
        staff_lines_expected_area.append((y - distance, y + distance * 6))

    for note in note_head_positions:

        # PASO 1; Buscamos la linea de staff_line mas cercana a la nota
        _, note_y = note
        minimum_y_distance = None
        note_staff_list_position = None

        for i in range(len(staff_lines_expected_area)):
            y_distance = staff_lines_expected_area - note_y
            if y_distance is None or minimum_y_distance < minimum_y_distance:
                minimum_y_distance = y_distance
                note_staff_list_position = floor(i / 2)

        # PASO 2: Calculamos el tono de la nota en base a la distancia relativa entre la primera linea del pentagrama y la nota
        note_staff_first_line, note_staff_line_distance = staff_lines_positions[
            note_staff_list_position
        ]
        difference = ((note_staff_first_line + note_staff_line_distance) - note_y) / 2
        notes_pitch[note] = notes_mapping[difference]

    return notes_pitch


def beat_analysis(image):
    pass
