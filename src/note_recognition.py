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
    #[(328, 16), (549, 17), (769, 16), (990, 17), (1211, 16), (1432, 17), (1653, 16)]
    # Calculamos la distancia entre las lineas del pentagrama
    staff_distance = []
    staff_lines_distance = [staff_lines[0][1]]
    staff_gap = []
    for i in range(1, len(staff_lines)):
        distance = staff_lines[i][0] - staff_lines[i - 1][0]
        staff_distance.append(distance)
        staff_lines_distance.append(staff_lines[i][1])
        staff_gap.append(staff_lines[i - 1][0] + (staff_lines[i - 1][1]*5) + distance)
    # Calculamos la media de las distancias entre las lineas del pentagrama
    staff_distance = np.mean(staff_distance).round()
    lines_distance = np.mean(staff_lines_distance).round()
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


def stem_filtering(staff_images):
    '''
    Recorre la lista de imágenes de pentagrama y realiza una segmentación por regiones donde genera una imagen con los tallos de las notas.

    Parámetros:
        staff_images (array(imagen)): Un array con imágenes representando cada pentagrama.

    Salidas:
        stem_lines (imagen): Una imagen de los tallos de las notas.
    '''
    stem_lines = None
    for staff_index in range(0, len(staff_images)):
        hist = vertical_projection(cv2.bitwise_not(staff_images[staff_index])) #se invierte para que cuente los bits negros, no los blancos
        current_stem_lines = np.zeros_like(staff_images[staff_index])
        threshold = int(np.max(hist)*0.4)
        for x in range(current_stem_lines.shape[1]):
            if hist[x] <= threshold:
                cv2.line(current_stem_lines, (x, 0), (x, current_stem_lines.shape[0]), (255), 1)
        if staff_index == 0:
            stem_lines = current_stem_lines
        else:
            stem_lines = cv2.vconcat([stem_lines, current_stem_lines])
    return stem_lines


# En el paper se llama size filtering
def head_filtering_v1(image, head_size, staffs_positions):
    pass


def head_filtering_v2(image, head_size, staffs_positions):
    pass


def shape_filtering(image):
    pass


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
