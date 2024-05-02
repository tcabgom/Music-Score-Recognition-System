import cv2
import numpy as np
import matplotlib as plt
from accidental_and_rest_recognition import FIGURES_POSITIONS
from image_preprocessing import connected_component_labeling, morphological_processing
from stuff_region_segmentation import get_histogram_image
from math import floor


notes_mapping = {
    0: "C4",  # Do bajo
    1: "D4",  # Re bajo
    2: "E4",  # Mi bajo
    3: "F4",  # Fa bajo
    4: "G4",  # Sol bajo
    5: "A4",  # La bajo
    6: "B4",  # Si bajo
    7: "C5",  # Do alto
    8: "D5",  # Re alto
    9: "E5",  # Mi alto
    10: "F5",  # Fa alto
    11: "G5",  # Sol alto
    12: "A5",  # La alto
    13: "B5",  # Si alto
    14: "C6",  # Do muy alto
    15: "D6",  # Re muy alto
}


BEAT_MAPPING = {
    0: "1",  # Semicorchea
    1: "2",  # Corchea
    3: "4",  # Negra
    4: "8",  # Blanca
    5: "16", # Redonda
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
    Recorre la lista de imágenes de pentagrama y realiza una segmentación por regiones donde genera una imagen sin los tallos de las notas.

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
        current_stem_lines = current_stem_lines.astype(np.uint8)
        eroded_image_1 = cv2.erode(current_stem_lines, kernel, iterations=2)
        dilated_image_1 = cv2.dilate(eroded_image_1, kernel, iterations=2)
        dilated_image_2= cv2.dilate(dilated_image_1, kernel, iterations=2)
        eroded_image_2 = cv2.erode(dilated_image_2, kernel, iterations=2)
        stem_lines.append(eroded_image_2)
    return stem_lines

def extract_bounding_boxes(binary_image):

    # Encontrar componentes conexas y sus estadísticas
    num_labels, labels, stats, centroids = connected_component_labeling(binary_image)
    bounding_boxes = []
    
    for i in range(1, num_labels):
        
        # Calculate the bounding box coordinates of the connected component
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        bounding_boxes.append((x, y, w, h))
    return bounding_boxes

'''
def stem_filtering_on_bounding_boxes(image, bounding_boxes=None):
    if bounding_boxes is None:
        bounding_boxes = extract_bounding_boxes(image)
    combined_filtered_image = np.ones_like(image) * 255  # Crear una imagen en blanco del mismo tamaño que la original
    for bbox in bounding_boxes:
        # Extraer la región de interés (ROI) de la imagen original basada en la bounding box
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        # Aplicar stem_filtering a la ROI
        filtered_roi = stem_filtering([roi])[0]  # La función stem_filtering devuelve una lista, por lo tanto, tomamos el primer elemento
        # Superponer la imagen filtrada en la imagen combinada
        combined_filtered_image[y:y+h, x:x+w] = filtered_roi
    return combined_filtered_image
'''


def stem_filtering_on_bounding_boxes(image, bounding_boxes=None):
    if bounding_boxes is None:
        bounding_boxes = extract_bounding_boxes(image)
    
    combined_filtered_image = np.ones_like(image) * 255  # Crear una imagen en blanco del mismo tamaño que la original
    for bbox in bounding_boxes:
        # Extraer la región de interés (ROI) de la imagen original basada en la bounding box
        x, y, w, h = bbox
        roi = image[y:y+h, x:x+w]
        # Aplicar stem_filtering a la ROI
        filtered_roi = stem_filtering([roi])[0]  # La función stem_filtering devuelve una lista, por lo tanto, tomamos el primer elemento
        # Superponer la imagen filtrada en la imagen combinada
        combined_filtered_image[y:y+h, x:x+w] = filtered_roi
    combined_filtered_image
    return combined_filtered_image

# Condición para filtrar bounding boxes basada en el ancho más del doble de la altura
def aspect_ratio_condition_horizontal(bbox, threshold = 1.4):
    x, y, w, h = bbox
    aspect_ratio = w / h
    return aspect_ratio >= threshold

def aspect_ratio_condition_vertical(bbox, threshold = 1.8):
    x, y, w, h = bbox
    aspect_ratio = h / w
    return aspect_ratio >= threshold


# Función para eliminar componentes conexas que no cumplen con la condición
def remove_components_and_find_notes(image, bounding_boxes, clean_image=False):
    areas = []
    centers = []
    bounding_boxes_aux = []
    for bbox in bounding_boxes:
        x, y, w, h = bbox

        if aspect_ratio_condition_horizontal(bbox):
            image[y:y+h, x:x+w] = 255  # Rellenar con blanco la región de la bounding box
        elif aspect_ratio_condition_vertical(bbox, 1.8):
            image[y:y+h, x:x+w] = 255
        else:
            if not clean_image:
                image[y:y+h, x:x+w] = 0

            area = w * h
            areas.append(area)
            bounding_boxes_aux.append(bbox)
    
    if areas:
        mean_area = sum(areas) / len(areas)
        areas2 = []
        bounding_boxes_aux2 = []
        # Eliminar bounding boxes con área menor a la mitad de la media
        for i, bbox in enumerate(bounding_boxes_aux):
            x, y, w, h = bbox
            if areas[i] < mean_area / 2:
                image[y:y+h, x:x+w] = 255
            if areas[i] > mean_area * 1.6 and aspect_ratio_condition_horizontal(bbox, 1.17):
                image[y:y+h, x:x+w] = 255
            else:
                area = w * h
                areas2.append(area)
                bounding_boxes_aux2.append(bbox)

        
        mean_area = sum(areas2) / len(areas2)
        for i, bbox in enumerate(bounding_boxes_aux2):
            x, y, w, h = bbox
            if areas2[i] < mean_area / 1.6:
                image[y:y+h, x:x+w] = 255
            else:
                if not clean_image:
                    image[y:y+h, x:x+w] = 0
                x_center = x + w // 2
                y_center = y + h // 2
                centers.append((x_center, y_center))

    return image, centers, 0

def stem_filtering_and_notes_positions(image, bounding_boxes, clean_image=False):

    image = stem_filtering_on_bounding_boxes(image, bounding_boxes)
    bounding_boxes = extract_bounding_boxes(image)

    # Eliminar componentes conexas que no cumplen con la condición
    final_image, centers, fulls = remove_components_and_find_notes(image, bounding_boxes, clean_image)
    return final_image, centers, fulls

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
        center = (y + h // 2, x + w // 2)

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


def add_fulls_to_detected_notes(fulls, note_head_positions):
    for full in fulls:
        note_head_positions.append((round(full[0]+full[2]*0.5),round(full[1]+full[3]*0.5)))
    return note_head_positions


def pitch_analysis_v1(note_head_positions, staff_lines_positions):
    '''
    
    '''
    note_pitch = {}

    for note in note_head_positions:
        _, note_y = note
        # TODO Primero voy a hacerlo que detecte las notas dentro del pentagrama, adaptar más tarde
        note_staff = 0
        note_staff_found = False
        
        # Detecta el pentagrama al que pertenece la nota
        while not note_staff_found:
            if note_y < staff_lines_positions[note_staff][4] or note_staff == len(staff_lines_positions) - 1:
                note_staff_found = True
            else:
                note_staff += 1
                
        # Detecta la posición de la nota dentro del pentagrama
        potential_positions = []
        
        # Añade las posiciones de las notas por debajo de la partitura
        staff_lines_distance = staff_lines_positions[note_staff][4] - staff_lines_positions[note_staff][3]
        potential_positions.append(staff_lines_positions[note_staff][4]+staff_lines_distance)
        potential_positions.append(staff_lines_positions[note_staff][4]+staff_lines_distance/2)

        # Añade las posiciones de las notas dentro de la partitura
        for i in range(4,0,-1):
            potential_positions.append(staff_lines_positions[note_staff][i])
            potential_positions.append(staff_lines_positions[note_staff][i] + (staff_lines_positions[note_staff][i-1] - staff_lines_positions[note_staff][i]) / 2)
        potential_positions.append(staff_lines_positions[note_staff][0])
        

        # Añade las posiciones de las notas por encima de la partitura
        for i in range(1,6):
            potential_positions.append(staff_lines_positions[note_staff][0] - staff_lines_distance * i*0.5)

        # Busca la posición más cercana a la nota seleccionando el indice del valor mas cercano
        selected_position = min(range(len(potential_positions)), key=lambda i: abs(potential_positions[i]-note_y))
        note_pitch[note] = notes_mapping[selected_position]

    return note_pitch
        



def pitch_analysis_v2(note_head_positions, staff_lines_positions):
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
        staff_lines_expected_area.append(y - distance)
        staff_lines_expected_area.append(y + distance * 6)


    for note in note_head_positions:

        # PASO 1; Buscamos la linea de staff_line mas cercana a la nota
        _, note_y = note
        minimum_y_distance = None
        note_staff_list_position = None

        for i in range(len(staff_lines_expected_area)):
            y_distance = abs(staff_lines_expected_area[i] - note_y)
            if minimum_y_distance is None or minimum_y_distance > y_distance:
                minimum_y_distance = y_distance
                note_staff_list_position = floor(i / 2)
                

        # PASO 2: Calculamos el tono de la nota en base a la distancia relativa entre la primera linea del pentagrama y la nota
        note_staff_first_line, note_staff_line_distance = staff_lines_positions[note_staff_list_position]

        closest_note = None
        closest_distance = None
        for i in range(16):
            distance = abs(((note_staff_first_line + note_staff_line_distance*5)-note_staff_line_distance*0.5*i) - note_y)
            if closest_distance == None or closest_distance > distance:
                closest_note = i
                closest_distance = distance
        notes_pitch[note] = notes_mapping[closest_note]

    return notes_pitch

#SIN TERMINAR (WIP)
def beat_analysis(processed_image, detected_notes, bounding_boxes=None):
    ROUNDNESS_PROPORTION_THRESHOLD=0.7 #Si las proporciones son menores, se considera rectángulo (1 sería cuadrado)
    detected_notes = detected_notes.copy()
    bbox_images_by_center = {}
    if bounding_boxes is None:
        bounding_boxes = extract_bounding_boxes(processed_image)
    for center in detected_notes.keys(): #detected_notes es un diccionario de la forma {centro:nota}
        for bbox in bounding_boxes:
            x, y, w, h = bbox
            #Si el centro está en la bounding box
            if center[0] in range(x, x + w) and center[1] in range(y, y + h):
                bbox_image = processed_image[y:y+h, x:x+w]
                filtered_bbox_image = stem_filtering([bbox_image])[0]
                #Hacemos una operación morfológica vertical (7x4) para cerrar bien el proceso de stem_filtering
                kernel = np.ones((7,4), np.uint8)                           
                dilated_image = cv2.erode(filtered_bbox_image, kernel, iterations=1)     
                filtered_bbox_image = cv2.dilate(dilated_image, kernel, iterations=1)
                bbox_images_by_center[center]=filtered_bbox_image
                #Por cada componente dentro de la bounding box analizamos sus proporciones y contamos rectángulos
                num_of_rectangles = 0
                num_labels, _, stats, _ = connected_component_labeling(filtered_bbox_image)
                for label_index in range(1, num_labels): #El rango es (1,N) para no contar el fondo
                    _, _, width, height, _ = stats[label_index]
                    if min(width,height)/max(width,height) < ROUNDNESS_PROPORTION_THRESHOLD: #Comprobamos si tiene proporciones de rectángulo
                        num_of_rectangles += 1
                note = detected_notes[center]
                if len(note) <= 2:
                    detected_notes[center] = note + ":" + str(num_of_rectangles)

    return bounding_boxes, bbox_images_by_center, detected_notes


def draw_detected_notes_v1(sheet, detected_notes, staff_lines):
    draw_note_y = [line[4] + (line[4] - line[3]) * 3 for line in staff_lines]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    color = (0, 0, 255)
    thickness = 2

    sheet_color = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
    for note in dict.keys(detected_notes):
        note_x, note_y = note
        pitch_and_beat = detected_notes[note]

        # Buscar las líneas que están debajo de la nota
        lines_below_note = [line for line in draw_note_y if line > note_y]

        if lines_below_note:  # Si hay líneas debajo
            # Encontrar la línea más cercana debajo de la nota
            closest_y = min(lines_below_note, key=lambda y: abs(y - note_y) if y > note_y else float('inf'))
            # Usar esa posición como note_y
            note_y = closest_y

            # Obtener el tamaño del texto para centrarlo
            text_size, _ = cv2.getTextSize(pitch_and_beat, font, scale, thickness)
            # Calcular las coordenadas del centro del texto
            text_x = note_x - text_size[0] // 2
            text_y = note_y + text_size[1] // 2

            coordinates = (text_x, text_y)

            cv2.putText(sheet_color, pitch_and_beat, coordinates, font, scale, color, thickness)

    return sheet_color


def draw_detected_notes_v2(sheet, detected_notes, staff_lines):
    draw_note_y = [staff[0]+staff[1]*7 for staff in staff_lines]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    color = (0, 0, 255)
    thickness = 2

    sheet_color = cv2.cvtColor(sheet, cv2.COLOR_GRAY2BGR)
    for note in dict.keys(detected_notes):
        note_x, note_y = note
        pitch_and_beat = detected_notes[note]

        # Buscar las líneas que están debajo de la nota
        lines_below_note = [line for line in draw_note_y if line > note_y]

        if lines_below_note:  # Si hay líneas debajo
            # Encontrar la línea más cercana debajo de la nota
            closest_y = min(lines_below_note, key=lambda y: abs(y - note_y) if y > note_y else float('inf'))
            # Usar esa posición como note_y
            note_y = closest_y

            # Obtener el tamaño del texto para centrarlo
            text_size, _ = cv2.getTextSize(pitch_and_beat, font, scale, thickness)
            # Calcular las coordenadas del centro del texto
            text_x = note_x - text_size[0] // 2
            text_y = note_y + text_size[1] // 2

            coordinates = (text_x, text_y)

            cv2.putText(sheet_color, pitch_and_beat, coordinates, font, scale, color, thickness)

    return sheet_color