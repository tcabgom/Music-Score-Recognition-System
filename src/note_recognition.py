
from math import floor


notes_mapping = {
    0: "C",   # Do bajo
    1: "D",   # Re bajo
    2: "E",   # Mi bajo
    3: "F",   # Fa bajo
    4: "G",   # Sol bajo
    5: "A",   # La bajo
    6: "B",   # Si bajo
    7: "^C",  # Do alto
    8: "^D",  # Re alto
    9: "^E",  # Mi alto
    10: "^F",  # Fa alto
    11: "^G",  # Sol alto
    12: "^A",  # La alto
    13: "^B",  # Si alto
    14: "^^C",  # Do muy alto
    15: "^^D"   # Re muy alto
}



def get_head_size(staff_lines):
    pass


def stem_filtering(image):
    pass


# En el paper se llama size filtering
def head_filtering_v1(image, head_size, staffs_positions):
    pass


def head_filtering_v2(image, head_size, staffs_positions):
    pass


def shape_filtering(image):
    pass


def pitch_analysis_v1(note_head_positions, staff_lines_positions):
    '''
    Analiza las posiciones de las cabezas de notas en un pentagrama y determina las notas asociadas.

    Entradas:
        note_head_positions (list[tuple]): Una lista de tuplas que representan las posiciones (x, y) de las cabezas de las notas en el pentagrama.
        staff_lines_positions (list[tuple]): Una lista de tuplas que representan las posiciones (y, distancia) de las líneas del pentagrama, empezando desde la línea más baja.

    Salida:
        notes_pitch_dict (map): Un diccionario que a cada nota de note_head_positions le asigna una nota musical.
    '''
    notes_pitch = {}

    staff_lines_expected_area = []
    for staff_line in staff_lines_positions:
        y, distance = staff_line
        # Añadimos la posición minima y maxima de la nota en el pentagrama para poder detectar a cual de ellos pertenece
        staff_lines_expected_area.append((y - distance, y + distance*6))
    
    for note in note_head_positions:

        # PASO 1; Buscamos la linea de staff_line mas cercana a la nota
        _, note_y = note
        minimum_y_distance = None
        note_staff_list_position = None
        
        for i in range(len(staff_lines_expected_area)):
            y_distance = staff_lines_expected_area - note_y
            if y_distance is None or minimum_y_distance < minimum_y_distance:
                minimum_y_distance = y_distance
                note_staff_list_position = floor(i/2)

        # PASO 2: Calculamos el tono de la nota en base a la distancia relativa entre la primera linea del pentagrama y la nota
        note_staff_first_line, note_staff_line_distance = staff_lines_positions[note_staff_list_position]
        difference = ((note_staff_first_line + note_staff_line_distance) - note_y) / 2
        notes_pitch[note] = notes_mapping[difference]
    
    return notes_pitch
    


def beat_analysis(image):
    pass
