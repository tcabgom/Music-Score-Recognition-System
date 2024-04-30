

import cv2
import numpy as np

TEMPLATES = ["Sharp.png", "Flat.png", "Natural.png", "Quarter_Rest.png", "Eighth_Rest.png", "Sixteenth_Rest.png", "Treble_Clef.png", "Compass_Number.png"]
TEMPLATE_THRESHOLD = 0.75


FIGURES_POSITIONS = list()
for i in range(6):
    FIGURES_POSITIONS.append(list())


def template_matching(binary_image, template):
    # Convert images to required data types if necessary
    if binary_image.dtype != 'uint8':
        binary_image = (binary_image * 255).astype('uint8')

    if template.dtype != 'uint8':
        template = (template * 255).astype('uint8')

    # Perform template matching
    result = cv2.matchTemplate(binary_image, template, cv2.TM_CCOEFF_NORMED)
    normalized_result = (result + 1) / 2

    return normalized_result


def element_recognition(num_labels, labels, stats, returns_binary=False):
    bounding_boxes = []
    
    for i in range(1, num_labels):
        
        # Calculate the bounding box coordinates of the connected component
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        bounding_box = labels[y:y+h, x:x+w]     # Crop the region of interest from the image
        bounding_box[bounding_box != 0] = 255   # Convert the cropped image to binary
        bounding_boxes.append((x, y, w, h))
        found = False
        for j in range(len(TEMPLATES)):
            template = TEMPLATES[j]
            template_image = cv2.imread(f'templates/{template}', cv2.IMREAD_GRAYSCALE)

            # Calculate the scaling factor for resizing the bounding box
            scale_factorY = template_image.shape[0] / bounding_box.shape[0]
            scale_factorX = template_image.shape[1] / bounding_box.shape[1]

            # Resize the bounding box to match the size of the template
            resized_bounding_box = cv2.resize(bounding_box, None, fx=scale_factorX, fy=scale_factorY, interpolation=cv2.INTER_NEAREST)
            resized_bounding_box = cv2.bitwise_not(resized_bounding_box)

            # Perform template matching on the resized bounding box
            result = template_matching(resized_bounding_box, template_image)
            #cv2.imwrite('testing/07_divide_staff_images/a.png', resized_bounding_box)
            cv2.imwrite('testing/07_divide_staff_images/b.png', template_image)
            # Check if result is greater than threshold
            if np.max(result) > TEMPLATE_THRESHOLD:
                found = True
                bounding_boxes.pop(-1)
                # Calcular la nueva posición (x, y) en la imagen original
                new_x = x + int((w - bounding_box.shape[1]) / 2)
                new_y = y + int((h - bounding_box.shape[0]) / 2)
                # Borrar el elemento de la imagen original
                labels[new_y:new_y+bounding_box.shape[0], new_x:new_x+bounding_box.shape[1]][bounding_box != 0] = 0
                if j < 6:
                    FIGURES_POSITIONS[j].append((new_x, new_y, w, h))

                break  # Salir del bucle una vez que se encuentra una coincidencia
        if not returns_binary:
            if not found:
                bounding_box[bounding_box != 0] = i
            else:
                bounding_box[bounding_box != 0] = 255

    if returns_binary:
        labels = np.uint8(labels)
        _, labels = cv2.threshold(labels, 10, 255, cv2.THRESH_BINARY)
        labels = cv2.bitwise_not(labels)
    print(FIGURES_POSITIONS)
    return labels, bounding_boxes


def detect_accidentals(note_positions):
    for note in note_positions:
        note.append("0")

    for figure_list in range(3):
        for i in figure_list:
            distances = []
            for note_position in note_positions:
                distance = np.sqrt((i[0] - note_position[0])**2 + (i[1] - note_position[1])**2)
                distances.append(distance)
            closest_note_index = np.argmin(distances)
            note_positions.replace(note_positions[closest_note_index], (note_positions[closest_note_index][0], note_positions[closest_note_index][1], i))
    
    return note_positions
