

import cv2
import numpy as np

TEMPLATES = ["Flat.png", "Sharp.png", "Natural.png", "Treble_Clef.png", "Quarter_Rest.png", "Eighth_Rest.png", "Sixteenth_Rest.png", "Compass_Number.png"]
TEMPLATE_THRESHOLD = 0.75


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
    for i in range(1, num_labels):
        
        # Calculate the bounding box coordinates of the connected component
        x, y, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        bounding_box = labels[y:y+h, x:x+w]     # Crop the region of interest from the image
        bounding_box[bounding_box != 0] = 255   # Convert the cropped image to binary
        
        found = False
        for template in TEMPLATES:

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
                # Calcular la nueva posición (x, y) en la imagen original
                new_x = x + int((w - bounding_box.shape[1]) / 2)
                new_y = y + int((h - bounding_box.shape[0]) / 2)
                # Borrar el elemento de la imagen original
                labels[new_y:new_y+bounding_box.shape[0], new_x:new_x+bounding_box.shape[1]][bounding_box != 0] = 0
                break  # Salir del bucle una vez que se encuentra una coincidencia
        if returns_binary:
            if not found:
                bounding_box[bounding_box != 0] = i
            else:
                bounding_box[bounding_box != 0] = 255

    if not returns_binary:
        labels = np.uint8(labels)
        _, labels = cv2.threshold(labels, 10, 255, cv2.THRESH_BINARY)
        labels = cv2.bitwise_not(labels)

    return labels
