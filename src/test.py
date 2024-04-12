import os
import cv2
import sys

import numpy as np
import stuff_region_segmentation, image_preprocessing

def test_project(image_path):

    image = cv2.imread(image_path, 0)

    ######################################################## STUFF REGION SEGMENTATION ########################################################

    binary_image = stuff_region_segmentation.binary_transform(image)
    cv2.imwrite('testing/00_binary_transform_result.png', binary_image)
    print('\nBinary image succesfully created and saved')

    horizontal_sum = stuff_region_segmentation.horizontal_projection(binary_image)
    hist_image = stuff_region_segmentation.get_histogram_image(binary_image, horizontal_sum)
    cv2.imwrite('testing/01_horizontal_projection_result.png', hist_image)
    print('\nHorizontal sum succesfully created and saved')

    staff_lines = stuff_region_segmentation.region_segmentation(binary_image, horizontal_sum)
    cv2.imwrite('testing/02_region_segmentation_result.png', staff_lines)
    print('\nRegion segmentation succesfully created and saved')

    columns_with_lines = stuff_region_segmentation.get_black_column_positions(staff_lines)
    print('\nColumns with lines detected:', columns_with_lines)

    staff_lines_positions = stuff_region_segmentation.get_staff_lines_positions(columns_with_lines)
    print('\nStaff lines positions:', staff_lines_positions)

    all_staff_lines = []
    for staff in staff_lines_positions:
        for line in staff:
            all_staff_lines.append(line)
    result = np.zeros((staff_lines.shape[0], staff_lines.shape[1], 3), dtype=np.uint8)
    result[:, :, 0] = staff_lines
    result[:, :, 1] = staff_lines
    result[:, :, 2] = staff_lines
    for y in staff_lines_positions:
        result[y, :] = [0, 0, 255]
    cv2.imwrite('testing/03_detected_staff_lines_locations.png', result)
            

    ######################################################## IMAGE PREPROCESSING ########################################################

    image_without_lines = image_preprocessing.staff_line_filtering(binary_image, staff_lines)
    cv2.imwrite('testing/04_image_without_lines_result.png', image_without_lines)
    print('\nImage without lines succesfully created and saved')

    processed_image = image_preprocessing.morphological_processing(image_without_lines, (3, 3))
    #cv2.imwrite('testing/04_processed_image.png', processed_image)
    #print('\nProcessed image succesfully created and saved')

if __name__ == '__main__':
    test_project('images/Test Sheet 10.png')
