import os
import cv2
import sys
import numpy as np
import stuff_region_segmentation
import image_preprocessing

def test_project(image_path):

    image = cv2.imread(image_path, 0)

    ######################################################## STUFF REGION SEGMENTATION ########################################################

    binary_image = stuff_region_segmentation.binary_transform(image)
    cv2.imwrite('testing/00_binary_transform_result.png', binary_image)
    print('\n[STEP 01/XX] Binary image successfully created and saved')

    horizontal_sum = stuff_region_segmentation.horizontal_projection(binary_image)
    hist_image = stuff_region_segmentation.get_histogram_image(binary_image, horizontal_sum)
    cv2.imwrite('testing/01_horizontal_projection_result.png', hist_image)
    print('\nHorizontal sum successfully created and saved')

    staff_lines = stuff_region_segmentation.region_segmentation(binary_image, horizontal_sum)
    cv2.imwrite('testing/02_region_segmentation_result.png', staff_lines)
    print('\n[STEP 03/XX] Region segmentation successfully created and saved')

    columns_with_lines = stuff_region_segmentation.get_black_column_positions(staff_lines)
    print('\n[STEP 04/XX] Columns with lines detected:', columns_with_lines)

    staff_lines_positions, max_thickness = stuff_region_segmentation.get_staff_lines_positions_and_thickness(columns_with_lines)
    print('\n[STEP 05/XX] Staff lines positions:', staff_lines_positions)
    print('\n[STEP 05/XX] Maximum line thickness:', max_thickness)

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
    print('\n[STEP 06/XX] Image without lines successfully created and saved')

    processed_image = image_preprocessing.morphological_processing(image_without_lines, (5,5))
    cv2.imwrite('testing/05_processed_image.png', processed_image)
    print('\n[STEP 07/XX] Processed image successfully created and saved')

    #print('\nLabeling connected components. This process might take a while...')
    num_labels, labeled_image = image_preprocessing.connected_component_labeling(processed_image)
    print(num_labels)
    print(labeled_image)
    cv2.imwrite('testing/06_labeled_image.png', labeled_image)
    print('\n[STEP 08/XX] Labeled image successfully created and saved. Detected', num_labels, 'components')


if __name__ == '__main__':
    test_project('images/Test Sheet 10.png')

