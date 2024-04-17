import os
import cv2
import sys
import numpy as np
import stuff_region_segmentation
import image_preprocessing
import note_recognition

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

    staff_lines_v2 = stuff_region_segmentation.get_staff_lines_positions_v2(columns_with_lines)
    print('\n[STEP 05/XX] Staff lines positions v2:', staff_lines_v2)

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

    processed_image = image_preprocessing.morphological_processing(image_without_lines, ((max_thickness+1),(max_thickness+1)))
    cv2.imwrite('testing/05_processed_image.png', processed_image)
    print('\n[STEP 07/XX] Processed image successfully created and saved')

    num_labels, labels, stats, _ = image_preprocessing.connected_component_labeling(processed_image)
    cv2.imwrite('testing/06_labeled_image.png', labels)
    print('\n[STEP 08/XX] Labeled image successfully created and saved. Detected', num_labels, 'components')

    ######################################################## NOTE RECOGNITION ########################################################

    staff_images = note_recognition.divide_staff(processed_image, staff_lines_positions)
    print('\n[STEP 09/XX] Staff images successfully created')
    for i in range(len(staff_images)):
        cv2.imwrite('testing/07_divide_staff_images/07_image_' + str(i) + '.png', staff_images[i])

    stem_lines = note_recognition.stem_filtering(staff_images)
    print('\n[STEP 10/XX] Stem lines successfully created')
    for i in range(len(stem_lines)):
        cv2.imwrite('testing/08_stem_filtering_images/08_image_' + str(i) + '.png', stem_lines[i])

    sizes = note_recognition.size_filtering(staff_lines_v2)
    print('\n[STEP 11/XX] Sizes successfully created', "\n", sizes)

    staff_images = note_recognition.divide_staff_v2(processed_image, staff_lines_v2, sizes[2])
    print('\n[STEP 09_alt/XX] Staff images successfully created')
    for i in range(len(staff_images)):
        cv2.imwrite('testing/07_divide_staff_images_v2/07_image_' + str(i) + '.png', staff_images[i])

    note_head_centers = note_recognition.shape_filtering(sizes[0], num_labels, labels, stats)
    print('\n[STEP 13/XX] Note head centers successfully created', "\n", note_head_centers)

if __name__ == '__main__':
    test_project('images/Test Sheet 3.png')

