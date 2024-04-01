import os
import cv2
import sys
import stuff_region_segmentation, image_preprocessing

def test_project(image_path):

    image = cv2.imread(image_path, 0)

    ######################################################## STUFF REGION SEGMENTATION ########################################################

    binary_image = stuff_region_segmentation.binary_transform(image)
    cv2.imwrite('testing/00_binary_image.png', binary_image)
    print('\nBinary image succesfully created and saved')

    horizontal_sum = stuff_region_segmentation.horizontal_projection(binary_image)
    cv2.imwrite('testing/01_horizontal_sum.png', horizontal_sum)
    print('\nHorizontal sum succesfully created and saved')

    staff_lines = stuff_region_segmentation.region_segmentation(horizontal_sum)
    cv2.imwrite('testing/02_region_segmentation.png', staff_lines)
    print('\nRegion segmentation succesfully created and saved')

    columns_with_lines = stuff_region_segmentation.get_black_column_positions(staff_lines)
    print('\nColumns with lines detected:', columns_with_lines)

    staff_lines_positions = stuff_region_segmentation.get_staff_lines_positions(columns_with_lines)
    print('\nStaff lines positions:', staff_lines_positions)

    ######################################################## IMAGE PREPROCESSING ########################################################

    image_without_lines = image_preprocessing.staff_line_filtering(binary_image, staff_lines)
    cv2.imwrite('testing/03_image_without_lines.png', image_without_lines)
    print('\nImage without lines succesfully created and saved')

    processed_image = image_preprocessing.morphological_processing(image_without_lines, (3, 3))
    #cv2.imwrite('testing/04_processed_image.png', processed_image)
    #print('\nProcessed image succesfully created and saved')

if __name__ == '__main__':
    test_project('images/Test Sheet 8.png')
