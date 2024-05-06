import os
import shutil
import cv2
import sys
import stuff_region_segmentation
import image_preprocessing
import note_recognition
import accidental_and_rest_recognition


def test_project(tested_sheets, is_executed_on_notebook):
    for sheet in tested_sheets:
        # Extraer el nombre del archivo sin extensión
        sheet_name = os.path.splitext(os.path.basename(sheet))[0]
        # Crear el directorio para esta hoja si no existe
        sheet_dir = 'testing/' + sheet_name
        os.makedirs(sheet_dir, exist_ok=True)
        
        # Crear el archivo de registro en el directorio
        log_file_path = os.path.join(sheet_dir, 'log.txt')
        with open(log_file_path, 'w') as log_file:
            # Redirigir stdout al archivo de registro
            sys.stdout = log_file
            print("Archivo log para:", sheet)
            test_sheet(sheet, sheet_dir)
            # Restaurar stdout
            sys.stdout = sys.__stdout__


def test_sheet(image_path, base_dir):

    image = cv2.imread(image_path, 0)

    binary_image = stuff_region_segmentation.binary_transform(image)
    cv2.imwrite(f'{base_dir}/00_binary_transform_result.png', binary_image)

    horizontal_sum = stuff_region_segmentation.horizontal_projection(binary_image)
    hist_image = stuff_region_segmentation.get_histogram_image(binary_image, horizontal_sum)
    cv2.imwrite(f'{base_dir}/01_horizontal_projection_result.png', hist_image)

    staff_lines = stuff_region_segmentation.region_segmentation(binary_image, horizontal_sum)
    cv2.imwrite(f'{base_dir}/02_region_segmentation_result.png', staff_lines)

    columns_with_lines = stuff_region_segmentation.get_black_column_positions(staff_lines)
    staff_lines_v1, max_thickness = stuff_region_segmentation.get_staff_lines_positions_and_thickness(columns_with_lines)
    staff_lines_v2 = stuff_region_segmentation.get_staff_lines_positions_v2(columns_with_lines)

    print(f"\nStaff lines positions (v1): {staff_lines_v1}")
    print(f"\nStaff lines positions (v2): {staff_lines_v2}")
    print(f"\nMaximum line thickness: {max_thickness}")

    image_without_lines = image_preprocessing.staff_line_filtering(binary_image, staff_lines)
    cv2.imwrite(f'{base_dir}/03_image_without_lines_result.png', image_without_lines)

    processed_image = image_preprocessing.morphological_processing(image_without_lines, ((max_thickness+1),(max_thickness+1)))
    cv2.imwrite(f'{base_dir}/04_processed_image.png', processed_image)

    sizes = note_recognition.size_filtering(staff_lines_v2)
    print(f"\nSizes: {sizes}")

    staff_images, staff_boundaries = note_recognition.divide_staff_v2(processed_image, staff_lines_v2, sizes[2])
    print(f"\nStaff boundaries: {staff_boundaries}")

    num_labels, labels, stats, _ = image_preprocessing.connected_component_labeling(processed_image)
    cv2.imwrite(f'{base_dir}/05_labeled_image.png', labels)
    print(f"\nNumber of labels: {num_labels}")

    labels, bounding_boxes = accidental_and_rest_recognition.element_recognition(num_labels, labels, stats, True, is_executed_on_notebook)
    cv2.imwrite(f'{base_dir}/06_image_with_only_notes.png', labels)

    print(f"Detected elements: {accidental_and_rest_recognition.FIGURES_POSITIONS}")

    stem_lines, centers = note_recognition.stem_filtering_and_notes_positions(labels, bounding_boxes)
    cv2.imwrite(f'{base_dir}/07_note_heads.png', stem_lines)
    for i in range(6,9):
        centers = note_recognition.add_fulls_to_detected_notes(accidental_and_rest_recognition.FIGURES_POSITIONS[i], centers)
    print(f"\nNote head centers: {centers}")

    cv2.imwrite(f'{base_dir}/08_note_heads_and_staff_lines.png', cv2.bitwise_and(stem_lines, staff_lines))

    pitchs_v1 = note_recognition.pitch_analysis_v1(centers, staff_lines_v1)
    print(f"\nPitchs (v1): {pitchs_v1}")

    pitchs_v2 = note_recognition.pitch_analysis_v2(centers, staff_lines_v2)
    print(f"\nPitchs (v2): {pitchs_v2}")

    detected_notes_v1 = accidental_and_rest_recognition.detect_accidentals(centers, pitchs_v1)
    detected_notes_v2 = accidental_and_rest_recognition.detect_accidentals(centers, pitchs_v2)

    different_keys = []
    for key in detected_notes_v1:
        if key in detected_notes_v2 and detected_notes_v1[key] != detected_notes_v2[key]:
            different_keys.append(key)
    print(f"\nDifferent keys in both methods: {different_keys}")
    print(f"Difference percentage: {len(different_keys) / len(detected_notes_v1) * 100:.2f}%")

    result_v1 = note_recognition.draw_detected_notes_v1(binary_image, detected_notes_v1, staff_lines_v1)
    result_v2 = note_recognition.draw_detected_notes_v2(binary_image, detected_notes_v2, staff_lines_v2)

    cv2.imwrite(f'{base_dir}/09_detected_notes_result_v1.png', result_v1)
    cv2.imwrite(f'{base_dir}/10_detected_notes_result_v2.png', result_v2)

    # Se vuelve a definir para que no se mantenga entre ejecuciones
    accidental_and_rest_recognition.FIGURES_POSITIONS = list()
    for _ in range(9):
        accidental_and_rest_recognition.FIGURES_POSITIONS.append(list())


def delete_testing_folders():
    folder = 'testing'
    if os.path.exists(folder):
        shutil.rmtree(folder)
        print("Carpetas de pruebas eliminadas")
    else:
        print("No hay carpetas de pruebas para eliminar")


if __name__ == '__main__':
    
    tested_sheets = []
    for i in range(1,13):
        tested_sheets.append('images/Test Sheet ' + str(i) + '.png')
    
    is_executed_on_notebook = False # La ruta a los templates es diferente si se ejecuta desde un notebook
    delete_testing_folders()
    
    # Comentar esta linea para limpiar la carpeta de tests
    test_project(tested_sheets, is_executed_on_notebook)


