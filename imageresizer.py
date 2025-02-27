import os
import cv2

def process_and_resize_images(input_folder="faces", output_folder="04_resized_faces"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        
        image = cv2.imread(file_path)
        if image is None:
            print(f"Skipping {filename}, not a valid image.")
            continue

        height, width = image.shape[:2]

        if height < 96 or width < 96:
            print(f"Removing {filename}, size {width}x{height} is too small.")
            os.remove(file_path)
            continue

        resized_image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

        output_path = os.path.join(output_folder, filename)
        cv2.imwrite(output_path, resized_image)
        print(f"Saved resized image to {output_path}")


# process_and_resize_images("03_filtered_faces", "04_resized_faces")
process_and_resize_images("my_photos", "my_photos_resized")
