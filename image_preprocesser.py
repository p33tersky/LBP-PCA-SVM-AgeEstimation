import os
import cv2

def process_and_save_faces(input_folder="all_faces", output_root="processed_faces"):
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    race_mapping = {"0": "white", "1": "black", "2": "asian", "3": "indian"}
    gender_mapping = {"0": "men", "1": "women"}

    file_list = os.listdir(input_folder)
    total_files = len(file_list)

    for i, filename in enumerate(file_list):
        print(f"Processing {total_files - i} images remaining...")

        file_path = os.path.join(input_folder, filename)
        parts = filename.split('_')

        if len(parts) < 4 or not parts[0].isdigit() or parts[1] not in gender_mapping or parts[2] not in race_mapping:
            print(f"Skipping {filename}, incorrect format.")
            continue

        age = parts[0]
        gender = gender_mapping[parts[1]]
        race = race_mapping[parts[2]]

        output_folder = os.path.join(output_root, f"{race}_{gender}", "resized_faces")
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, filename)
        
        if os.path.exists(output_path):
            print(f"Skipping {filename}, already processed.")
            continue

        image = cv2.imread(file_path)
        if image is None:
            print(f"Skipping {filename}, not a valid image.")
            continue

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.05, minNeighbors=6, minSize=(40, 40))

        if len(faces) == 0:
            print(f"No faces detected in {filename}")
            continue

        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face
        face = image[y:y+h, x:x+w]

        resized_face = cv2.resize(face, (256, 256), interpolation=cv2.INTER_AREA)

        success = cv2.imwrite(output_path, resized_face)
        if not success:
            print(f"Error saving {filename}, check file format and path.")
            continue

        print(f"Saved resized face to {output_path}")

process_and_save_faces("all_faces", "processed_faces")
