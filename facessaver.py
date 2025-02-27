import cv2
import os

def process_folder(input_folder="faces", output_folder="detected_faces"):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        parts = filename.split('_')

        if len(parts) >= 1 and parts[0].isdigit():
            age = parts[0] 
            

            image = cv2.imread(file_path)
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray_image,
                scaleFactor=1.05,  
                minNeighbors=6,    
                minSize=(40, 40)   
            )

            face_count = 0
            for (x, y, w, h) in faces:
                face = image[y:y+h, x:x+w]

                output_filename = f"{age}_{filename}"
                output_path = os.path.join(output_folder, output_filename)
                cv2.imwrite(output_path, face)
                print(f"Saved face to {output_path}")
                face_count += 1

            if face_count == 0:
                print(f"No faces detected in {filename}")

process_folder("02_detected_facesxx", "03_filtered_facesxx")
