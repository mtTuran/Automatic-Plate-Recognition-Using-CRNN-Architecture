import cv2
import os
import csv
import json
from ultralytics import YOLO

model = YOLO('runs/detect/train43/weights/best.pt')

input_folder = 'C:/Users/Mustafa Taner TURAN/Desktop/plate_recognition/STAJ FOTO/images/'
output_folder = 'C:/Users/Mustafa Taner TURAN/Desktop/plate_recognition/labelled_test_dataset/'

os.makedirs(output_folder, exist_ok=True)

json_path = os.path.join(input_folder, 'data.json')
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

image_to_plaka = {str(item['KGM_RESIM_ID']): item['UAB_PLAKA'] for item in data}

csv_file_path = os.path.join(output_folder, 'labels.csv')
with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['img_name', 'license_plate'])  # Write the header

    i = 1
    for filename in os.listdir(input_folder):
        if filename.endswith('.png') or filename.endswith('.jpg'):  
            img_name, _ = os.path.splitext(filename)
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path)

            results = model(img)

            if len(results[0].boxes) > 0:
                # Extract the box with the highest confidence
                best_box = max(results[0].boxes, key=lambda box: box.conf[0])

                # Extract the coordinates of the best box
                x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())

                # Crop the plate from the image 
                cropped_img = img[y1:y2, x1:x2]

                output_path = os.path.join(output_folder, filename)
                cv2.imwrite(output_path, cropped_img)

                plaka = image_to_plaka.get(img_name)
                if plaka:
                    csv_writer.writerow([filename, plaka])

                print(f"{i} -> Processed and saved: {filename}")
            else:
                print(f"{i} -> No license plate detected in: {filename}")
            i += 1

print("Processing completed.")
