import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
from calculate_acc import predict, ctc_decode, preprocess_img, load_model, calculate_accuracy
from PlatesDataset import CharMapping
import cv2
from ultralytics import YOLO
import torch
import csv
import os


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def align_screen(window, s_width, s_height):
    window_width = root.winfo_screenwidth()
    window_height = root.winfo_screenheight()
    x = (window_width / 2) - (s_width / 2)
    y = (window_height / 2) - (s_height / 2)
    window.geometry("%dx%d+%d+%d" % (s_width, s_height, x, y))

'''
valid chars listesi modelin tanimak için egitildigi karakterlerden olusmali. bu da modelin egitiminde kullanilan plakalarin icerdigi karakterler anlamina geliyor
'''
def prepare_models():
    valid_chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    char_mapping = CharMapping(valid_chars)
    num_classes = len(char_mapping.char_list) + 1
    reading_model = load_model('saved_models/6/license_model_6.pth', device, num_classes)
    detection_model = YOLO('detection_model/best.pt')
    return reading_model, detection_model, char_mapping

def zoom_on_plate(path, detection_model):
    img = cv2.imread(path)
    results = detection_model(img)

    if len(results[0].boxes) > 0:
        best_box = max(results[0].boxes, key=lambda box: box.conf[0])
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        cropped_img = img[y1:y2, x1:x2]
        return cropped_img
    else:
        return None
    
def read_img(img, reading_model):
    img_tensor = preprocess_img(img)
    model_output = predict(reading_model, img_tensor.unsqueeze(0), device)
    decoded_texts = ctc_decode(model_output, char_mapping)
    return decoded_texts

def predict_single_image(path, reading_model, detection_model):
    img = zoom_on_plate(path, detection_model)
    if img is None:
        print("No plates detected in the chosen image!")
        return None, None
    else:
        decoded_texts = read_img(img, reading_model)
        return decoded_texts, img

def acc_of_folder(path, reading_model, detection_model):
    csv_file = path + "labels.csv"
    total_correct_chars = 0
    total_chars = 0
    total_correct_plates = 0
    total_plates = 0
    not_detected_plate_number = 0
    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header

        for row in reader:
            img_name, actual_text = row
            decoded_text, _ = predict_single_image(path + img_name, reading_model, detection_model)
            if decoded_text is not None:
                # Calculate character-level accuracy
                correct_chars, chars_in_plate = calculate_accuracy(decoded_text, actual_text)
                total_correct_chars += correct_chars
                total_chars += chars_in_plate

                # Calculate plate-level accuracy
                if decoded_text == actual_text:
                    total_correct_plates += 1
            else:
                not_detected_plate_number += 1

            total_plates += 1
        
    general_accuracy = total_correct_plates / total_plates * 100
    char_accuracy = total_correct_chars / total_chars * 100
    reading_accuracy = total_correct_plates / (total_plates - not_detected_plate_number) * 100

    print(f"Total plates number: {total_plates}\n Total detected plates number: {total_plates - not_detected_plate_number}\nTotal char number: {total_chars}\nTotal correct plates number: {total_correct_plates}\nTotal correct char number: {total_correct_chars}")

    return general_accuracy, char_accuracy, reading_accuracy

def choose_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        result_text, cropped_img = predict_single_image(file_path, reading_model, detection_model)
        if result_text:
            show_image(file_path, cropped_img, result_text)
        else:
            print("No plates detected in the chosen image!")

def choose_folder():
    folder_path = filedialog.askdirectory()
    if folder_path:
        general_acc, char_acc, read_acc = acc_of_folder(folder_path + '/', reading_model, detection_model)
        result_label.config(text=f"General Accuracy: {general_acc:.2f}%\nCharacter Accuracy: {char_acc:.2f}%\nReading Accuracy: {read_acc}%\t(doesn't include the plates that are not detected)")
        print(f"General Accuracy: {general_acc:.2f}%")
        print(f"Character Accuracy: {char_acc:.2f}%")
        print(f"Reading Accuracy: {read_acc:.2f}%")

def find_license_plate_numbers(reading_model, detection_model):
    folder_path = filedialog.askdirectory()
    csv_file = folder_path + '/' + 'license_plate_numbers.csv'

    if folder_path:
        images = os.listdir(folder_path)
        with open(csv_file, mode='w', newline='', encoding='utf-8') as cf:
            writer = csv.writer(cf)
            writer.writerow(['img_name', 'license_plate'])
            for filename in images:
                if filename.endswith('.png') or filename.endswith('.jpg') or filename.endswith('.jpeg'):
                    decoded_text, _ = predict_single_image(folder_path + '/' + filename, reading_model, detection_model)
                    if decoded_text is None:
                        decoded_text = "NoPlateDetected!"
                    writer.writerow([filename, decoded_text])
        print("Licensing completed!")

def show_image(original_path, cropped_img, result_text):
    original_img = Image.open(original_path)
    original_img.thumbnail((400, 400))
    original_img_tk = ImageTk.PhotoImage(original_img)
    original_label.config(image=original_img_tk)
    original_label.image = original_img_tk

    if cropped_img is not None:
        cropped_img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        cropped_img_pil.thumbnail((400, 400))
        cropped_img_tk = ImageTk.PhotoImage(cropped_img_pil)
        cropped_label.config(image=cropped_img_tk)
        cropped_label.image = cropped_img_tk

    result_label.config(text=f"Detected Text: {result_text}")

if __name__ == '__main__':
    root = tk.Tk()
    root.title('Automatic License Plate OCR')
    w = 1300
    h = 800
    align_screen(root, w, h)

    reading_model, detection_model, char_mapping = prepare_models()

    choose_button = Button(root, text="Choose Image", command=choose_image)
    choose_button.grid(row=0, column=0, padx=10, pady=10, sticky='nsew')

    choose_folder_button = Button(root, text="Choose Folder", command=choose_folder)
    choose_folder_button.grid(row=0, column=1, padx=10, pady=10, sticky='nsew')

    find_licenses_button = Button(root, text="Licenses", command=lambda: find_license_plate_numbers(reading_model, detection_model))
    find_licenses_button.grid(row=0, column=2, padx=10, pady=10, sticky='nsew')

    original_label = Label(root)
    original_label.grid(row=1, column=0, padx=10, pady=10)

    cropped_label = Label(root)
    cropped_label.grid(row=1, column=1, padx=10, pady=10)

    result_label = Label(root, text="")
    result_label.grid(row=2, column=0, columnspan=2, padx=10, pady=10)

    root.mainloop()
    