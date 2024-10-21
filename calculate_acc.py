import cv2
import torch
import numpy as np
import csv
from arch import CRNN
from PlatesDataset import create_char_mapping
import os

def preprocess_img(img):
    multiply_size = 2
    height, width = img.shape[:2]
    height, width = height * multiply_size, width * multiply_size
    img = cv2.resize(img, (width, height))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_img = cv2.GaussianBlur(gray_img, (15, 15), 0)
    divided_img = cv2.divide(gray_img, blurred_img, scale=255)
    thresholded_img = cv2.adaptiveThreshold(divided_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((3, 3), np.uint8)
    morph_img = cv2.morphologyEx(thresholded_img, cv2.MORPH_OPEN, kernel=kernel, iterations=1)
    morph_img = cv2.resize(morph_img, (300, 106))
    morph_img = np.expand_dims(morph_img, axis=0).astype(np.float32)  # Add channel dimension and ensure dtype is float32
    morph_img /= 255.0
    return torch.tensor(morph_img)

def load_model(model_path, device, num_classes):
    model = CRNN(num_classes)
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

def predict(model, image_tensor, device):
    image_tensor = image_tensor.to(device)
    with torch.no_grad():
        output = model(image_tensor)
    return output

def ctc_decode(output, char_mapping, blank_index=0):
    output = output.log_softmax(2)  # (T, N, C)
    output = output.permute(1, 0, 2)  # (N, T, C) for easier processing

    decoded_text = []
    prev_index = -1
    for i in range(output.size(0)):
        probs = output[i].cpu().numpy()
        predicted_indices = np.argmax(probs, axis=1)
        decoded_indices = []
        for index in predicted_indices:
            if index != blank_index and index != prev_index:
                decoded_indices.append(index)
            prev_index = index
        decoded_char = char_mapping.decode(decoded_indices)
        decoded_text.append(decoded_char)
    
    decoded_text = ''.join(decoded_text)
    return decoded_text

def calculate_accuracy(pred_text, actual_text):
    correct_chars = sum(1 for pred, actual in zip(pred_text, actual_text) if pred == actual)
    total_chars = len(actual_text)
    return correct_chars, total_chars

def main():
    root_dir = 'C:/Users/Mustafa Taner TURAN/Desktop/plate_recognition/pw_crnn_ctc/big_plates_dataset/'
    csv_file = os.path.join(root_dir, 'merged_csv_file.csv')
    char_mapping_file = "C:/Users/Mustafa Taner TURAN/Desktop/plate_recognition/pw_crnn_ctc/big_plates_dataset/merged_csv_file.csv"     # modelin eğitiminde kullanılan csv dosyasından modelin 
                                                                                                                                        # tanıdığı karakter setini tekrardan oluşturma
    char_mapping = create_char_mapping(char_mapping_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model("saved_models/6/license_model_6.pth", device, len(char_mapping.char_list) + 1)

    total_correct_chars = 0
    total_chars = 0
    total_correct_plates = 0
    total_plates = 0

    with open(csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header

        for row in reader:
            img_name, actual_text = row
            img = cv2.imread(os.path.join(root_dir, img_name))
            img_tensor = preprocess_img(img)
            model_output = predict(model, img_tensor.unsqueeze(0), device)  # Add batch dimension

            decoded_text = ctc_decode(model_output, char_mapping)

            correct_chars, chars_in_plate = calculate_accuracy(decoded_text, actual_text)
            total_correct_chars += correct_chars
            total_chars += chars_in_plate

            if decoded_text == actual_text:
                total_correct_plates += 1
            total_plates += 1


    general_accuracy = total_correct_plates / total_plates * 100
    char_accuracy = total_correct_chars / total_chars * 100

    print(f"General Accuracy: {general_accuracy:.2f}%")
    print(f"Character-Level Accuracy: {char_accuracy:.2f}%")

if __name__ == '__main__':
    main()
