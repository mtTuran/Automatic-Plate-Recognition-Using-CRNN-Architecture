from PlatesDataset import PlatesDataset, create_char_mapping
from torch.utils.data import DataLoader, random_split
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from arch import print_training_info, train, test_model, CRNN
import torch
import pandas as pd

def collate_fn(batch):
    imgs, labels, input_lengths, label_lengths = zip(*batch)
    imgs = torch.stack(imgs)
    labels = torch.stack(labels)
    input_lengths = torch.stack(input_lengths)
    label_lengths = torch.stack(label_lengths)
    return imgs, labels, input_lengths, label_lengths

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
    resized_img = cv2.resize(morph_img, (300, 106))
    resized_img = np.array(resized_img)
    return resized_img

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using the {device} device to train model.")
    root_dir = 'C:/Users/Mustafa Taner TURAN/Desktop/plate_recognition/pw_crnn_ctc/big_plates_dataset/'
    csv_file = root_dir + 'merged_csv_file.csv'

    char_mapping = create_char_mapping(csv_file)
    num_classes = len(char_mapping.char_list) + 1

    df = pd.read_csv(csv_file)
    max_label_length = df['license_plate'].str.len().max()
    print(f"Max label length: {max_label_length}")

    full_dataset = PlatesDataset(csv_file=csv_file, root_dir=root_dir, char_mapping=char_mapping, preprocess=preprocess_img, max_label_length=max_label_length)
    train_size = int(0.8 * len(full_dataset))
    val_size = int(0.1 * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, collate_fn=collate_fn)

    model = CRNN(num_classes=num_classes).to(device)
    epochs = 20

    print_training_info(train_dataset, char_mapping)
    train(model=model, train_loader=train_loader, val_loader=val_loader, epochs=epochs, char_list=char_mapping, device=device)
    test_model(model, test_loader, char_mapping, device)

