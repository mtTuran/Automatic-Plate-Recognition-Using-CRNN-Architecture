import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import os

class CharMapping:
    def __init__(self, char_list):
        self.char_list = char_list
        self.char_to_index = {char: idx + 1 for idx, char in enumerate(char_list)}  # Start index from 1
        self.char_to_index['<BLANK>'] = 0  # CTC blank token

    def encode(self, text):
        return [self.char_to_index[char] for char in text]

    def decode(self, indices):
        index_to_char = {idx: char for char, idx in self.char_to_index.items()}
        return ''.join([index_to_char[idx] for idx in indices if idx != 0])

class PlatesDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, preprocess=None, char_mapping=None, max_label_length=10):
        self.associations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.preprocess = preprocess
        self.char_mapping = char_mapping
        self.max_label_length = max_label_length

    def __len__(self):
        return len(self.associations)

    def pad_sequences(self, sequences, maxlen, padding_value=0):
        padded_sequences = []
        for seq in sequences:
            if len(seq) < maxlen:
                seq = seq + [padding_value] * (maxlen - len(seq))
            else:
                seq = seq[:maxlen]
            padded_sequences.append(seq)
        return padded_sequences

    def __getitem__(self, index):
        try:
            img_id = self.associations.iloc[index, 0]  # Image ID from the CSV
            img_path = os.path.join(self.root_dir, img_id)
            label_text = self.associations.iloc[index, 1]  # Corresponding label from the CSV
        except IndexError as e:
            print(f"IndexError: {e} - Check the index value and DataFrame dimensions.")
        except KeyError as e:
            print(f"KeyError: {e} - Check the column names in the DataFrame.")
        except Exception as e:
            print(f"Unexpected error: {e}")

        img = cv2.imread(img_path)
        if self.preprocess:
            img = self.preprocess(img)
        img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)  # Convert to tensor and add channel dimension
        img = img / 255.0
        
        # Encode the label text
        label = self.char_mapping.encode(label_text)
        label = self.pad_sequences([label], self.max_label_length)[0]
        label = torch.tensor(label, dtype=torch.long)

        return img, label, torch.tensor([img.size(2)]), torch.tensor([len(label)])


def create_char_mapping(csv_file):
    df = pd.read_csv(csv_file)
    all_chars = ''.join(df.iloc[:, 1].values)
    unique_chars = sorted(set(all_chars))  # Get unique characters
    return CharMapping(unique_chars)


