import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np

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


def fix_label_lengths(labels):
    labels = [label[label != 0] for label in labels]
    label_lengths = [len(label) for label in labels]  # Calculate the length of each label after removing padding
    
    label_lengths = torch.tensor(label_lengths, dtype=torch.long).unsqueeze(1)
    
    return label_lengths

class CRNN(nn.Module):
    def __init__(self, num_classes):
        super(CRNN, self).__init__()
        self.conv_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.pool_1 = nn.MaxPool2d(2, 2)
        self.dropout_1 = nn.Dropout(0.5)

        self.conv_2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool_2 = nn.MaxPool2d(2, 2)
        self.dropout_2 = nn.Dropout(0.2)

        self.conv_3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batch_norm_3 = nn.BatchNorm2d(256)
        self.conv_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool_4 = nn.MaxPool2d((2, 1))
        self.dropout_3 = nn.Dropout(0.2)

        self.conv_5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.batch_norm_5 = nn.BatchNorm2d(512)

        self.conv_6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.batch_norm_6 = nn.BatchNorm2d(512)
        self.pool_6 = nn.MaxPool2d((2, 1))
        self.dropout_4 = nn.Dropout(0.2)

        self.conv_7 = nn.Conv2d(512, 512, kernel_size=2)
        self.batch_norm_7 = nn.BatchNorm2d(512)

        self.blstm_1 = nn.LSTM(2560, 640, bidirectional=True, batch_first=True)
        self.dropout_5 = nn.Dropout(0.2)
        self.blstm_2 = nn.LSTM(1280, 320, bidirectional=True, batch_first=True)
        self.dropout_6 = nn.Dropout(0.2)

        self.dense = nn.Linear(640, num_classes)

    def forward(self, x):
        x = F.selu(self.conv_1(x))
        x = self.pool_1(x)
        x = self.dropout_1(x)
        x = F.selu(self.conv_2(x))
        x = self.pool_2(x)
        x = self.dropout_2(x)
        x = F.selu(self.conv_3(x))
        x = self.batch_norm_3(x)
        x = F.selu(self.conv_4(x))
        x = self.pool_4(x)
        x = self.dropout_3(x)
        x = F.selu(self.conv_5(x))
        x = self.batch_norm_5(x)
        x = F.selu(self.conv_6(x))
        x = self.batch_norm_6(x)
        x = self.pool_6(x)
        x = self.dropout_4(x)
        x = F.selu(self.conv_7(x))
        x = self.batch_norm_7(x)

        x = x.permute(0, 3, 2, 1)  # Reorder to (batch, width, height, channels)
        
        x = x.contiguous().view(x.size(0), x.size(1), -1)  # Flatten height dimension to get (batch, width, channels*height)

        (x, h_n) = self.blstm_1(x)
        x = self.dropout_5(x)
        (x, h_n) = self.blstm_2(x)
        x = self.dropout_6(x)

        x = self.dense(x)

        return x


def train(model, train_loader, val_loader, epochs, char_list, device, patience=5):
    criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    model.to(device)
    
    train_losses = []
    val_losses = []

    best_val_loss = float('inf')
    epochs_since_improvement = 0

    for epoch in range(epochs):
        model.train()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Learning Rate {current_lr}")
        epoch_train_loss = 0
        for batch_idx, (inputs, labels, input_lengths, label_lengths) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            input_lengths = input_lengths.to(device)
            label_lengths = fix_label_lengths(labels=labels)
            label_lengths = label_lengths.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)

            outputs = outputs.log_softmax(2)
            outputs = outputs.permute(1, 0, 2)  # CTC expects (T, N, C)

            input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long, device=device)
            
            loss = criterion(outputs, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()

            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch+1}/{epochs} | Batch: {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}')

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        scheduler.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, labels, input_lengths, label_lengths in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                input_lengths = input_lengths.to(device)
                label_lengths = fix_label_lengths(labels=labels)
                label_lengths = label_lengths.to(device)

                outputs = model(inputs)
                outputs = outputs.log_softmax(2)
                outputs = outputs.permute(1, 0, 2)

                input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long, device=device)

                loss = criterion(outputs, labels, input_lengths, label_lengths)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        print(f'Epoch: {epoch+1}/{epochs} | Validation Loss: {avg_val_loss:.4f}')

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            model_name = 'license_model_best.pth'
            torch.save(model.state_dict(), model_name)
            print(f'Model saved to {model_name}')
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f'Early stopping triggered. No improvement for {patience} epochs.')
                break

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.savefig('loss_plot.png')
    plt.show()


def test_model(model, test_loader, char_mapping, device):
    model.eval()
    
    total_correct_chars = 0
    total_chars = 0
    total_correct_plates = 0
    total_plates = 0
        
    with torch.no_grad():
        for inputs, labels, input_lengths, label_lengths in test_loader:
            inputs = inputs.to(device)
            
            outputs = model(inputs)  # outputs shape should be [N, T, C]
            
            # Decode predictions using CTC decode function
            decoded_texts = []
            for i in range(outputs.size(0)):
                single_output = outputs[i]  # Shape is [T, C]
                
                # Expand dimensions to [T, N, C] where N=1
                single_output = single_output.unsqueeze(1)  # New shape is [T, 1, C]
                
                decoded_text = ctc_decode(single_output, char_mapping)
                decoded_texts.append(decoded_text)
            
            labels = labels.cpu().numpy()
            
            for decoded_text, label in zip(decoded_texts, labels):
                label_text = char_mapping.decode(label)
                
                # character-level accuracy
                correct_chars, chars_in_plate = calculate_accuracy(decoded_text, label_text)
                total_correct_chars += correct_chars
                total_chars += chars_in_plate
                
                # plate-level accuracy
                if decoded_text == label_text:
                    total_correct_plates += 1
                total_plates += 1
    

    general_accuracy = total_correct_plates / total_plates * 100
    char_accuracy = total_correct_chars / total_chars * 100

    print(f"General Accuracy: {general_accuracy:.2f}%")
    print(f"Character-Level Accuracy: {char_accuracy:.2f}%")



def print_training_info(dataset, char_mapping):
    example_img, example_label, m, n = dataset[0]
    
    # Number of unique characters including the CTC blank token
    num_unique_chars = len(char_mapping.char_list) + 1
    
    # Decode example label from indices
    decoded_label = char_mapping.decode(example_label.tolist())
    
    print(f"Number of unique characters: {num_unique_chars}")
    print(f"Example label text: {decoded_label}")
    print(f"Encoded label: {example_label.tolist()}")
    print(f"Example image tensor: {example_img}")
