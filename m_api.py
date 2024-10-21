from flask import Flask, request, jsonify
from flask_cors import CORS
from calculate_acc import predict, ctc_decode, preprocess_img, load_model
from PlatesDataset import CharMapping
import cv2
from ultralytics import YOLO
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def zoom_on_plate(file):
    img = cv2.imread(file)
    results = detection_model(img)

    if len(results[0].boxes) > 0:
        best_box = max(results[0].boxes, key=lambda box: box.conf[0])
        x1, y1, x2, y2 = map(int, best_box.xyxy[0].tolist())
        cropped_img = img[y1:y2, x1:x2]
        return cropped_img
    else:
        return None
    
def read_img(img):
    img_tensor = preprocess_img(img)
    model_output = predict(reading_model, img_tensor.unsqueeze(0), device)
    decoded_texts = ctc_decode(model_output, char_mapping)
    return decoded_texts

app = Flask(__name__)
CORS(app)
reading_model, detection_model, char_mapping = prepare_models()


@app.route("/")
def info():
    return jsonify({"char_mapping": char_mapping.char_list, "device": device.type})

@app.route("/predict", methods=['POST'])
def predict_single_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    else:
        img_file = request.files['image']

    img = zoom_on_plate(img_file)

    if img is None:
        return jsonify({"error": "No plate detected"}), 400
    else:
        decoded_texts = read_img(img)
        return jsonify({"plate": decoded_texts})

if __name__ == "__main__":
    app.run(debug=True)