import torch
torch.cuda.set_device(0)

from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8s.yaml")
    results = model.train(data="C:/Users/Mustafa Taner TURAN/Desktop/plate_recognition/wide_data/data.yaml", epochs=20)   
