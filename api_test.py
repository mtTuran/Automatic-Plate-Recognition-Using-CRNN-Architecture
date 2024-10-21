import requests
import os

url = 'http://localhost:5000/predict'

path = "C:/Users/Mustafa Taner TURAN/Desktop/plate_recognition/STAJ FOTO/images/"
images = os.listdir(path)
img_path = path + images[0]

with open(img_path, 'rb') as image_file:
    files = {'image': image_file}
    response = requests.post(url, files=files)

print(response.json())
