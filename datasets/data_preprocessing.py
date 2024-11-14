# datasets/data_preprocessing.py
import cv2
import os

def preprocess_images(input_dir, output_dir, size=(128, 128)):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path)
        if img is not None:
            resized_img = cv2.resize(img, size)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, resized_img)