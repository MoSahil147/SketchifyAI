# backend/model/utils.py
import cv2

def load_image(image_path):
    return cv2.imread(image_path)

def save_image(image, path):
    cv2.imwrite(path, image)