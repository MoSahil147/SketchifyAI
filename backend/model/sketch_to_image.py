# backend/model/sketch_to_image.py
import torch
import cv2
import os
from .pix2pix_model import Pix2PixGenerator

def load_sketch_to_image_model():
    model = Pix2PixGenerator()
    model.load_state_dict(torch.load("models/sketch_to_image_model.pth"))
    model.eval()
    return model

def generate_realistic_image(sketch_path):
    model = load_sketch_to_image_model()
    sketch_img = cv2.imread(sketch_path, cv2.IMREAD_GRAYSCALE)
    sketch_img = cv2.resize(sketch_img, (128, 128))
    sketch_tensor = torch.from_numpy(sketch_img).unsqueeze(0).unsqueeze(0).float() / 255.0

    with torch.no_grad():
        generated_img = model(sketch_tensor)

    result_path = os.path.join("static/uploads", "generated_image.jpg")
    generated_img = generated_img.squeeze().numpy() * 255
    generated_img = cv2.resize(generated_img, (512, 512))
    cv2.imwrite(result_path, generated_img.astype("uint8"))
    return result_path