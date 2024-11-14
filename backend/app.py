# backend/app.py
from flask import Flask, request, jsonify
from model.sketch_to_image import generate_realistic_image
from model.edit_features import edit_image_features
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_sketch():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    return jsonify({"message": "File uploaded successfully", "filepath": filepath}), 200

@app.route('/generate', methods=['POST'])
def generate_image():
    data = request.json
    sketch_path = data.get('sketch_path')
    result_path = generate_realistic_image(sketch_path)
    return jsonify({"generated_image": result_path}), 200

@app.route('/edit', methods=['POST'])
def edit_features():
    data = request.json
    image_path = data.get('image_path')
    edits = data.get('edits')  
    result_path = edit_image_features(image_path, edits)
    return jsonify({"edited_image": result_path}), 200

if __name__ == '__main__':
    app.run(debug=True)