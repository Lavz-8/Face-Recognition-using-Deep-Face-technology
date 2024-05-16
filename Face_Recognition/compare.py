from flask import Flask, render_template, request, jsonify
from deepface import DeepFace
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Dictionary of names and corresponding image paths
image_paths = {
    "Obama": "static\\Obama.jpg",
    "Trump": "static\\Trump.jpg",
    # Add more entries as needed
}

# Lowering the threshold to make the verification process more lenient
threshold = 0.5

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    name = request.form['name']
    if name in image_paths:
        try:
            # Read the base64 encoded image from the form data
            image_data = request.form['image']
            image_data = image_data.split(',')[1]  # Remove data URI prefix
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            img2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Perform face recognition
            result = DeepFace.verify(image_paths[name], img2, enforce_detection=False, model_name='Facenet', distance_metric='euclidean')

            # Return JSON response indicating whether the images match
            return jsonify({"match": result["verified"]})
        except Exception as e:
            return jsonify({"error": str(e)})
    else:
        return jsonify({"error": "No stored image found for the entered name."})

if __name__ == '__main__':
    app.run(debug=True)