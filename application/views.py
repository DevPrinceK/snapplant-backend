from flask import Blueprint, request, jsonify
from flask import render_template
from werkzeug.utils import secure_filename
from . import app
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import io
import base64
from . import app

views = Blueprint("views", __name__)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Plant Disease Detection API"})

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    # Make sure the uploaded file is an image (you can add more robust checks)
    if file and allowed_file(file.filename):
        input_img = secure_filename(file.filename)
        file.save(os.path.join(app.config['IMAGE_UPLOADS'], input_img))

        pred = predict_save(input_img)
        pred['Early_blight'] = float(pred['Early_blight'])*100
        pred['Healthy'] = float(pred['Healthy'])*100
        pred['Late_blight'] = float(pred['Late_blight'])*100
        print(f"prediction: {pred}")
        class_name = get_name_from_prediction(pred)
        confidence = pred[class_name]
        print(f"Class name: {class_name} | Confidence: {confidence}")
        return jsonify({"prediction": pred, "class": class_name, "confidence": confidence})

    return jsonify({"error": "Invalid file format"})

def allowed_file(filename):
    # basically check if the file extension is that of an image
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'gif'}

def get_name_from_prediction(prediction):
    # get the class name from the prediction
    return list(prediction.keys())[np.argmax(list(prediction.values()))]

model = load_model(app.config['MODEL'])
class_names = ['Early_blight', 'Healthy', 'Late_blight']

def predict_save(img):
    my_image = load_img(app.config['IMAGE_UPLOADS'] + img, target_size=(128, 128))
    my_image = img_to_array(my_image)
    my_image = np.expand_dims(my_image, 0)
    
    out = np.round(model.predict(my_image)[0], 2)
    prediction = {
        class_names[i]: float(out[i]) for i in range(len(class_names))
    }

    return prediction
