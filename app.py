import os
import uuid
import numpy as np
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename

# Import TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# 1. Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB

# 2. Load your trained model (which should output 6 classes)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "mobilenet_v2_model.h5")
model = load_model(MODEL_PATH)

# CLASS_LABELS must match the exact order your model outputs
CLASS_LABELS = [
    "Basil - Healthy",       # index 0
    "Diseased Basil - Downy Mildew",        # index 1
    "Diseased Basil - Fusarium Wilt",       # index 2 
    "Diseased Basil - Gray Mold",           # index 3
    "Diseased Basil - Septoria leaf spot",  # index 4
                  
]

# 3. Dictionary containing disease details
DISEASE_INFO = {
    "Diseased Basil - Downy Mildew": {
        "lost_properties": [
            "Antioxidant properties: It can significantly reduce the levels of essential oils and antioxidants like eugenol and linalool.",
            "Antimicrobial properties: The disease can compromise the plant's ability to produce compounds with antimicrobial activity.",
            "Anti-inflammatory properties: Reduction in essential oils can diminish the plant's anti-inflammatory properties."
        ],
        "measures": [
            "Prevention:",
            "1. Plant resistant varieties.",
            "2. Ensure proper spacing for good air circulation.",
            "3. Avoid overhead watering.",
            "4. Rotate crops to prevent soil-borne pathogens.",
            "Control:",
            "1. Remove and destroy infected plants.",
            "2. Apply fungicides labeled for downy mildew on basil."
        ]
    },
    "Diseased Basil - Fusarium Wilt": {
        "lost_properties": [
            "Antioxidant and antimicrobial properties: Fusarium wilt can disrupt the plant's metabolism, reducing the production of essential oils."
        ],
        "measures": [
            "Prevention:",
            "1. Plant disease-free seedlings.",
            "2. Use well-draining soil.",
            "3. Avoid overwatering.",
            "4. Rotate crops.",
            "Control:",
            "1. Unfortunately, there is no effective cure for Fusarium wilt once it infects a plant."
        ]
    },
    "Diseased Basil - Gray Mold": {
        "lost_properties": [
            "Antioxidant and antimicrobial properties: Gray mold can significantly reduce the quality and quantity of essential oils."
        ],
        "measures": [
            "Prevention:",
            "1. Ensure good air circulation around plants.",
            "2. Avoid overcrowding.",
            "3. Water at the base of the plant, avoiding wetting the foliage.",
            "Control:",
            "1. Apply fungicides labeled for gray mold on basil."
        ]
    },
    "Diseased Basil - Septoria leaf spot": {
        "lost_properties": [
            "Antioxidant and antimicrobial properties: Septoria leaf spot can reduce the overall health of the plant, affecting the production of essential oils."
        ],
        "measures": [
            "Prevention:",
            "1. Plant disease-free seedlings.",
            "2. Maintain good sanitation practices.",
            "3. Remove and destroy infected plant debris.",
            "4. Avoid overhead watering.",
            "Control:",
            "1. Apply fungicides labeled for Septoria leaf spot on basil."
        ]
    }
}

def classify_plant(image_path):
    """
    Loads an image from 'image_path', preprocesses it, and uses
    the MobileNetV2 model to predict among 6 classes.
    """
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    predictions = model.predict(img_array)  # shape = (1, 6)
    predicted_class_index = np.argmax(predictions[0])     # 0..5
    predicted_class_label = CLASS_LABELS[predicted_class_index]
    confidence = float(np.max(predictions[0]))  # highest confidence in [0..1]

    is_healthy = (predicted_class_label == "Basil - Healthy")
    is_medicinal = is_healthy

    disease_name = None
    lost_properties = []
    measures = []
    basil_benefits = []

    if is_healthy:
        basil_benefits = [
            "This Basil leaf is Antioxidant Rich which Contains compounds like eugenol and rosmarinic acid that protect cells from damage.",
            "This Basil leaf is Anti-inflammatory which May help reduce inflammation associated with conditions like arthritis.",
            "This Basil leaf is Antimicrobial which Exhibits activity against certain bacteria and fungi."
        ]

    if not is_healthy:
        disease_name = predicted_class_label
        disease_data = DISEASE_INFO.get(disease_name, {})
        lost_properties = disease_data.get("lost_properties", [])
        measures = disease_data.get("measures", [])

    return {
        "plant_name": "Basil (Predicted)",
        "is_medicinal": is_medicinal,
        "is_healthy": is_healthy,
        "disease_name": disease_name,
        "lost_properties": lost_properties,
        "measures": measures,
        "predicted_label": predicted_class_label,
        "confidence": round(confidence * 100, 2),  # percentage
        "basil_benefits": basil_benefits  # Include the basil health benefits
    }

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'plant_image' not in request.files:
            return redirect(request.url)
        file = request.files['plant_image']
        if file.filename == '':
            return redirect(request.url)

        if file:
            filename = secure_filename(file.filename)
            unique_filename = str(uuid.uuid4()) + os.path.splitext(filename)[1]
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(save_path)

            classification_result = classify_plant(save_path)

            return render_template(
                'index.html',
                classification=classification_result,
                uploaded_image=url_for('static', filename=f'uploads/{unique_filename}')
            )

    return render_template('index.html', classification=None)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename))

if __name__ == '__main__':
    app.run(debug=True)
