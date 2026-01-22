from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io
import base64

app = Flask(__name__)

# ------------------ CONFIG ------------------
MODEL_PATH = r"C:\Users\dakshanya\Desktop\CropDiseaseAI\crop_disease_model.h5"

CLASSES = [
    'Pepper__bell__Bacterial_spot',
    'Pepper__bell__healthy',
    'Potato__Early_blight',
    'Potato__healthy',
    'Potato__Late_blight',
    'Tomato__Tomato_mosaic_virus',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Bacterial_spot',
    'Tomato__Early_blight',
    'Tomato__healthy',
    'Tomato__Late_blight',
    'Tomato__Leaf_Mold',
    'Tomato__Septoria_leaf_spot',
    'Tomato__Spider_mites_Two_spotted_spider_mite'
]

IMG_SIZE = (224, 224)
# --------------------------------------------

# Load trained model
model = load_model(MODEL_PATH)

# ------------------ RECOMMENDATIONS ------------------
RECOMMENDATIONS = {
    'Pepper__bell__Bacterial_spot': {
        'cause': 'Bacterial infection caused by warm and wet conditions.',
        'solution': [
            'Remove infected leaves immediately',
            'Use copper-based bactericides',
            'Avoid overhead watering'
        ]
    },
    'Pepper__bell__healthy': {
        'cause': 'No disease detected.',
        'solution': [
            'Maintain regular irrigation',
            'Apply balanced fertilizers',
            'Monitor plants regularly'
        ]
    },
    'Potato__Early_blight': {
        'cause': 'Fungal disease caused by Alternaria solani.',
        'solution': [
            'Apply fungicides like chlorothalonil',
            'Remove affected leaves',
            'Practice crop rotation'
        ]
    },
    'Potato__Late_blight': {
        'cause': 'Fungal-like pathogen (Phytophthora infestans).',
        'solution': [
            'Apply fungicides such as mancozeb',
            'Destroy infected plants',
            'Avoid excess moisture'
        ]
    },
    'Potato__healthy': {
        'cause': 'Healthy crop with no disease detected.',
        'solution': [
            'Ensure proper drainage',
            'Use disease-free seeds',
            'Monitor regularly'
        ]
    },
    'Tomato__Early_blight': {
        'cause': 'Fungal infection due to high humidity.',
        'solution': [
            'Apply appropriate fungicides',
            'Remove infected leaves',
            'Improve air circulation'
        ]
    },
    'Tomato__Late_blight': {
        'cause': 'Fungal-like pathogen affecting tomato plants.',
        'solution': [
            'Use recommended fungicides',
            'Avoid watering leaves directly',
            'Remove infected plants'
        ]
    },
    'Tomato__Leaf_Mold': {
        'cause': 'Fungal disease caused by high humidity.',
        'solution': [
            'Reduce humidity',
            'Ensure proper ventilation',
            'Apply fungicides if required'
        ]
    },
    'Tomato__Septoria_leaf_spot': {
        'cause': 'Fungal disease caused by Septoria lycopersici.',
        'solution': [
            'Remove infected leaves',
            'Avoid overhead irrigation',
            'Apply fungicides'
        ]
    },
    'Tomato__Spider_mites_Two_spotted_spider_mite': {
        'cause': 'Pest infestation due to hot and dry conditions.',
        'solution': [
            'Use insecticidal soap',
            'Increase humidity',
            'Remove heavily infested leaves'
        ]
    },
    'Tomato__Tomato_mosaic_virus': {
        'cause': 'Viral infection spread through contact.',
        'solution': [
            'Remove infected plants',
            'Disinfect tools',
            'Avoid handling healthy plants after infected ones'
        ]
    },
    'Tomato__Tomato_YellowLeaf__Curl_Virus': {
        'cause': 'Virus transmitted by whiteflies.',
        'solution': [
            'Control whiteflies',
            'Remove infected plants',
            'Use resistant plant varieties'
        ]
    },
    'Tomato__Bacterial_spot': {
        'cause': 'Bacterial infection caused by wet conditions.',
        'solution': [
            'Use copper sprays',
            'Avoid overhead irrigation',
            'Remove infected leaves'
        ]
    },
    'Tomato__healthy': {
        'cause': 'No disease detected.',
        'solution': [
            'Maintain good irrigation',
            'Apply organic fertilizers',
            'Inspect crops regularly'
        ]
    }
}
# ----------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Read image
    img = Image.open(file.stream).convert('RGB')
    img = img.resize(IMG_SIZE)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Predict
    prediction = model.predict(img_array)
    predicted_class = CLASSES[np.argmax(prediction)]
    confidence = float(np.max(prediction) * 100)

    # Get recommendation
    recommendation = RECOMMENDATIONS.get(
        predicted_class,
        {
            'cause': 'Information not available.',
            'solution': ['Consult an agricultural expert']
        }
    )

    # Convert image to base64
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()

    return jsonify({
        'prediction': predicted_class,
        'confidence': round(confidence, 2),
        'recommendation': recommendation,
        'image': img_base64
    })

if __name__ == '__main__':
    app.run(debug=True)
