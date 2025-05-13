from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import os
import numpy as np
np.__version__ = '1.26.4'  # Force la version attendue

app = Flask(__name__)
CORS(app)
# Chemin vers les modèles
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model')

# Chargement du modèle et des préprocesseurs
model = joblib.load(os.path.join(MODEL_DIR, 'attrition_model.joblib'))
scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.joblib'))

# Chargement des encodeurs
label_encoders = {
    'Department': joblib.load(os.path.join(MODEL_DIR, 'Department_encoder.joblib')),
    'EducationField': joblib.load(os.path.join(MODEL_DIR, 'EducationField_encoder.joblib')),
    'JobRole': joblib.load(os.path.join(MODEL_DIR, 'JobRole_encoder.joblib'))
}

# Chargement des métadonnées
with open(os.path.join(MODEL_DIR, 'model_metadata.json'), 'r') as f:
    metadata = json.load(f)

@app.route('/')
def home():
    return "API de prédiction d'attrition - Utilisez /predict pour les prédictions"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupération des données de la requête
        data = request.get_json()
        
        # Vérification des champs requis
        required_fields = metadata['features']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Champ manquant: {field}'}), 400
        
        # Préparation des données dans le bon ordre
        input_data = []
        for feature in metadata['features']:
            if feature in metadata['categorical_columns']:
                # Encodage des variables catégorielles
                encoder = label_encoders[feature]
                try:
                    encoded_value = encoder.transform([data[feature]])[0]
                    input_data.append(encoded_value)
                except ValueError as e:
                    return jsonify({
                        'error': f'Valeur non valide pour {feature}',
                        'valid_values': list(encoder.classes_)
                    }), 400
            else:
                input_data.append(data[feature])
        
        # Conversion en array numpy et reshape
        input_array = np.array(input_data).reshape(1, -1)
        
        # Standardisation
        scaled_input = scaler.transform(input_array)
        
        # Prédiction
        prediction = model.predict(scaled_input)
        probability = model.predict_proba(scaled_input)
        
        # Formatage de la réponse
        response = {
            'prediction': int(prediction[0]),
            'probability': float(probability[0][1]),  # Probabilité de départ
            'interpretation': 'Likely to leave' if prediction[0] == 1 else 'Likely to stay'
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)