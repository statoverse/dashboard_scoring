from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os
import pickle
import matplotlib
matplotlib.use('Agg')  # Utiliser un backend adapté pour les environnements serveur
import matplotlib.pyplot as plt
import warnings
import json
import plotly
from functions.functions import *  # Assurez-vous que ce module est dans le même dossier ou dans PYTHONPATH

# Désactiver les warnings si nécessaire, mais à utiliser avec prudence
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# URL de base pour les images et les ressources statiques
base_url = "https://dashboardscoring-2a7a07653340.herokuapp.com"

# Charger les données au démarrage de l'application
df, customer_ids = load_data()

app = Flask(__name__)

@app.route('/customer_ids', methods=['GET'])
def get_customer_ids():
    # Retourner la liste des IDs clients au format JSON
    _, customer_ids = load_data()
    return jsonify(customer_ids)

@app.route('/')
def welcome():
    # Passer la liste des IDs clients à la page d'accueil
    return render_template('welcome.html', customer_ids=customer_ids)

@app.route('/predict', methods=['POST'])
def predict():
    # Récupérer l'ID client depuis le JSON
    data = request.get_json()
    customer_id_str = data.get('customer_id')
    
    if customer_id_str is None:
        return jsonify({"error": "Aucun ID client fourni"}), 400
    
    try:
        selected_id = int(customer_id_str)
    except ValueError:
        return jsonify({"error": "ID client non valide"}), 400
    
    # Extraire les données du client et effectuer la prédiction
    customer_data = extract_features_from_custom(df, selected_id)
    if customer_data.empty:
        return jsonify({"error": "Client non trouvé"}), 404
    
    decision, prediction_success, prediction_failure = predict_score(customer_data)
    
    # Retourner les résultats sous forme de JSON
    return jsonify({
        'decision': decision,
        'prediction_success': prediction_success,
        'prediction_failure': prediction_failure,
        'customer_id': selected_id
    })

@app.route('/result', methods=['GET'])
def show_prediction():
    # Récupérer les paramètres de la requête pour afficher les résultats
    decision = request.args.get('decision')
    prediction_success = request.args.get('prediction_success')
    prediction_failure = request.args.get('prediction_failure')
    customer_id = request.args.get('customer_id')
    
    return render_template('prediction.html', 
                           decision=decision, 
                           prediction_success=prediction_success, 
                           prediction_failure=prediction_failure,
                           customer_id=customer_id)

@app.route('/explain/<int:customer_id>', methods=['GET'])
def explain(customer_id):
    try:
        # Récupérer le paramètre max_display depuis la requête
        max_display = int(request.args.get('max_display', 10))

        # Extraire les données du client et générer l'image SHAP
        customer_data_raw = extract_features_from_custom(df, customer_id)
        if customer_data_raw.empty:
            return jsonify({"error": "Client non trouvé"}), 404
        
        plot_path = generate_shap_image(customer_data_raw, max_display=max_display)
        filename = os.path.basename(plot_path)

        return jsonify({"image_url": f"{base_url}/static/{filename}"})
    
    except Exception as e:
        print("Erreur dans la route explain:", str(e))
        return jsonify({"error": "Une erreur s'est produite"}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    # Servir les fichiers statiques (comme les images SHAP)
    return send_from_directory('static', filename)

@app.route('/distributions/<int:customer_id>', methods=['GET'])
def distributions(customer_id):
    try:
        # Générer et retourner les distributions de features en format JSON pour Plotly
        fig = generate_feature_distributions(df, customer_id)
        fig_json = json.loads(plotly.io.to_json(fig))
        return jsonify(fig_json)
    except Exception as e:
        print("Erreur dans la génération des distributions:", str(e))
        return jsonify({"error": "Une erreur s'est produite"}), 500

if __name__ == '__main__':
    app.run(debug=True)
