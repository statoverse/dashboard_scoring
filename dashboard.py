import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import plotly.io as pio
import json
import time
from streamlit_option_menu import option_menu

# Charger les données clients
data_path = 'data/customers.csv'
df = pd.read_csv(data_path)
customer_ids = df['SK_ID_CURR'].astype(str).tolist()

# URL de base pour l'API déployée sur Heroku
base_url = "https://dashboardscoring-2a7a07653340.herokuapp.com"  # Remplacez par votre URL Heroku

# Configuration de la barre latérale avec autocomplétion et sliders
with st.sidebar:
    st.write("### Rechercher un Client ID")
    search_input = st.text_input("Tapez le numéro du client")
    matching_ids = [cid for cid in customer_ids if search_input in cid]
    selected_customer_id = st.selectbox("Client ID", matching_ids)
    num_features = st.slider("Nombre de features SHAP à afficher", min_value=1, max_value=20, value=5)
    decision_threshold = st.slider("Seuil de décision", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

# Navigation entre les trois sous-panneaux
selected_panel = option_menu(
    menu_title=None,
    options=["Résultat Prêt", "Graphique SHAP", "Distributions"],
    icons=["check-circle", "chart-bar", "chart-line"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
)

# Vérification de la sélection et récupération des données de prédiction
if selected_customer_id:
    response = requests.post(f"{base_url}/predict", json={"customer_id": selected_customer_id})
    
    if response.ok:
        data = response.json()
        prediction_success = data.get("prediction_success")
        prediction_failure = data.get("prediction_failure")
        decision = data.get("decision")
        
        # Page 1 : Jauge et Résultat de Prêt
        if selected_panel == "Résultat Prêt":
            st.write("## Résultat de l'accord de prêt")
            decision_finale = "Accordé" if prediction_failure <= decision_threshold else "Refusé"
            st.write(f"**Décision finale à un seuil de {decision_threshold}:** {decision_finale}")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction_failure,
                title={"text": "Probabilité de défaut de paiement"},
                gauge={'axis': {'range': [0, 1]},
                       'steps': [
                           {'range': [0, decision_threshold], 'color': "lightgreen"},
                           {'range': [decision_threshold, 1], 'color': "red"}]},
            ))
            st.plotly_chart(gauge)

        # Page 2 : Graphique SHAP
        elif selected_panel == "Graphique SHAP":
            st.write("## Explication du modèle : SHAP")
            shap_response = requests.get(f"{base_url}/explain/{selected_customer_id}", params={"max_display": num_features})
            if shap_response.ok:
                shap_data = shap_response.json()
                image_url = shap_data.get("image_url")
                if image_url:
                    st.image(f"{image_url}?t={time.time()}", caption="Graphique SHAP", use_column_width=True)
                else:
                    st.error("L'URL de l'image SHAP est introuvable.")
            else:
                st.error("Erreur lors de la récupération du graphique SHAP.")

        # Page 3 : Distributions des Features
        elif selected_panel == "Distributions":
            st.write("## Positionnement du client par rapport aux autres clients")
            dist_response = requests.get(f"{base_url}/distributions/{selected_customer_id}")
            if dist_response.ok:
                fig_data = dist_response.json()
                fig = pio.from_json(json.dumps(fig_data))
                st.plotly_chart(fig)
            else:
                st.error("Erreur lors de la récupération des distributions des features.")
    else:
        st.error("Erreur lors de la récupération des données de prédiction. Veuillez vérifier l'ID client et réessayer.")
else:
    st.info("Veuillez sélectionner un Client ID dans la barre latérale pour commencer.")
