import streamlit as st
import pandas as pd
import requests
import plotly.graph_objs as go
import time

# Charger le DataFrame dans Streamlit
data_path = 'data/customers.csv'
df = pd.read_csv(data_path)

# Vérifier le chargement des données
if df.empty or 'SK_ID_CURR' not in df.columns:
    st.error("Erreur : Impossible de charger les données clients.")
else:
    customer_ids = df['SK_ID_CURR'].tolist()

# Placer les éléments de sélection dans la barre latérale
with st.sidebar:
    st.write("### Sélectionnez un Client ID")
    selected_customer_id = st.selectbox("Client ID", customer_ids)

# Vérifier qu'un ID client a été sélectionné et afficher les résultats dans le panneau principal
if selected_customer_id:
    # Envoyer une requête POST à l'API Flask pour obtenir la prédiction
    response = requests.post("http://localhost:5000/predict", json={"customer_id": selected_customer_id})
    print("Deshboard, selected customer id",response)
    # Vérifier la réponse de l'API
    if response.ok:
        # Extraire les données de prédiction
        data = response.json()
        prediction_success = data.get("prediction_success")
        prediction_failure = data.get("prediction_failure")
        decision = data.get("decision")
        
        # Afficher les résultats dans le panneau principal
        st.write("## Résultats de Prédiction")
        st.write(f"**Probabilité de remboursement :** {prediction_success}")
        st.write(f"**Probabilité de défaut de paiement :** {prediction_failure}")
        st.write(f"**Décision finale :** {decision}")
        
        # Créer et afficher une jauge de score
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prediction_failure,
            title={"text": "Probabilité de défaut de paiement"},
            gauge={'axis': {'range': [0, 1]},
                   'steps': [
                       {'range': [0, 0.25], 'color': "lightgreen"},
                       {'range': [0.25, 1], 'color': "red"}]},
        ))
        st.plotly_chart(gauge)
    else:
        st.error("Erreur dans la récupération des données de prédiction.")

    # Récupérer l'image SHAP à partir de l'API Flask
    shap_response = requests.get(f"http://localhost:5000/explain/{selected_customer_id}")
    
    if shap_response.ok:
        # Extraire l'URL de l'image depuis la réponse JSON
        shap_data = shap_response.json()
        image_url = shap_data.get("image_url")
        
        if image_url:
            # Ajouter un paramètre de cache-bypass avec horodatage à l'URL de l'image
            timestamp = time.time()
            image_url_with_bypass = f"{image_url}?t={timestamp}"
            
            # Afficher l'image SHAP dans le panneau principal
            st.image(image_url_with_bypass, caption="Graphique SHAP", use_column_width=True)
        else:
            st.error("URL de l'image SHAP non trouvée.")
    else:
        st.error("Erreur dans la récupération de l'image SHAP.")
