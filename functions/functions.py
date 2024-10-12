from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Utiliser un backend adapté pour les environnements serveur
import matplotlib.pyplot as plt
import joblib
import warnings

# Désactiver les warnings si nécessaire, mais à utiliser avec prudence
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

def load_data():
    # Load the CSV file into a DataFrame
    data_path = 'data/customers.csv'
    df = pd.read_csv(data_path)
    # Extract the customer IDs (SK_ID_CURR) column
    customer_ids = df['SK_ID_CURR'].tolist()
    return df, customer_ids

def extract_features_from_custom(df, customer_id):
    import pandas as pd
    # Filtrer les données du client et forcer le retour sous forme de DataFrame
    customer_data = df[df['SK_ID_CURR'] == customer_id].copy()
    customer_data = customer_data.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')
    
    # Si la sélection n'a pas de colonnes, cela indiquera un problème de données
    if customer_data.empty:
        print(f"Aucun client trouvé pour l'ID {customer_id}")
    elif not isinstance(customer_data, pd.DataFrame):
        customer_data = pd.DataFrame([customer_data], columns=df.columns.drop(['SK_ID_CURR', 'TARGET'], errors='ignore'))
    
    return customer_data

def predict_score(customer_data):
    # Load the pre-trained model
    model_path = 'score/final_model.joblib'
    process_path = 'score/preprocessor.joblib'
    model = joblib.load(model_path)
    processors = joblib.load(process_path)
    df_predict = processors.transform(customer_data)
    df_predict = pd.DataFrame(df_predict,index=customer_data.index,columns=customer_data.columns)
    prediction_success = np.round(model.predict_proba(df_predict)[:, 0],3)[0]
    prediction_failure = np.round(model.predict_proba(df_predict)[:, 1],3)[0]
    #decision = model.predict(customer_data)[0]
    if prediction_failure > 0.25:
        decision = "Bank loan not granted"
    else:
        decision = "Bank loan granted"
        
    return decision, prediction_success, prediction_failure



def generate_shap_image(customer_data_raw):
        
    import joblib
    import shap
    import matplotlib.pyplot as plt
    import pandas as pd
    
    # Chemins pour le pipeline et l'explainer
    process_path = 'score/preprocessor.joblib'
    explainer_path = 'score/local_importance.joblib'
    
    # Charger le préprocesseur et l'explainer
    processors = joblib.load(process_path)
    explainer = joblib.load(explainer_path)
    
    # Prétraiter les données du client
    df_predict = processors.transform(customer_data_raw)
    df_predict = pd.DataFrame(df_predict, index=customer_data_raw.index, columns=customer_data_raw.columns)
    
    # Calculer les valeurs SHAP
    shap_values = explainer(df_predict)
    
    # Générer et enregistrer le graphique SHAP
    plt.figure()
    shap.waterfall_plot(shap_values[0], show=False)
    plot_path = 'static/shap_global_importance.png'
    plt.savefig(plot_path)
    plt.close()

    return plot_path

