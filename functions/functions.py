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



def generate_shap_image(customer_data_raw, max_display=10):
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
    
    # Générer et enregistrer le graphique SHAP avec le paramètre max_display
    plt.figure()
    shap.plots.waterfall(shap_values[0], max_display=max_display, show=False)
    plot_path = f'static/shap_global_importance_{customer_data_raw.index[0]}.png'
    plt.savefig(plot_path)
    plt.close()

    return plot_path




# Nouvelle fonction pour générer une grille de distributions
def generate_feature_distributions(df, customer_id, cols_per_row=2):
    from plotly.subplots import make_subplots
    import plotly.graph_objs as go
    import plotly.express as px
    customer_data = df[df['SK_ID_CURR'] == customer_id].squeeze()
    features = [col for col in df.columns if col != 'SK_ID_CURR']
    num_features = len(features)
    num_rows = (num_features // cols_per_row) + int(num_features % cols_per_row > 0)

    # Créer une grille de sous-graphiques
    fig = make_subplots(rows=num_rows, cols=cols_per_row, subplot_titles=features)

    for i, feature in enumerate(features):
        row = i // cols_per_row + 1
        col = i % cols_per_row + 1

        # Créer un histogramme pour la distribution de la feature
        hist = go.Histogram(x=df[feature], opacity=0.7, name=feature, showlegend=False)

        # Ajouter une ligne pour la valeur du client
        client_value = customer_data[feature]
        vline = go.Scatter(x=[client_value, client_value], y=[0, df[feature].value_counts().max()],
                           mode="lines", name="Valeur du client", line=dict(color="red"), showlegend=False)
        
        # Ajouter les traces à la sous-figure
        fig.add_trace(hist, row=row, col=col)
        fig.add_trace(vline, row=row, col=col)

        # Mettre à jour les axes pour chaque sous-figure
        fig.update_xaxes(title_text=feature, row=row, col=col)

    fig.update_layout(title_text="Distributions des Features par Rapport au Client Sélectionné", height=600*num_rows, showlegend=False)
    return fig

