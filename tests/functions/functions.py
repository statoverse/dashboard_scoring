from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

def load_data():
    # Load the CSV file into a DataFrame
    data_path = 'data/customers.csv'
    df = pd.read_csv(data_path)
    # Extract the customer IDs (SK_ID_CURR) column
    customer_ids = df['SK_ID_CURR'].tolist()
    return df, customer_ids

#def extract_features_from_custom(df, customer_id):
#    # Get the row of the customer
#    customer_data = df[df['SK_ID_CURR'] == customer_id]
#    # Drop the SK_ID_CURR and TARGET columns
#    customer_data = customer_data.drop(columns=['SK_ID_CURR', 'TARGET'], errors='ignore')   
#   return customer_data

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
    model = joblib.load(model_path)
    
    # Predict the score (probability)
    prediction_success = np.round(model.predict_proba(customer_data)[:, 0],3)[0]
    prediction_failure = np.round(model.predict_proba(customer_data)[:, 1],3)[0]
    #decision = model.predict(customer_data)[0]
    if prediction_failure > 0.25:
        decision = "Bank loan not granted"
    else:
        decision = "Bank loan granted"
        
    return decision, prediction_success, prediction_failure


