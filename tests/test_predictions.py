# tests/test_predictions.py

import pytest
import joblib
from functions.functions import load_data, extract_features_from_custom, predict_score
import pandas as pd

@pytest.fixture(scope='module')
def load_test_data():
    df, customer_ids = load_data()
    return df, customer_ids

@pytest.fixture(scope='module')
def load_model():
    model_path = 'score/final_model.joblib'
    return joblib.load(model_path)

def test_predict_score_loan_not_granted(load_test_data, load_model):
    customer_id = 122701  # ID valide pour un refus de prêt
    df, _ = load_test_data
    customer_data = extract_features_from_custom(df, customer_id)
    decision, prediction_success, prediction_failure = predict_score(customer_data)
    assert decision == "Bank loan not granted"
    assert prediction_failure > 0.25

def test_predict_score_loan_granted(load_test_data, load_model):
    customer_id = 453454  # ID valide pour une acceptation de prêt
    df, _ = load_test_data
    customer_data = extract_features_from_custom(df, customer_id) 
    decision, prediction_success, prediction_failure = predict_score(customer_data)
    assert decision == "Bank loan granted"
    assert prediction_failure <= 0.25