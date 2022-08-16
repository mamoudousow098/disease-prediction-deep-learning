import os
import joblib
from tensorflow.keras.models import save_model, load_model
from django.apps import AppConfig
from django.conf import settings


class DiseaseApiConfig(AppConfig):
    MODEL_FILE = os.path.join(settings.MODELS, "disease_model")
    model = load_model(MODEL_FILE)
    name = 'disease_api'
