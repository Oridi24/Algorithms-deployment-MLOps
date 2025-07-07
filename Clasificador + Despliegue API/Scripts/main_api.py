# main_api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Cargar modelo y scaler entrenados
model = joblib.load("wine_model_multiclass.pkl")
scaler = joblib.load("wine_scaler.pkl")

app = FastAPI(title="Wine Classifier API", description="Predict wine class from chemical features", version="1.0")

# Definir la estructura del input
class WineInput(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Wine Classifier API "}

@app.post("/predict")
def predict(input_data: WineInput):
    features = np.array([[
        input_data.alcohol,
        input_data.malic_acid,
        input_data.ash,
        input_data.alcalinity_of_ash,
        input_data.magnesium,
        input_data.total_phenols,
        input_data.flavanoids,
        input_data.nonflavanoid_phenols,
        input_data.proanthocyanins,
        input_data.color_intensity,
        input_data.hue,
        input_data.od280_od315_of_diluted_wines,
        input_data.proline
    ]])

    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]

    class_names = ["class_0", "class_1", "class_2"]
    return {"predicted_class": class_names[prediction]}