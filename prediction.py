from fastapi import FastAPI
import joblib

app = FastAPI()

model = joblib.load("placement_model.pkl")

@app.get("/")
def home():
    return {"message": "PlacementPredict API"}

@app.post("/predict")
def predict(features: dict):
    prediction = model.predict([list(features.values())])
    return {"prediction": int(prediction[0])}