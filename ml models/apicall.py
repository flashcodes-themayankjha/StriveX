from fastapi import FastAPI
import numpy as np
import tensorflow as tf
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import uvicorn

app = FastAPI()

# Load trained model
model = load_model("injury_risk_lstm.keras")

# Define input format
class AthleteData(BaseModel):
    Strength: float
    Agility: float
    Endurance: float
    Reaction_Time: float
    Eye_Sight: float
    Yoyo_Test: float
    Achievements: float
    Fitness_Score: float

# MinMaxScaler to normalize data (use the same scale from training)
scaler = MinMaxScaler()
scaler.fit([[50, 50, 50, 0.2, 0.8, 5, 0, 60], [100, 100, 100, 1.5, 1.2, 25, 10, 100]])  

@app.post("/predict")
def predict(data: AthleteData):
    input_data = np.array([[data.Strength, data.Agility, data.Endurance, data.Reaction_Time, 
                            data.Eye_Sight, data.Yoyo_Test, data.Achievements, data.Fitness_Score]])
    input_scaled = scaler.transform(input_data).reshape((1, 1, 8))

    # Predict Injury Risk
    prediction = model.predict(input_scaled)
    return {"injury_risk": float(prediction[0, 0])}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
