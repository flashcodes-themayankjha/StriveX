import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Explicitly define the loss function
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}

# Load the model with custom objects
model = load_model("injury_risk_lstm.h5", custom_objects=custom_objects)

print("Model Loaded Successfully!")

# Define the same feature scaling process
scaler = MinMaxScaler()

# Sample input data for testing (Replace with actual user input)

sample_data = np.array([[80, 85, 75, 0.9, 1.0, 15, 5, 85]])  # Example athlete data
sample_data_scaled = scaler.fit_transform(sample_data)  # Scale the data
sample_data_scaled = sample_data_scaled.reshape((1, 1, sample_data.shape[1]))  # Reshape for LSTM

# Make a prediction

predicted_risk = model.predict(sample_data_scaled)
print(f"Predicted Injury Risk: {predicted_risk[0][0]:.4f}")  # Display the predicted risk
