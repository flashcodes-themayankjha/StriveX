import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import load_model
import tensorflow as tf

# Define custom objects for loading
custom_objects = {"mse": tf.keras.losses.MeanSquaredError()}

# Load model correctly
model = load_model("injury_risk_lstm.h5", custom_objects=custom_objects)

model.compile(optimizer='adam', loss=tf.keras.losses.MeanSquaredError(), metrics=['mae'])

# Generating synthetic data
np.random.seed(42)
num_samples = 1000

data = {
    "Strength": np.random.randint(50, 100, num_samples),
    "Agility": np.random.randint(50, 100, num_samples),
    "Endurance": np.random.randint(50, 100, num_samples),
    "Reaction_Time": np.random.uniform(0.2, 1.5, num_samples),
    "Eye_Sight": np.random.uniform(0.8, 1.2, num_samples),
    "Yoyo_Test": np.random.randint(5, 25, num_samples),
    "Achievements": np.random.randint(0, 10, num_samples),
    "Fitness_Score": np.random.randint(60, 100, num_samples),
    "Injury_Risk": np.random.uniform(0, 1, num_samples),
}

df = pd.DataFrame(data)

# Splitting features and target
features = ["Strength", "Agility", "Endurance", "Reaction_Time", "Eye_Sight", "Yoyo_Test", "Achievements", "Fitness_Score"]
X = df[features].values
y = df["Injury_Risk"].values

# Normalize and reshape data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define LSTM model
model = Sequential([
    LSTM(64, activation='relu', return_sequences=True, input_shape=(1, len(features))),
    Dropout(0.2),
    LSTM(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Sigmoid since risk is between 0 and 1
])

# Compile and train model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=1)

# Save the model
model.save("injury_risk_lstm.keras")
print("Model saved as injury_risk_lstm.keras")

# Load model and predict
model = load_model("injury_risk_lstm.keras")
y_pred = model.predict(X_test).flatten()

# Plot Actual vs Predicted
plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test)), y_test, label="Actual Injury Risk", color='blue', alpha=0.6)
plt.scatter(range(len(y_pred)), y_pred, label="Predicted Injury Risk", color='red', alpha=0.6)
plt.xlabel("Test Sample Index")
plt.ylabel("Injury Risk Score")
plt.legend()
plt.title("Actual vs Predicted Injury Risk")
plt.show()
