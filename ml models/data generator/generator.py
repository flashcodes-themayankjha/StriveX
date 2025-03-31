# Re-run the code since execution state was reset

import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

# Generate 1000 synthetic data points
num_samples = 1000

data = {
    "Stamina": np.random.randint(40, 100, num_samples),  # Between 40-100
    "Endurance": np.random.randint(40, 100, num_samples),
    "Agility": np.random.randint(40, 100, num_samples),
    "Reaction Time": np.random.uniform(0.2, 1.5, num_samples),  # Between 0.2s - 1.5s
    "Eye Sight": np.random.uniform(0.8, 1.5, num_samples),  # 1.0 is normal, variation included
    "Yoyo Test Score": np.random.randint(10, 50, num_samples),
    "Achievements": np.random.randint(20, 100, num_samples),  # Weighted score 0-100
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Compute Fitness Score based on the given formula
weighted_sports_index = np.random.uniform(0.5, 1.5, size=(7,))  # Random weights for attributes

df["Fitness Score"] = (
    weighted_sports_index[0] * df["Stamina"]
    + weighted_sports_index[1] * df["Endurance"]
    + weighted_sports_index[2] * df["Agility"]
    + weighted_sports_index[3] * (1 / df["Reaction Time"])  # Lower reaction time is better
    + weighted_sports_index[4] * (1 / df["Eye Sight"])  # Lower eyesight value is better
    + weighted_sports_index[5] * df["Yoyo Test Score"]
)

# Compute Performance Score
df["Performance Score"] = df["Achievements"] + df["Fitness Score"]

# Display first few rows
df.head()
