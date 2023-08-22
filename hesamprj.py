import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError

# Load and preprocess your data
dataset = pd.read_csv('./dataset/final.csv')
X = dataset.drop(columns=['kick_power', 'kick_angle']).values
Y = dataset.loc[:, 'kick_power':].values

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Create a Sequential model
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]), # 64*70
    Dense(32, activation='relu'), # 64 * 32
    Dense(Y_train.shape[1])  # Output layer 32 * 3
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=MeanSquaredError())

# Train the model
history = model.fit(X_train, Y_train, epochs=100, batch_size=32,
                    validation_data=(X_test, Y_test))

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Save the trained model
model.save('keras_model.h5')
