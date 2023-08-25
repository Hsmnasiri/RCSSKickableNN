import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError, MeanAbsoluteError

# Load and preprocess your data
dataset = pd.read_csv('./dataset/all-csv-files.csv')
X = dataset.drop(columns=['kick_power', 'kick_angle']).values
Y = dataset.loc[:, 'kick_power':].values

# # Apply one-hot encoding to 'ball_area' column
# ball_area_column = dataset['ball_area'].values.reshape(-1, 1)
# encoder = OneHotEncoder()
# ball_area_encoded = encoder.fit_transform(ball_area_column)
# encoded_df = pd.DataFrame(ball_area_encoded.toarray(), columns=encoder.get_feature_names_out(['ball_area']))
# dataset = pd.concat([dataset, encoded_df], axis=1)
# dataset.drop(columns=['ball_area'], inplace=True)

# Standardize the data
scaler = StandardScaler()
scaled_X = scaler.fit_transform(X)

# # Perform PCA
# num_components = 10
# pca = PCA(n_components=num_components)
# pca_X = pca.fit_transform(scaled_X)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(scaled_X, Y, test_size=0.20, random_state=42)

# Create a Sequential model
model = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dense(32, activation='relu'),
    Dense(Y_train.shape[1])
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=MeanSquaredError(),
              metrics=[MeanAbsoluteError(), MeanSquaredError()])

# Train the model
history = model.fit(X_train, Y_train, epochs=100, batch_size=32,
                    validation_data=(X_test, Y_test))

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
# plt.yticks(np.arange(3000, 14000, 1000))
plt.savefig("./loss")



# Save the trained model
model.save('keras_model_with_pca.h5')
