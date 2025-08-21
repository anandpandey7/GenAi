# load data from sklearn
import tensorflow as tf
import numpy as np
from tensorflow.keras import models, layers
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# load data
data = fetch_california_housing()
X = data.data
y = data.target

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#build model

model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ]
)

# compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# train model
model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# evaluate model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: ${mae * 100000:.2f}")

# make predictions
sample_input = np.expand_dims(X_test[0], axis=0)
predictions = model.predict(sample_input)
print(f'Predicted house value: ${predictions[0][0] * 100000:.2f}')
