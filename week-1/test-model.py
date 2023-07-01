import tensorflow as tf
from tensorflow import keras
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("version: ", tf.__version__)

# a sequential model with 1 neuron and an input shape of 1
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

x = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y= np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# fitting the model for 500 epochs
model.fit(x, y, epochs=500)

print(model.predict([10.0]))

