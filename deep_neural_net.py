import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

# Dense layer with dynamic neuron creation
class Layer_Dense:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        self.output = None  # Initialize the output attribute

    # Forward pass
    def forward(self, inputs):
        # Calculates output values from inputs, weights, and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    # Add new neurons dynamically
    def add_neurons(self, n_new_neurons):
        # Generate random weights for new neurons with the same number of input connections
        new_weights = 0.10 * np.random.randn(self.n_inputs, n_new_neurons)

        # Append the new weights to the existing weights
        self.weights = np.hstack((self.weights, new_weights))

        # Increase the number of neurons in biases
        self.biases = np.zeros((1, self.weights.shape[1]))

# ReLU Activation
class Activation_ReLU:
    # Forward pass
    def forward(self, inputs):
        # Calculate and return output values from input
        return np.maximum(0, inputs)

# Assign the number of samples and iterations as integers
samples = range(5)  # Creates a range from 0 to 4 (inclusive)
iterations = 10  # You can change this value as needed

# Create dataset
X, y = spiral_data(samples=len(samples), classes=3)

# Split the data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create the first dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)

# Create ReLU activation for the first layer
activation1 = Activation_ReLU()

# Create the second dense layer with 3 input features (matching the output of the first layer)
# and 3 output values
dense2 = Layer_Dense(3, 3)

# Create the third dense layer with 3 input features (matching the output of the second layer)
# and 3 output values
dense3 = Layer_Dense(3, 3)

# Initialize a counter to keep track of the number of samples processed
sample_count = 0

# Iterate through the dataset in batches of 5 samples for the specified number of iterations
for _ in range(iterations):
    for i in samples:
        start_idx = i * 5
        end_idx = (i + 1) * 5

        batch_X = X[start_idx:end_idx]

        # Perform a forward pass with the first dense layer
        dense1.forward(batch_X)

        # Perform a forward pass with the second dense layer
        dense2.forward(dense1.output)

        # Perform a forward pass with the third dense layer
        dense3.forward(dense2.output)

        # Increment the sample count
        sample_count += len(batch_X)

        # Check if it's time to add new neurons (every 100 samples) for the second and third dense layers
        if sample_count % 100 == 0:
            # Add new neurons (e.g., 2 new neurons) to the second and third dense layers
            dense2.add_neurons(0)
            dense3.add_neurons(2)

        # Apply ReLU activation for the second and third dense layers
        output_activation2 = activation1.forward(dense2.output)
        output_activation3 = activation1.forward(dense3.output)

        # Print output of the current batch for all three layers
        print(f"Output after processing {sample_count} samples (Layer 1):")
        print(dense1.output[:5])

        print(f"Output after processing {sample_count} samples (Layer 2):")
        print(output_activation2[:5])

        print(f"Output after processing {sample_count} samples (Layer 3):")
        print(output_activation3[:5])

        # Save the output to a CSV file
        output_filename = f"output_iteration_{iterations}_sample_{sample_count}.csv"
        np.savetxt(output_filename, dense3.output, delimiter=',')

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train)

crv = cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy")
print(f"Cross-Validation Score: {crv}")

model = Sequential([
    Dense(64, input_shape=(2,), activation='relu', name='Layer_1'),
    Dense(64, activation='relu', name='Layer_2'),
    Dense(64, activation='relu', name='Layer_3')
])

model = Sequential([
    Dense(64, input_shape=(2,), activation='relu', name='Layer_1'),
    Dense(64, activation='relu', name='Layer_2'),
    Dense(64, activation='relu', name='Layer_3'),
    Dense(64, activation='relu', name='Layer_4'),
    Dense(64, activation='relu', name='Layer_5')
])

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=optimizer, loss='mse', metrics=['accuracy'])

log_dir = "logs/fit/"  # Directory to store TensorBoard logs
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train your model
model.fit(X_train, y_train, epochs=90, batch_size=64, callbacks=[tensorboard_callback])  
