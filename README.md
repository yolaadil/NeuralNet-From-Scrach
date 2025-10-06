# 🧠 Neural Network from Scratch

An educational notebook that builds a simple neural network from scratch using NumPy,
trained on the Iris dataset. The project helps learners understand how neural networks
work internally — including forward propagation, loss computation, backpropagation,
and gradient descent — without using frameworks like TensorFlow or PyTorch.



# 📘 Overview

 This notebook covers:
- Implementing a two-layer neural network manually
- Training it step-by-step using NumPy
 - Visualizing the loss curve over time
- Evaluating its accuracy on the Iris dataset

 You can run this notebook directly in Google Colab — it’s self-contained.


# ------------------------------------------------
# ⚙️ 1. Import Required Libraries
# ------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ------------------------------------------------
# 📊 2. Load and Prepare the Iris Dataset
# ------------------------------------------------
# The Iris dataset contains 150 samples with 4 features each.
# There are 3 classes (Setosa, Versicolor, Virginica).

iris = load_iris()
X = iris.data                      # Input features
y = iris.target.reshape(-1, 1)     # Class labels reshaped to column vector

# Standardize features (zero mean, unit variance)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# One-hot encode labels (e.g., class 1 -> [0,1,0])
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# Split data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ------------------------------------------------
# 🧮 3. Define Neural Network Architecture
# ------------------------------------------------
# A simple 2-layer neural network:
# Input: 4 neurons (features)
# Hidden: 8 neurons (sigmoid activation)
# Output: 3 neurons (softmax activation for classification)

n_inputs = X_train.shape[1]    # 4 features
n_hidden = 8                   # hidden neurons
n_outputs = y_train.shape[1]   # 3 output classes

# Initialize weights and biases
np.random.seed(42)
W1 = np.random.randn(n_inputs, n_hidden)
b1 = np.zeros((1, n_hidden))
W2 = np.random.randn(n_hidden, n_outputs)
b2 = np.zeros((1, n_outputs))


# ------------------------------------------------
# ⚙️ 4. Define Activation and Helper Functions
# ------------------------------------------------

def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))

def sigmoid_derivative(a):
    """Derivative of sigmoid function."""
    return a * (1 - a)

def softmax(z):
    """Softmax activation for multi-class classification."""
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def compute_loss(y_true, y_pred):
    """Cross-entropy loss."""
    m = y_true.shape[0]
    return -np.sum(y_true * np.log(y_pred + 1e-9)) / m


# ------------------------------------------------
# 🔁 5. Training Loop: Forward + Backpropagation
# ------------------------------------------------
# We'll use basic gradient descent to minimize the loss.

learning_rate = 0.1
epochs = 1000
losses = []

for epoch in range(epochs):
    # --- Forward propagation ---
    z1 = np.dot(X_train, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = softmax(z2)

    # --- Compute loss ---
    loss = compute_loss(y_train, a2)
    losses.append(loss)

    # --- Backpropagation ---
    m = y_train.shape[0]
    dz2 = a2 - y_train
    dW2 = np.dot(a1.T, dz2) / m
    db2 = np.sum(dz2, axis=0, keepdims=True) / m

    dz1 = np.dot(dz2, W2.T) * sigmoid_derivative(a1)
    dW1 = np.dot(X_train.T, dz1) / m
    db1 = np.sum(dz1, axis=0, keepdims=True) / m

    # --- Update weights ---
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # --- Print progress every 100 epochs ---
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")


# ------------------------------------------------
# 📉 6. Visualize Training Loss
# ------------------------------------------------
plt.plot(losses)
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()


# ------------------------------------------------
# 🧪 7. Evaluate the Model on Test Data
# ------------------------------------------------
# Perform forward pass on test set and compute accuracy.

z1_test = np.dot(X_test, W1) + b1
a1_test = sigmoid(z1_test)
z2_test = np.dot(a1_test, W2) + b2
a2_test = softmax(z2_test)

# Convert predictions to class indices
y_pred = np.argmax(a2_test, axis=1)
y_true = np.argmax(y_test, axis=1)
accuracy = np.mean(y_pred == y_true)

print(f"✅ Test Accuracy: {accuracy * 100:.2f}%")


# ------------------------------------------------
# 🧠 8. Reflection and Next Steps
# ------------------------------------------------
# This network achieves around 90–95% accuracy on Iris without deep learning frameworks.
#
# ✅ Key Takeaways:
# - Forward propagation passes input through the network
# - Backpropagation adjusts weights to minimize loss
# - Gradient descent iteratively improves performance
#
# 💡 Next Steps:
# - Try ReLU instead of sigmoid
# - Add more layers
# - Implement an optimizer (Adam, Momentum)
# - Apply to another dataset (e.g., MNIST)
