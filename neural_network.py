# neural_network.py
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        np.random.seed(42)
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        self.lr = lr

    def relu(self, x): return np.maximum(0, x)
    def relu_deriv(self, x): return (x > 0).astype(float)
    def softmax(self, x): return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def cross_entropy(self, y_true, y_pred):
        return -np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))

    def train(self, X, y, epochs=500):
        losses = []
        for _ in range(epochs):
            z1 = np.dot(X, self.W1) + self.b1
            a1 = self.relu(z1)
            z2 = np.dot(a1, self.W2) + self.b2
            y_pred = self.softmax(z2)

            loss = self.cross_entropy(y, y_pred)
            losses.append(loss)

            dZ2 = y_pred - y
            dW2 = np.dot(a1.T, dZ2) / X.shape[0]
            db2 = np.mean(dZ2, axis=0, keepdims=True)
            dA1 = np.dot(dZ2, self.W2.T)
            dZ1 = dA1 * self.relu_deriv(z1)
            dW1 = np.dot(X.T, dZ1) / X.shape[0]
            db1 = np.mean(dZ1, axis=0, keepdims=True)

            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2

        return losses

    def predict(self, X):
        a1 = self.relu(np.dot(X, self.W1) + self.b1)
        y_pred = self.softmax(np.dot(a1, self.W2) + self.b2)
        return np.argmax(y_pred, axis=1)

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    y = iris.target.reshape(-1, 1)

    encoder = OneHotEncoder(sparse_output=False)
    y = encoder.fit_transform(y)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = NeuralNetwork(input_size=4, hidden_size=8, output_size=3, lr=0.01)
    losses = model.train(X_train, y_train, epochs=500)

    y_pred = model.predict(X_test)
    acc = np.mean(np.argmax(y_test, axis=1) == y_pred)
    print(f"Test Accuracy: {acc:.2f}")
