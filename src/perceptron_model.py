import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from src.feature_engineering import load_dataset, preprocess_data, split_data


class Perceptron:

    def __init__(self, learning_rate=0.1, epochs=20):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def predict_single(self, x):
        return 1 if np.dot(x, self.weights) + self.bias >= 0 else 0

    def predict(self, X):
        return np.array([self.predict_single(x) for x in X])

    def fit(self, X, y):

        n_features = X.shape[1]
        self.weights = np.zeros(n_features)

        for epoch in range(self.epochs):

            for i in range(len(X)):

                y_pred = self.predict_single(X[i])
                error = y.iloc[i] - y_pred

                self.weights += self.lr * error * X[i]
                self.bias += self.lr * error


if __name__ == "__main__":

    # Load dataset
    df = load_dataset()

    # Feature processing
    X, y = preprocess_data(df)

    # Train/Test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train model
    model = Perceptron()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Evaluation metrics
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))