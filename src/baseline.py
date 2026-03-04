import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from feature_engineering import load_dataset, preprocess_data, split_data


def reuse_distance_heuristic(df, threshold=10):

    predictions = []

    for reuse_distance in df["reuse_distance"]:
        pred = 1 if reuse_distance <= threshold else 0
        predictions.append(pred)

    return np.array(predictions)


def random_predictor(size):

    return np.random.randint(0, 2, size)


if __name__ == "__main__":

    # Load dataset
    df = load_dataset()

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    test_size = len(y_test)

    # Heuristic predictions
    heuristic_pred = reuse_distance_heuristic(df.iloc[-test_size:])

    # Random predictions
    random_pred = random_predictor(test_size)

    print("===== Reuse Distance Heuristic =====")
    print("Accuracy:", accuracy_score(y_test, heuristic_pred))
    print("Precision:", precision_score(y_test, heuristic_pred))
    print("Recall:", recall_score(y_test, heuristic_pred))
    print("F1 Score:", f1_score(y_test, heuristic_pred))

    print("\n===== Random Predictor =====")
    print("Accuracy:", accuracy_score(y_test, random_pred))
    print("Precision:", precision_score(y_test, random_pred))
    print("Recall:", recall_score(y_test, random_pred))
    print("F1 Score:", f1_score(y_test, random_pred))