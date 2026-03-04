import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from feature_engineering import load_dataset, preprocess_data, split_data


def evaluate_model(name, model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print(f"\n===== {name} =====")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred))
    print("Recall:", recall_score(y_test, y_pred))
    print("F1 Score:", f1_score(y_test, y_pred))


if __name__ == "__main__":

    # Load dataset
    df = load_dataset()

    # Feature processing
    X, y = preprocess_data(df)

    # Train/Test split
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Models
    logistic = LogisticRegression(max_iter=1000)
    random_forest = RandomForestClassifier(n_estimators=100)
    svm = SVC()

    evaluate_model("Logistic Regression", logistic, X_train, y_train, X_test, y_test)

    evaluate_model("Random Forest", random_forest, X_train, y_train, X_test, y_test)

    evaluate_model("Support Vector Machine", svm, X_train, y_train, X_test, y_test)