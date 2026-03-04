import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.feature_engineering import load_dataset, preprocess_data, split_data
from src.perceptron_model import Perceptron


def evaluate(name, model, X_train, y_train, X_test, y_test):

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred)
    }


def main():

    df = load_dataset()

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    results = []

    # Perceptron
    perceptron = Perceptron()
    results.append(evaluate("Perceptron", perceptron, X_train, y_train, X_test, y_test))

    # Logistic Regression
    logreg = LogisticRegression(max_iter=1000)
    results.append(evaluate("Logistic Regression", logreg, X_train, y_train, X_test, y_test))

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100)
    results.append(evaluate("Random Forest", rf, X_train, y_train, X_test, y_test))

    # SVM
    svm = SVC()
    results.append(evaluate("SVM", svm, X_train, y_train, X_test, y_test))

    df_results = pd.DataFrame(results)

    print(df_results)

    # Save results
    df_results.to_csv(
        "results/metrics/model_results.csv",
        index=False
    )


if __name__ == "__main__":
    main()