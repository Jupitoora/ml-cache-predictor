import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from feature_engineering import load_dataset, preprocess_data, split_data
from perceptron_model import Perceptron


def train_models(X_train, y_train, X_test, y_test):

    results = {}

    # Perceptron
    perceptron = Perceptron()
    perceptron.fit(X_train, y_train)
    y_pred = perceptron.predict(X_test)
    results["Perceptron"] = accuracy_score(y_test, y_pred)

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)
    y_pred = log_reg.predict(X_test)
    results["Logistic Regression"] = accuracy_score(y_test, y_pred)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results["Random Forest"] = accuracy_score(y_test, y_pred)

    # SVM
    svm = SVC()
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    results["SVM"] = accuracy_score(y_test, y_pred)

    return results


def plot_accuracy(results):

    models = list(results.keys())
    scores = list(results.values())

    plt.figure(figsize=(8,5))
    plt.bar(models, scores)

    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0,1)

    plt.savefig("results/plots/model_accuracy.png")

    plt.show()


if __name__ == "__main__":

    df = load_dataset()

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    results = train_models(X_train, y_train, X_test, y_test)

    print(results)

    plot_accuracy(results)