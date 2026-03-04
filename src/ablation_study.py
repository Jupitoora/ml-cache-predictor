from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.feature_engineering import load_dataset, preprocess_data, split_data

df = load_dataset()

X, y = preprocess_data(df)

X_train, X_test, y_train, y_test = split_data(X, y)

model = RandomForestClassifier()

model.fit(X_train, y_train)

pred = model.predict(X_test)

full_accuracy = accuracy_score(y_test, pred)

print("Accuracy with all features:", full_accuracy)

# Remove reuse_distance and gap
X_reduced = X[:, [0,1,3,5]]

X_train, X_test, y_train, y_test = split_data(X_reduced, y)

model.fit(X_train, y_train)

pred = model.predict(X_test)

reduced_accuracy = accuracy_score(y_test, pred)

print("Accuracy without reuse_distance and gap:", reduced_accuracy)