import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

from src.feature_engineering import load_dataset, preprocess_data

df = load_dataset()

X, y = preprocess_data(df)

model = RandomForestClassifier()

model.fit(X, y)

importance = model.feature_importances_

features = [
    "pc",
    "address",
    "reuse_distance",
    "frequency",
    "gap",
    "access_type"
]

plt.figure()

plt.bar(features, importance)

plt.title("Feature Importance")

plt.ylabel("Importance Score")

plt.xticks(rotation=45)

plt.savefig("results/plots/feature_importance.png")

plt.show()