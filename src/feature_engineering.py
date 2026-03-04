import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def load_dataset(path="data/raw/memory_trace.csv"):
    
    df = pd.read_csv(path)

    return df


def preprocess_data(df):

    # Separate features and labels
    X = df.drop("label", axis=1)
    y = df["label"]

    # Normalize features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    df = load_dataset()

    X, y = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Dataset loaded and processed")

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))