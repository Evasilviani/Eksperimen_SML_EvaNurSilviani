
# =========================================================
# AUTOMATE PREPROCESSING HEART DISEASE DATASET
# =========================================================

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def preprocess_data(input_path, output_path):

    # LOAD DATASET
    df = pd.read_csv(input_path)

    # REMOVE DUPLICATES
    df = df.drop_duplicates()

    # SPLIT FEATURE DAN TARGET
    X = df.drop('target', axis=1)
    y = df['target']

    # STANDARDIZATION
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # TRAIN TEST SPLIT
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=0.2,
        random_state=42
    )

    # SAVE PREPROCESSED DATA
    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
    processed_df['target'] = y.values

    processed_df.to_csv(output_path, index=False)

    print("Preprocessing berhasil")
    print("Dataset tersimpan:", output_path)

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":

    preprocess_data(
        input_path='heart.csv',
        output_path='heart_preprocessing.csv'
    )
