# import numpy as np 
# import pandas as pd 
# from sklearn.feature_extraction.text import CountVectorizer
# import os
# import yaml

# max_features = yaml.safe_load(open('params.yaml','r'))['feature_engineering']['max_features']

# train_data = pd.read_csv("data/processed/train.csv")
# test_data = pd.read_csv("data/processed/test.csv")

# # Fill NaNs and convert to list of strings
# X_train = train_data['content'].fillna("").tolist()
# y_train = train_data['sentiment'].values

# X_test = test_data['content'].fillna("").tolist()
# y_test = test_data['sentiment'].values

# #bow vectorizer
# vectorizer = CountVectorizer(max_features=max_features)

# x_train_bow = vectorizer.fit_transform(X_train)
# x_test_bow = vectorizer.transform(X_test)

# train_df = pd.DataFrame(x_train_bow.toarray())
# train_df['label'] = y_train

# test_df = pd.DataFrame(x_test_bow.toarray())
# test_df['label'] = y_test

# data_path = os.path.join('data', 'features')
# os.makedirs(data_path, exist_ok=True)

# train_df.to_csv(os.path.join(data_path, "train.csv"), index=False)
# test_df.to_csv(os.path.join(data_path, "test.csv"), index=False)


import os
import numpy as np
import pandas as pd
import yaml
import logging
from sklearn.feature_extraction.text import CountVectorizer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("feature_engineering.log"),
        logging.StreamHandler()
    ]
)

def load_params(file_path="params.yaml"):
    logging.info(f"Loading parameters from {file_path}")
    with open(file_path, 'r') as file:
        params = yaml.safe_load(file)
    return params


def load_data(train_path="data/processed/train.csv", test_path="data/processed/test.csv"):
    logging.info(f"Loading train data from {train_path}")
    train_data = pd.read_csv(train_path)
    logging.info(f"Train data shape: {train_data.shape}")

    logging.info(f"Loading test data from {test_path}")
    test_data = pd.read_csv(test_path)
    logging.info(f"Test data shape: {test_data.shape}")

    return train_data, test_data


def preprocess_data(train_data, test_data):
    logging.info("Preprocessing: filling missing values and extracting labels.")
    X_train = train_data['content'].fillna("").tolist()
    y_train = train_data['sentiment'].values

    X_test = test_data['content'].fillna("").tolist()
    y_test = test_data['sentiment'].values

    return X_train, y_train, X_test, y_test


def vectorize_data(X_train, X_test, max_features):
    logging.info(f"Vectorizing data using CountVectorizer with max_features={max_features}")
    vectorizer = CountVectorizer(max_features=max_features)
    
    x_train_bow = vectorizer.fit_transform(X_train)
    x_test_bow = vectorizer.transform(X_test)

    logging.info(f"Train BOW shape: {x_train_bow.shape}, Test BOW shape: {x_test_bow.shape}")
    return x_train_bow, x_test_bow


def create_dataframe(x_train_bow, x_test_bow, y_train, y_test):
    logging.info("Creating DataFrames from vectorized data")
    train_df = pd.DataFrame(x_train_bow.toarray())
    train_df['label'] = y_train

    test_df = pd.DataFrame(x_test_bow.toarray())
    test_df['label'] = y_test

    return train_df, test_df


def save_data(train_df, test_df, data_path="data/features"):
    logging.info(f"Saving feature data to {data_path}")
    os.makedirs(data_path, exist_ok=True)

    train_path = os.path.join(data_path, "train.csv")
    test_path = os.path.join(data_path, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logging.info(f"Saved train features to {train_path}")
    logging.info(f"Saved test features to {test_path}")


def main():
    logging.info("Feature engineering pipeline started.")
    
    params = load_params()
    max_features = params['feature_engineering']['max_features']

    train_data, test_data = load_data()
    X_train, y_train, X_test, y_test = preprocess_data(train_data, test_data)
    x_train_bow, x_test_bow = vectorize_data(X_train, X_test, max_features)
    train_df, test_df = create_dataframe(x_train_bow, x_test_bow, y_train, y_test)
    save_data(train_df, test_df)

    logging.info("Feature engineering pipeline completed successfully.")


if __name__ == "__main__":
    main()
