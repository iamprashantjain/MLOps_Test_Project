# import pickle
# import os
# import numpy as np
# import pandas as pd
# import xgboost as xgb
# from sklearn.metrics import accuracy_score, classification_report
# import yaml

# n_estimators = yaml.safe_load(open('params.yaml','r'))['model_trainer']['n_estimators']
# learning_rate = yaml.safe_load(open('params.yaml','r'))['model_trainer']['learning_rate']

# # Load features
# train_data = pd.read_csv("data/features/train.csv")
# test_data = pd.read_csv("data/features/test.csv")

# # Split into features and labels
# x_train = train_data.drop(columns=["label"]).values
# y_train = train_data["label"].values

# x_test = test_data.drop(columns=["label"]).values
# y_test = test_data["label"].values

# # Train XGBoost model
# xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_estimators=n_estimators, learning_rate=learning_rate)

# xgb_model.fit(x_train, y_train)

# # Create model directory if it doesn't exist
# model_path = 'data/models'
# os.makedirs(model_path, exist_ok=True)

# # Save model
# with open(os.path.join(model_path, 'model.pkl'), 'wb') as f:
#     pickle.dump(xgb_model, f)



import os
import pickle
import yaml
import numpy as np
import pandas as pd
import xgboost as xgb
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)


def load_params(config_file='params.yaml'):
    logging.info(f"Loading configuration from {config_file}")
    with open(config_file, 'r') as file:
        params = yaml.safe_load(file)
    return params


def load_data(train_file="data/features/train.csv", test_file="data/features/test.csv"):
    logging.info(f"Loading training data from {train_file}")
    train_data = pd.read_csv(train_file)
    logging.info(f"Training data shape: {train_data.shape}")

    logging.info(f"Loading test data from {test_file}")
    test_data = pd.read_csv(test_file)
    logging.info(f"Test data shape: {test_data.shape}")

    return train_data, test_data


def preprocess_data(train_data, test_data):
    logging.info("Preprocessing data: splitting features and labels")
    x_train = train_data.drop(columns=["label"]).values
    y_train = train_data["label"].values

    x_test = test_data.drop(columns=["label"]).values
    y_test = test_data["label"].values

    return x_train, y_train, x_test, y_test


def train_model(x_train, y_train, n_estimators, learning_rate):
    logging.info("Training XGBoost model")
    model = xgb.XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        n_estimators=n_estimators,
        learning_rate=learning_rate
    )
    model.fit(x_train, y_train)
    logging.info("Model training completed")
    return model


def save_model(model, model_path="data/models", model_name="model.pkl"):
    logging.info(f"Saving model to {model_path}/{model_name}")
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, model_name), 'wb') as f:
        pickle.dump(model, f)
    logging.info("Model saved successfully")


def main():
    logging.info("Model training pipeline started")

    params = load_params()
    n_estimators = params['model_trainer']['n_estimators']
    learning_rate = params['model_trainer']['learning_rate']

    train_data, test_data = load_data()
    x_train, y_train, x_test, y_test = preprocess_data(train_data, test_data)
    model = train_model(x_train, y_train, n_estimators, learning_rate)
    save_model(model)

    logging.info("Model training pipeline finished")


if __name__ == "__main__":
    main()
