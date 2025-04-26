# import os
# import json
# import pandas as pd
# import joblib
# from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report

# # Load features
# test_data = pd.read_csv("data/features/test.csv")

# # Split features and labels
# x_test = test_data.drop(columns=["label"]).values
# y_test = test_data["label"].values

# # Load trained model
# model = joblib.load("data/models/model.pkl")

# # Predict on test data
# y_pred = model.predict(x_test)

# # Evaluate performance
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)
# roc_auc = roc_auc_score(y_test, y_pred)

# # Store metrics
# metrics = {
#     'accuracy': accuracy,
#     'precision': precision,
#     'recall': recall,
#     'roc_auc': roc_auc
# }

# # Ensure 'reports/' directory exists
# os.makedirs("data/reports", exist_ok=True)

# # Save metrics to JSON
# with open("data/reports/metrics.json", "w") as f:
#     json.dump(metrics, f, indent=4)



import os
import json
import pandas as pd
import joblib
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("model_evaluation.log"),
        logging.StreamHandler()
    ]
)

def load_test_data(file_path="data/features/test.csv"):
    logging.info(f"Loading test data from {file_path}")
    test_data = pd.read_csv(file_path)
    x_test = test_data.drop(columns=["label"]).values
    y_test = test_data["label"].values
    logging.info(f"Test data shape: {test_data.shape}")
    return x_test, y_test


def load_model(model_path="data/models/model.pkl"):
    logging.info(f"Loading model from {model_path}")
    return joblib.load(model_path)


def predict(model, x_test):
    logging.info("Making predictions on test data")
    return model.predict(x_test)


def evaluate_model(y_test, y_pred):
    logging.info("Evaluating model performance")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    }

    logging.info(f"Evaluation metrics: {metrics}")
    return metrics


def save_metrics(metrics, output_dir="data/reports", output_file="metrics.json"):
    logging.info(f"Saving metrics to {output_dir}/{output_file}")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, output_file), "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info("Metrics saved successfully.")


def main():
    logging.info("Model evaluation started.")

    x_test, y_test = load_test_data()
    model = load_model()
    y_pred = predict(model, x_test)
    metrics = evaluate_model(y_test, y_pred)
    save_metrics(metrics)

    logging.info("Model evaluation completed successfully.")


if __name__ == "__main__":
    main()
