# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# import yaml

# test_size = yaml.safe_load(open('params.yaml','r'))['data_ingestion']['test_size']

# # Load dataset
# url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'
# df = pd.read_csv(url)

# # Drop unnecessary column
# df.drop(columns=['tweet_id'], inplace=True)

# # Filter for binary classification and encode labels
# final_df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
# final_df['sentiment'] = final_df['sentiment'].replace({'happiness': 1, 'sadness': 0})

# # Split the data
# train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)

# # Create directory if it doesn't exist
# data_path = os.path.join('data', 'raw')
# os.makedirs(data_path, exist_ok=True)

# # Save to CSV
# train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
# test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)

# print(f"Train and test data saved in: {data_path}")


import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data_ingestion.log"),
        logging.StreamHandler()
    ]
)


def load_config(path='params.yaml'):
    logging.info(f"Loading configuration from {path}...")
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    test_size = config['data_ingestion']['test_size']
    logging.info(f"Test size loaded: {test_size}")
    return test_size


def load_dataset(url):
    logging.info(f"Loading dataset from URL: {url}")
    df = pd.read_csv(url)
    logging.info(f"Dataset loaded with shape: {df.shape}")
    return df


def preprocess_data(df):
    logging.info("Starting preprocessing...")
    initial_rows = df.shape[0]
    df = df.drop(columns=['tweet_id'])
    df = df[df['sentiment'].isin(['happiness', 'sadness'])].copy()
    filtered_rows = df.shape[0]
    df['sentiment'] = df['sentiment'].replace({'happiness': 1, 'sadness': 0})
    logging.info(f"Preprocessing complete. Rows before: {initial_rows}, after filtering: {filtered_rows}")
    return df


def split_data(df, test_size):
    logging.info(f"Splitting data with test size = {test_size}")
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)
    logging.info(f"Train set size: {train_df.shape}, Test set size: {test_df.shape}")
    return train_df, test_df


def save_data(train_df, test_df, output_dir='data/raw'):
    logging.info(f"Saving train and test data to '{output_dir}'...")
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train.csv')
    test_path = os.path.join(output_dir, 'test.csv')
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    logging.info(f"Train data saved to {train_path}")
    logging.info(f"Test data saved to {test_path}")


def main():
    logging.info("Starting data ingestion pipeline...")
    test_size = load_config()
    url = 'https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv'

    df = load_dataset(url)
    df = df.sample(100)
    df = preprocess_data(df)
    train_df, test_df = split_data(df, test_size)
    save_data(train_df, test_df)
    logging.info("Data ingestion pipeline completed successfully.")


if __name__ == "__main__":
    main()
