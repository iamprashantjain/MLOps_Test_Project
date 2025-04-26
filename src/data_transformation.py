# import os
# import numpy as np 
# import pandas as pd 
# import re 
# import string
# import nltk 
# import string 
# from nltk. corpus import stopwords 
# from nltk.stem import SnowballStemmer, WordNetLemmatizer 
# from sklearn.feature_extraction.text import CountVectorizer


# nltk.download('wordnet')
# nltk.download('stopwords')

# train_data = pd.read_csv("data/raw/train.csv")
# test_data = pd.read_csv("data/raw/test.csv")


# def lemmatization(text):
#     lemmatizer = WordNetLemmatizer()
#     text = text.split()
#     text = [lemmatizer.lemmatize(y) for y in text]
#     return " ".join(text)

# def remove_stop_words (text):
#     stop_words = set(stopwords.words('english'))
#     Text = [i for i in str(text).split() if i not in stop_words]
#     return " ".join(Text)

# def removing_numbers(text):
#     text = "".join([i for i in text if not i.isdigit()])
#     return text

# def lower_case(text):
#     text = text.split()
#     text = [y.lower() for y in text]
#     return " ".join(text)


# def removing_punctuations(text):
#     # Remove punctuation using regex and string.punctuation
#     text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    
#     #remove extra whitespace
#     text = re.sub('\s+',' ', text)
#     text = " ".join(text.split())
#     return text.strip()


# def removing_urls(text):
#     url_pattern = re.compile(r'https://\S+|www\.\S+')
#     return url_pattern.sub(r'', text)

# def remove_small_sentences(df):
#     for i in range(len(df)):
#         if len(df.text.iloc[i].split()) < 3:
#             df.text.iloc[1] = np.nan
            
            
# def normalize_text(df):
#     df.content = df.content.apply(lambda content : lower_case(content))
#     df.content = df.content.apply(lambda content : remove_stop_words(content))
#     df.content = df.content.apply(lambda content : removing_numbers(content))
#     df.content = df.content.apply(lambda content : removing_punctuations(content))
#     df.content = df.content.apply(lambda content : removing_urls(content))
#     df.content = df.content.apply(lambda content : lemmatization(content))    
#     return df


# def normalize_sentence(sentence):
#     sentence = lower_case(sentence)
#     sentence = remove_stop_words(sentence)
#     sentence = removing_numbers(sentence)
#     sentence = removing_punctuations(sentence)
#     sentence = removing_urls(sentence)
#     sentence = lemmatization(sentence)
#     return sentence


# train_data = normalize_text(train_data)
# test_data = normalize_text(test_data)

# data_path = os.path.join('data', 'processed')
# os.makedirs(data_path, exist_ok=True)

# train_data.to_csv(os.path.join(data_path, "train.csv"), index=False)
# test_data.to_csv(os.path.join(data_path, "test.csv"), index=False)



import os
import re
import string
import numpy as np
import pandas as pd
import nltk
import logging
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download NLTK resources
nltk.download('wordnet')
nltk.download('stopwords')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("data_transformation.log"),
        logging.StreamHandler()
    ]
)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    return " ".join([lemmatizer.lemmatize(word) for word in words])

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    return " ".join([word for word in text.split() if word not in stop_words])

def remove_numbers(text):
    return "".join([char for char in text if not char.isdigit()])

def lower_case(text):
    return text.lower()

def remove_punctuations(text):
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    return re.sub('\s+', ' ', text).strip()

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(text):
    text = lower_case(text)
    text = remove_stop_words(text)
    text = remove_numbers(text)
    text = remove_punctuations(text)
    text = remove_urls(text)
    text = lemmatize_text(text)
    return text

def load_data(train_path, test_path):
    logging.info(f"Loading data from {train_path} and {test_path}")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    logging.info(f"Train shape: {train_data.shape}, Test shape: {test_data.shape}")
    return train_data, test_data

def save_data(train_data, test_data, output_path="data/processed"):
    logging.info(f"Saving processed data to {output_path}")
    os.makedirs(output_path, exist_ok=True)
    train_data.to_csv(os.path.join(output_path, "train.csv"), index=False)
    test_data.to_csv(os.path.join(output_path, "test.csv"), index=False)
    logging.info("Processed data saved successfully.")

def remove_small_sentences(df):
    before = df.shape[0]
    df.loc[df['content'].str.split().str.len() < 3, 'content'] = np.nan
    after = df['content'].notna().sum()
    logging.info(f"Removed {before - after} short sentences (less than 3 words)")
    return df

def preprocess_data(train_data, test_data):
    logging.info("Starting preprocessing of train and test data...")
    train_data['content'] = train_data['content'].astype(str).apply(normalize_text)
    test_data['content'] = test_data['content'].astype(str).apply(normalize_text)
    train_data = remove_small_sentences(train_data)
    test_data = remove_small_sentences(test_data)
    logging.info("Preprocessing complete.")
    return train_data, test_data

def main():
    logging.info("Data transformation pipeline started.")
    
    train_file = "data/raw/train.csv"
    test_file = "data/raw/test.csv"

    train_data, test_data = load_data(train_file, test_file)
    train_data, test_data = preprocess_data(train_data, test_data)
    save_data(train_data, test_data)

    logging.info("Data transformation pipeline completed successfully.")

if __name__ == "__main__":
    main()
