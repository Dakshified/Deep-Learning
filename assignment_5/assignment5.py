import os
import re
import string
import pickle
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

# Configurations
DATASET_URL = "https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv"
DATASET_PATH = "IMDB-Dataset.csv"
MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"
CONF_MATRIX_IMAGE = "confusion_matrix.png"

def download_dataset():
    """Downloads the IMDb dataset if it does not exist locally."""
    if os.path.exists(DATASET_PATH):
        print(f"[INFO] Dataset already exists at {DATASET_PATH}")
        return

    print(f"[INFO] Downloading dataset from {DATASET_URL}...")
    print("[INFO] This might take a few moments (approx. 66 MB)...")
    
    response = requests.get(DATASET_URL, stream=True)
    response.raise_for_status()
    
    with open(DATASET_PATH, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    
    print("[INFO] Download complete!")

def preprocess_text(text):
    """Clean the text: lowercase, remove HTML tags, remove punctuation."""
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags (e.g., <br />)
    text = re.sub(r'<[^>]*>', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace punctuation with spaces to avoid merging words (e.g. "end.after" -> "end after")
    # string.punctuation contains: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)
    
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
    # 1. Download Dataset
    download_dataset()
    
    # 2. Load Dataset
    print("[INFO] Loading IMDb dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    print(f"[INFO] Dataset shape: {df.shape}")
    print("[INFO] Dataset preview:")
    print(df.head(3))
    
    # Verify features and samples
    assert df.shape == (50000, 2), f"Expected shape (50000, 2), got {df.shape}"
    assert "review" in df.columns and "sentiment" in df.columns, "Columns must be 'review' and 'sentiment'"
    
    # 3. Preprocess Text
    print("[INFO] Preprocessing reviews (removing HTML tags, punctuation, lowercasing)...")
    # Print a preview before and after preprocessing
    sample_before = df['review'].iloc[0]
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    sample_after = df['cleaned_review'].iloc[0]
    
    print("\n--- Preprocessing Preview ---")
    print(f"Original:\n{sample_before[:150]}...")
    print(f"\nCleaned:\n{sample_after[:150]}...")
    print("-----------------------------\n")
    
    # Map sentiments to numbers (positive: 1, negative: 0)
    df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # 4. Train-Test Split
    X = df['cleaned_review']
    y = df['label']
    
    print("[INFO] Splitting dataset (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"[INFO] Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 5. Vectorization (TF-IDF)
    print("[INFO] Vectorizing reviews using TF-IDF (unigrams + bigrams, max features 25k)...")
    vectorizer = TfidfVectorizer(
        max_features=25000, 
        ngram_range=(1, 2), 
        stop_words='english'
    )
    
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"[INFO] Vocabulary shape: {X_train_vec.shape}")
    
    # 6. Model Training (Logistic Regression)
    print("[INFO] Training Logistic Regression classifier...")
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_vec, y_train)
    
    # 7. Model Evaluation
    print("[INFO] Evaluating model performance...")
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n=====================================")
    print(f"TEST ACCURACY: {accuracy * 100:.2f}%")
    print(f"=====================================\n")
    
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))
    
    # Check if target accuracy is met
    if accuracy >= 0.75:
        print("[SUCCESS] Accuracy is above the 75% target!")
    else:
        print("[WARNING] Accuracy is below the 75% target.")
        
    # Generate and save confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['negative', 'positive'])
    
    plt.figure(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title(f"IMDb Sentiment Analysis Confusion Matrix (Acc: {accuracy*100:.2f}%)")
    plt.savefig(CONF_MATRIX_IMAGE, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"[INFO] Confusion matrix saved as {CONF_MATRIX_IMAGE}")
    
    # 8. Serialization (Save Model & Vectorizer)
    print(f"[INFO] Saving model to {MODEL_PATH}...")
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
        
    print(f"[INFO] Saving vectorizer to {VECTORIZER_PATH}...")
    with open(VECTORIZER_PATH, 'wb') as f:
        pickle.dump(vectorizer, f)
        
    print("[INFO] Pipeline executed successfully! All artifacts saved.")

if __name__ == "__main__":
    main()
