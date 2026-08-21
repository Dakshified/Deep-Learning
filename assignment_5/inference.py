import os
import re
import string
import pickle
import sys

MODEL_PATH = "sentiment_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

def preprocess_text(text):
    """Clean the text: lowercase, remove HTML tags, remove punctuation."""
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags (e.g., <br />)
    text = re.sub(r'<[^>]*>', ' ', text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Replace punctuation with spaces to avoid merging words
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))
    text = text.translate(translator)
    
    # Normalize whitespaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def load_artifacts():
    """Loads the serialized model and vectorizer."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        print("[ERROR] Model files not found. Please run the training script 'assignment5.py' first.")
        sys.exit(1)
        
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
        
    with open(VECTORIZER_PATH, 'rb') as f:
        vectorizer = pickle.load(f)
        
    return model, vectorizer

def predict_sentiment(review, model, vectorizer):
    """Predicts the sentiment of a single review."""
    cleaned = preprocess_text(review)
    vec = vectorizer.transform([cleaned])
    
    pred_label = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    
    sentiment = "Positive" if pred_label == 1 else "Negative"
    confidence = probs[pred_label] * 100
    
    return sentiment, confidence

def main():
    # Load model and vectorizer
    model, vectorizer = load_artifacts()
    
    # If text is provided as argument
    if len(sys.argv) > 1:
        review = " ".join(sys.argv[1:])
        sentiment, confidence = predict_sentiment(review, model, vectorizer)
        print(f"\nReview: \"{review}\"")
        print(f"Predicted Sentiment: {sentiment} ({confidence:.2f}% confidence)\n")
        return
        
    # Interactive mode
    print("==================================================")
    print("  IMDb Movie Review Sentiment Inference Tool")
    print("  Type 'exit' or 'quit' to close.")
    print("==================================================")
    
    while True:
        try:
            review = input("\nEnter a movie review to analyze: ").strip()
            if not review:
                continue
            if review.lower() in ['exit', 'quit']:
                break
                
            sentiment, confidence = predict_sentiment(review, model, vectorizer)
            print(f"Result: {sentiment} ({confidence:.2f}% confidence)")
        except KeyboardInterrupt:
            break
            
    print("\nGoodbye!")

if __name__ == "__main__":
    main()
