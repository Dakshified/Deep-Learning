# Assignment 5: IMDb Sentiment Analysis

This project builds a Machine Learning model for binary sentiment classification on the IMDb dataset of 50,000 movie reviews.

## Dataset
The dataset consists of 50,000 samples and 2 features:
- `review`: Text of the movie review.
- `sentiment`: The binary label (`positive` or `negative`).

It is downloaded automatically from a stable GitHub mirror if not present locally.

## Setup & Dependencies
This project uses standard Python machine learning packages:
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `requests`

## Preprocessing & Vectorization
1. **Lowercasing**: All text is converted to lower case to eliminate casing variants.
2. **HTML Tag Removal**: Common HTML markup (e.g., `<br />`) is cleaned using Regular Expressions.
3. **Punctuation Clean-up**: Punctuation symbols are replaced with spaces to ensure contiguous words are not wrongly merged (e.g. `great.movie` -> `great movie`).
4. **Vectorization**: The reviews are converted to numerical features using `TfidfVectorizer` (TF-IDF) with:
   - Max Features: 25,000 vocabulary size.
   - N-grams: Unigrams and Bigrams (`ngram_range=(1, 2)`).
   - Stop words: English stop words are filtered out to focus on semantic content.

## Model
A **Logistic Regression** classifier is trained on the TF-IDF representation. 
- Fast training speed (less than 5 seconds).
- Excellent performance on sparse high-dimensional text data.
- High accuracy, typically **88% to 90%** (exceeding the target of 75%).

## Running the Training Pipeline
Run the main script to download the dataset, pre-process, train, evaluate, and serialize the model:
```bash
py assignment5.py
```
This script saves the following artifacts:
- `sentiment_model.pkl`: Serialized Logistic Regression model.
- `vectorizer.pkl`: Serialized TF-IDF Vectorizer.
- `confusion_matrix.png`: Plotted confusion matrix displaying test predictions.

## Running Inference
You can analyze the sentiment of any custom movie review by passing it as a command-line argument:
```bash
py inference.py "The storyline was beautiful, and the acting was top-tier. I loved every second of it!"
```
Or run the script with no arguments to use the interactive mode:
```bash
py inference.py
```
