import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import spacy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK stopwords once
nltk.download('stopwords')

# Load SpaCy model (disable parser, ner for speed)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# Map numeric labels to sentiment strings
sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Load models and vectorizer/tokenizer
try:
    mnb_model = joblib.load('/content/drive/MyDrive/MNB_model.pkl')
    lr_model = joblib.load('/content/drive/MyDrive/LogisticRegression_model.pkl')
    rf_model = joblib.load('/content/drive/MyDrive/RandomForest_model.pkl')

    # Uncomment if you have XGBoost saved with joblib
    # xgboost_model = joblib.load('/content/drive/MyDrive/XGBoost_model.pkl')

    # Or if using xgb.Booster from JSON, import xgboost and load like this:
    # import xgboost as xgb
    # xgboost_model = xgb.Booster()
    # xgboost_model.load_model('/content/drive/MyDrive/XGBoost_model.json')

    lstm_model = load_model('/content/drive/MyDrive/LSTM_sentiment_model.keras')

    vectorizer = joblib.load('/content/drive/MyDrive/vectorizer.pkl')

    tokenizer = joblib.load('/content/drive/MyDrive/tokenizer.pkl')  # tokenizer for LSTM
    max_len = 100  # use your original max_len from training

except FileNotFoundError:
    st.error("Error loading one or more model/tokenizer/vectorizer files. Please verify the paths and files in your Google Drive.")
    st.stop()

# Preprocessing function
def preprocess_text(text):
    text = " ".join(text.lower() for text in text.split())
    text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
    stop = stopwords.words('english')
    text = " ".join(word for word in text.split() if word not in stop)
    doc = nlp(text)
    text = " ".join(token.lemma_ for token in doc)
    words_to_remove = ['chatgpt','app','chatgpts','chat','gpt','iphone','ipad','gpt4','phone','number','ai','use','io']
    text = " ".join(word for word in text.split() if word not in words_to_remove)
    return text

# Streamlit UI
st.title("Sentiment Analysis of App Reviews")
st.write("Enter an app review below to get sentiment predictions from multiple models.")

user_input_title = st.text_input("Review Title:")
user_input_review = st.text_area("Review:")

if st.button("Predict Sentiment"):

    if not user_input_review:
        st.warning("Please enter a review to predict the sentiment.")
    else:
        # Combine title and review
        complete_review_input = user_input_title + ' . ' + user_input_review if user_input_title else user_input_review

        # Preprocess input
        processed_input = preprocess_text(complete_review_input)

        # Vectorize for traditional ML models
        vectorized_input = vectorizer.transform([processed_input])

        # --- Predictions ---

        # MNB
        mnb_pred = mnb_model.predict(vectorized_input)[0]
        st.subheader("Multinomial Naive Bayes Prediction:")
        st.write(f"Sentiment: **{sentiment_map.get(mnb_pred, 'Unknown')}**")

        # Logistic Regression
        lr_pred = lr_model.predict(vectorized_input)[0]
        st.subheader("Logistic Regression Prediction:")
        st.write(f"Sentiment: **{sentiment_map.get(lr_pred, 'Unknown')}**")

        # Random Forest
        rf_pred = rf_model.predict(vectorized_input)[0]
        st.subheader("Random Forest Prediction:")
        st.write(f"Sentiment: **{sentiment_map.get(rf_pred, 'Unknown')}**")

        # XGBoost (optional)
        # Uncomment and adjust depending on your XGBoost model saving/loading method
        
        try:
            import xgboost as xgb
            dmatrix_input = xgb.DMatrix(vectorized_input)
            xgb_preds_prob = xgboost_model.predict(dmatrix_input)
            xgb_pred = xgb_preds_prob.argmax()
            st.subheader("XGBoost Prediction:")
            st.write(f"Sentiment: **{sentiment_map.get(xgb_pred, 'Unknown')}**")
        except Exception as e:
            st.warning(f"XGBoost prediction failed: {e}")
        

        # LSTM Prediction
        try:
            seq = tokenizer.texts_to_sequences([processed_input])
            pad_seq = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
            lstm_probs = lstm_model.predict(pad_seq)
            lstm_pred = lstm_probs.argmax(axis=1)[0]
            st.subheader("LSTM Prediction:")
            st.write(f"Sentiment: **{sentiment_map.get(lstm_pred, 'Unknown')}**")
        except Exception as e:
            st.warning(f"LSTM prediction failed: {e}")
