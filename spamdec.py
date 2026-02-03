import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
import warnings

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')

# --- Caching the Model ---
# This function loads data and trains the model only once.
@st.cache_resource
def load_model():
    # Load the dataset
    df = pd.read_csv('combined_data.csv')

    # Define the preprocessing function
    def preprocess_text(text):
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        return " ".join(filtered_tokens)

    # Apply preprocessing
    df['text'] = df['text'].apply(preprocess_text)

    # Initialize and fit the TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df['text'])
    y = df['label']

    # Initialize and train the Multinomial Naive Bayes model
    model = MultinomialNB()
    model.fit(X, y)
    
    # Return the trained vectorizer and model
    return vectorizer, model

# Load the trained model and vectorizer
st.write("Loading model... please wait.")
vectorizer, model = load_model()
st.write("Model loaded successfully!")

# --- Streamlit App Interface ---
st.title("📧 Email Spam Detector")
st.write("Enter an email text below to check if it's spam or not.")

# Create a text area for user input
user_input = st.text_area("Email Text:", height=200)

# Create a button to trigger the prediction
if st.button("Check Email"):
    if user_input:
        # 1. Preprocess the user's input
        preprocessed_text = "".join(ch for ch in user_input if ch not in string.punctuation).lower()
        vectorized_text = vectorizer.transform([preprocessed_text])
        
        # 2. Make a prediction
        prediction = model.predict(vectorized_text)
        
        # 3. Display the result
        if prediction[0] == 1:
            st.error("This looks like SPAM! 🚨")
        else:
            st.success("This looks like a safe email. ✅")
    else:
        st.warning("Please enter some email text to check.")