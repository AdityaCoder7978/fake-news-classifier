import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text preprocessing function
def preprocess(text):
    lemmatizer = WordNetLemmatizer()
    text = re.sub('[^a-zA-Z]', ' ', text)  # remove non-letter characters
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords.words('english')]
    return " ".join(text)

# Inject custom CSS for better UI
st.markdown("""
    <style>
    body {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .stApp {
        background-color: #1E1E1E;
    }
    .stTextArea textarea {
        background-color: #2B2B2B !important;
        color: #FFFFFF !important;
        font-size: 16px !important;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 5px;
    }
    .report-container {
        padding: 16px;
        border-radius: 10px;
        margin-top: 20px;
        font-size: 18px;
    }
    .real {
        background-color: #d4edda;
        color: #155724;
    }
    .fake {
        background-color: #f8d7da;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

# App title and instructions
st.markdown("### 📰 Fake News Detection")
st.write("Enter a news article below and check whether it's **Real** or **Fake**.")

# Text input area
text_input = st.text_area("Paste News Article Here", height=200)

# Prediction on button click
if st.button("Classify"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some news text first.")
    else:
        processed_text = preprocess(text_input)
        vector_input = vectorizer.transform([processed_text])
        prediction = model.predict(vector_input)[0]

        if prediction == 1:
            st.markdown('<div class="report-container real">✅ This news article is likely <strong>Real</strong>.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="report-container fake">❌ This news article is likely <strong>Fake</strong>.</div>', unsafe_allow_html=True)
