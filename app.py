import streamlit as st
import pickle
import nltk
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

model = pickle.load(open("model.pkl","rb"))
tfidf = pickle.load(open("tfidf.pkl","rb"))

def preprocess(text):
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    tokens = [w for w in tokens if w not in string.punctuation]
    return " ".join(tokens)

st.title("Fake News Detection App")

text = st.text_area("Enter News Text")

if st.button("Predict"):
    clean = preprocess(text)
    vec = tfidf.transform([clean])
    pred = model.predict(vec)[0]

    if pred == 1:
        st.success("REAL NEWS")
    else:
        st.error("FAKE NEWS")
