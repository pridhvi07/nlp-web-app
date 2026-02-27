import streamlit as st
import pickle

model = pickle.load(open("model.pkl","rb"))
tfidf = pickle.load(open("tfidf.pkl","rb"))

st.title("Fake News Detection")

text = st.text_area("Enter News")

if st.button("Predict"):
    vec = tfidf.transform([text])
    pred = model.predict(vec)[0]

    if pred == 1:
        st.success("REAL NEWS")
    else:
        st.error("FAKE NEWS")