import streamlit as st
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')

st.set_page_config(page_title="Text Summarizer", layout="centered")

st.title("🧠 AI Text Summarizer")
st.write("Summarize long text instantly (Fast & Lightweight)")

text = st.text_area("Enter your text here", height=250)

def summarize_text(text, num_sentences=3):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(sentences)
    similarity_matrix = cosine_similarity(tfidf)

    sentence_scores = similarity_matrix.sum(axis=1)
    ranked_sentences = sorted(
        ((score, sentence) for score, sentence in zip(sentence_scores, sentences)),
        reverse=True
    )

    summary = " ".join([sentence for _, sentence in ranked_sentences[:num_sentences]])
    return summary

if st.button("Summarize"):
    if len(text.strip()) < 50:
        st.warning("Please enter at least 50 characters.")
    else:
        summary = summarize_text(text)
        st.subheader("Summary")
        st.success(summary)
