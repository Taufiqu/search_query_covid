import pandas as pd
import re
import nltk
import spacy
import streamlit as st
from nltk.corpus import stopwords
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Download resources
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")
stop_words = set(stopwords.words('english'))

@st.cache_data
def load_data():
    df = pd.read_csv('covid_abstracts.csv')
    df['combined'] = df['title'].fillna('') + " " + df['abstract'].fillna('')
    return df

@st.cache_data
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    doc = nlp(text)
    tokens = [token.text for token in doc if token.is_alpha and token.text not in stop_words and len(token.text) > 2]
    return ' '.join(tokens)

@st.cache_resource
def encode_sentences(texts):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model, model.encode(texts, show_progress_bar=True)

def main():
    st.title("📚 Semantic Search COVID-19 Abstracts")
    df = load_data()

    # Preprocess
    with st.spinner("🔄 Preprocessing data..."):
        df['processed'] = df['combined'].apply(preprocess)

    # Embedding
    model, embeddings = encode_sentences(df['processed'])

    # Clustering
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(embeddings)

    # Input Query
    query = st.text_input("Masukkan kata kunci pencarian:")
    threshold = st.slider("Threshold relevansi (0.0 - 1.0)", 0.0, 1.0, 0.2, 0.05)
    top_n = st.slider("Jumlah hasil teratas", 1, 20, 10)

    if query:
        with st.spinner("🔍 Mencari..."):
            processed_query = preprocess(query)
            query_embedding = model.encode([processed_query])
            similarity = cosine_similarity(query_embedding, embeddings).flatten()

            top_indices = similarity.argsort()[::-1]
            filtered_indices = [i for i in top_indices if similarity[i] >= threshold][:top_n]

            results = df.iloc[filtered_indices][['title', 'abstract', 'url', 'cluster']]
            results['score'] = similarity[filtered_indices]

            for _, row in results.iterrows():
                st.markdown(f"### 📄 {row['title']}")
                st.markdown(f"**Cluster:** {row['cluster']} | **Score:** {row['score']:.4f}")
                st.markdown(f"{row['abstract']}")
                st.markdown(f"[🔗 Link ke sumber]({row['url']})")
                st.markdown("---")

if __name__ == '__main__':
    main()
