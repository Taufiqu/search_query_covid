import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# === Download resource NLTK ===
nltk.download('punkt')
nltk.download('stopwords')

# === Inisialisasi stopwords ===
stop_words = set(stopwords.words('english'))

# === Fungsi Preprocessing ===
def preprocess(text):
    text = text.lower()
    text = re.sub(r'\W+', ' ', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

# === Load dan proses data (cache agar tidak diulang) ===
@st.cache_data
def load_and_prepare_data():
    df = pd.read_csv('covid_abstracts.csv')
    df['combined'] = df['title'].fillna('') + ' ' + df['abstract'].fillna('')
    df['processed'] = df['combined'].apply(preprocess)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df['processed'], show_progress_bar=False)
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    df['cluster'] = kmeans.fit_predict(embeddings)
    
    return df, embeddings, model

df, embeddings, model = load_and_prepare_data()

# === Tampilan UI dengan Streamlit ===
st.title("🔍 COVID-19 Semantic Search App")
query = st.text_input("Masukkan kata kunci pencarian:")

if query:
    with st.spinner("Mencari..."):
        processed_query = preprocess(query)
        query_embedding = model.encode([processed_query])
        similarity = cosine_similarity(query_embedding, embeddings).flatten()
        
        # Ambil top-N hasil dengan threshold
        top_n = 10
        threshold = 0.2
        top_indices = similarity.argsort()[::-1][:top_n]
        
        results = df.iloc[top_indices].copy()
        results['score'] = similarity[top_indices]
        results = results[results['score'] >= threshold]
        
        if results.empty:
            st.warning("❌ Tidak ada hasil relevan ditemukan.")
        else:
            for _, row in results.iterrows():
                with st.container():
                    st.subheader(row['title'])
                    st.markdown(f"**📘 Abstract:** {row['abstract']}")
                    st.markdown(f"**🔗 URL:** [Link]({row['url']})")
                    st.markdown(f"**📊 Relevance Score:** {row['score']:.4f}")
                    st.markdown(f"**🧩 Cluster:** {row['cluster']}")
                    st.markdown("---")
