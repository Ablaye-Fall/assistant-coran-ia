import streamlit as st
import json
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# ---------- Chargement des fichiers ----------
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_data
def load_tafsir_data():
    with open("tafsir_fr_complet.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_keys():
    with open("tafsir_keys.json", "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_embeddings():
    return np.load("tafsir_embeddings.npy")

@st.cache_resource
def load_index():
    return joblib.load("tafsir_index_sklearn.joblib")

# Chargement
model = load_model()
tafsir_data = load_tafsir_data()
tafsir_keys = load_keys()
embeddings = load_embeddings()
index_model = load_index()

# ---------- Interface Streamlit ----------
st.title("📖 Assistant Coran - Tafsir & Recherche Sémantique")

tabs = st.tabs(["🔍 Verset & Tafsir", "🧠 Recherche Sémantique"])

# ---------- Onglet 1 : Choix sourate/verset ----------
with tabs[0]:
    st.subheader("📌 Choisissez un verset")

    surah_number = st.number_input("Numéro de la sourate", min_value=1, max_value=114, step=1)
    verse_number = st.number_input("Numéro du verset", min_value=1, step=1)

    tafsir_key = f"{surah_number}:{verse_number}"
    local_tafsir = tafsir_data.get(tafsir_key, "❌ Tafsir non trouvé pour ce verset.")

    st.markdown("**📝 Tafsir (Français)**")
    st.write(local_tafsir)

    if st.toggle("📘 Traduire en anglais"):
        translated = GoogleTranslator(source='auto', target='en').translate(local_tafsir)
        st.markdown("**🌍 Tafsir (English)**")
        st.write(translated)

# ---------- Onglet 2 : Recherche sémantique ----------
with tabs[1]:
    st.subheader("🔎 Recherche par mot-clé (Tafsir)")

    user_query = st.text_input("Entrez votre requête (ex: miséricorde, enfer, foi...)")
    
    if user_query:
        query_embedding = model.encode([user_query])
        distances, indices = index_model.kneighbors(query_embedding, n_neighbors=5)

        st.markdown("### 📚 Résultats les plus pertinents")
        for rank, idx in enumerate(indices[0], 1):
            key = tafsir_keys[idx]
            tafsir = tafsir_data.get(key, "Tafsir manquant")
            st.markdown(f"**{rank}. Verset {key}**")
            st.write(tafsir)
            st.markdown("---")
