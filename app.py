import streamlit as st
import requests
import json
import numpy as np
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import joblib

# 📁 Chargement des données
with open("sq-saadi.json", "r", encoding="utf-8") as f:
    tafsir_data = json.load(f)

tafsir_keys = json.load(open("tafsir_keys.json", "r", encoding="utf-8"))
embeddings = np.load("tafsir_embeddings.npy")
index = joblib.load("tafsir_index_sklearn.joblib")

# ⚙️ Initialisation du modèle d'embedding
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# 🌐 API URL
QURAN_API_URL = "https://api.quran.com:443/v4"

# 🌍 Choix de langue
langue = st.selectbox("🌐 Choisissez la langue de traduction :", ["fr", "en", "es", "id", "tr"], index=0)

# 📘 Récupérer toutes les sourates via API
@st.cache_data
def get_surahs():
    response = requests.get(f"{QURAN_API_URL}/chapters")
    return response.json()["chapters"]

surahs = get_surahs()
sourah_names = [f"{s['id']}. {s['name_arabic']} ({s['name_simple']})" for s in surahs]
selected_sourah = st.selectbox("📖 Choisissez une sourate :", sourah_names)
sourah_id = int(selected_sourah.split(".")[0])

# 📌 Récupérer nombre de versets
@st.cache_data
def get_verses_count(surah_id):
    response = requests.get(f"{QURAN_API_URL}/chapters/{surah_id}")
    return response.json()["chapter"]["verses_count"]

verse_count = get_verses_count(sourah_id)
verse_number = st.slider("📌 Choisissez le numéro du verset :", 1, verse_count)

# 📜 Récupérer verset et traduction
@st.cache_data
def get_verse_and_translation(surah_id, verse_id, lang_code):
    verse_res = requests.get(f"{QURAN_API_URL}/quran/verses/uthmani?chapter_number={surah_id}&verse_number={verse_id}")
    translation_res = requests.get(f"{QURAN_API_URL}/quran/translations/33?chapter_number={surah_id}&verse_number={verse_id}")  # 33 = français Hamidullah
    verse = verse_res.json()["verses"][0]["text_uthmani"]
    translation = translation_res.json()["translations"][0]["text"]
    return verse, translation

# 📖 Affichage verset
if st.button("Afficher le verset"):
    verse_ar, translation = get_verse_and_translation(sourah_id, verse_number, langue)
    st.markdown(f"### 🕋 Verset en arabe :")
    st.markdown(f"<div style='font-size: 26px; direction: rtl;'>{verse_ar}</div>", unsafe_allow_html=True)
    st.markdown("### 🌍 Traduction :")
    st.success(translation)

# 📚 Afficher le tafsir traduit automatiquement
tafsir_key = f"{sourah_id}:{verse_number}"
tafsir_text = tafsir_data.get(tafsir_key, "Tafsir non disponible.")
translated_tafsir = GoogleTranslator(source='auto', target=langue).translate(tafsir_text)
st.markdown("### 📖 Tafsir As-Saadi (traduit automatiquement) :")
st.info(translated_tafsir)

# 🔍 Recherche sémantique dans le tafsir
st.markdown("---")
st.markdown("### 🔍 Recherche sémantique dans le tafsir As-Saadi")
query = st.text_input("Entrez un mot ou une phrase pour rechercher dans le tafsir...")

if query:
    query_embedding = embedder.encode(query)
    similarities = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = similarities.argsort()[-5:][::-1]
    
    st.markdown("#### Résultats les plus pertinents :")
    for idx in top_indices:
        key = tafsir_keys[idx]
        tafsir = tafsir_data.get(key, "")
        translated = GoogleTranslator(source='auto', target=langue).translate(tafsir)
        st.markdown(f"✅ **Verset {key}** : {translated}")

