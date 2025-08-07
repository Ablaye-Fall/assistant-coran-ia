import streamlit as st
import requests
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
import joblib

# Charger les données
@st.cache_data
def load_tafsir_data():
    with open("sq-saadi.json", "r", encoding="utf-8") as f:
        tafsir_data = json.load(f)
    embeddings = np.load("tafsir_embeddings.npy")
    with open("tafsir_keys.json", "r", encoding="utf-8") as f:
        tafsir_keys = json.load(f)
    index = joblib.load("tafsir_index_sklearn.joblib")
    return tafsir_data, embeddings, tafsir_keys, index

tafsir_data, tafsir_embeddings, tafsir_keys, tafsir_index = load_tafsir_data()
model = SentenceTransformer('distiluse-base-multilingual-cased-v1')

# Obtenir la liste des sourates
@st.cache_data
def get_sourates():
    url = "https://api.quran.com/v4/chapters"
    response = requests.get(url)
    return response.json()["chapters"]

sourates = get_sourates()

# Obtenir les versets d'une sourate
@st.cache_data
def get_verses(sourah_id):
    url = f"https://api.quran.com/v4/quran/verses/uthmani?chapter_number={sourah_id}"
    response = requests.get(url)
    return response.json()["verses"]

# Obtenir la traduction d’un verset
@st.cache_data
def get_translation(sourah_id, verse_number, lang="fr"):
    url = f"https://api.quran.com/v4/quran/translations/33?chapter_number={sourah_id}"
    response = requests.get(url)
    translations = response.json()["translations"]
    for t in translations:
        if t["verse_key"] == f"{sourah_id}:{verse_number}":
            return t["text"]
    return "Traduction non disponible."

# Interface principale
st.title("📖 Assistant Coran IA")

# Choix de langue pour la traduction automatique
lang = st.selectbox("Choisissez la langue du tafsir :", ["fr", "en", "de", "es", "id"])

# Sélection sourate
sourah_name = st.selectbox("📚 Choisissez une sourate :", [s["name_arabic"] + f" ({s['name']})" for s in sourates])
sourah_id = sourates[[s["name_arabic"] + f" ({s['name']})" for s in sourates].index(sourah_name)]["id"]

verses = get_verses(sourah_id)
verse_numbers = [v["verse_number"] for v in verses]
verse_number = st.selectbox("📌 Choisissez le numéro du verset :", verse_numbers)

# Affichage verset en arabe et traduction
verse_ar = next((v["text_uthmani"] for v in verses if v["verse_number"] == verse_number), "")
verse_trad = get_translation(sourah_id, verse_number)

st.markdown(f"### 🕋 Verset en arabe\n{verse_ar}")
st.markdown(f"### 🌐 Traduction ({lang})\n{verse_trad}")

# Affichage tafsir traduit automatiquement
tafsir_key = f"{sourah_id}:{verse_number}"
tafsir_raw = tafsir_data.get(tafsir_key, "Tafsir non disponible.")
tafsir_translated = GoogleTranslator(source='auto', target=lang).translate(tafsir_raw)

st.markdown(f"### 📘 Tafsir traduit ({lang})")
st.info(tafsir_translated)

# Recherche sémantique
st.markdown("---")
st.markdown("### 🔍 Recherche sémantique dans le Tafsir")
query = st.text_input("Entrez une question ou un mot-clé…")

if query:
    query_embedding = model.encode([query])
    distances, indices = tafsir_index.kneighbors(query_embedding, n_neighbors=3)
    st.markdown("#### Résultats les plus pertinents :")
    for i, idx in enumerate(indices[0]):
        key = tafsir_keys[idx]
        tafsir_text = tafsir_data.get(key, "")
        translated = GoogleTranslator(source='auto', target=lang).translate(tafsir_text)
        st.markdown(f"**{i+1}. Verset {key}**")
        st.success(translated)
