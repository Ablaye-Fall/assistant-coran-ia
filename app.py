import streamlit as st
import requests
import json
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# Charger les données du tafsir As-Saadi
with open("sq-saadi.json", "r", encoding="utf-8") as f:
    tafsir_data = json.load(f)

# Charger les données de recherche sémantique
embeddings = np.load("tafsir_embeddings.npy")
keys = json.load(open("tafsir_keys.json", encoding="utf-8"))
index = joblib.load("tafsir_index_sklearn.joblib")

model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

st.set_page_config(page_title="Assistant Coran IA", layout="centered")

st.title("📖 Assistant Coran - Tafsir & Traduction")
st.markdown("---")

# API config
API_URL = "https://api.alquran.cloud/v1"

# Charger les sourates
@st.cache_data
def get_sourates():
    response = requests.get(f"{API_URL}/surah")
    if response.status_code == 200:
        data = response.json()["data"]
        return [(s["number"], f'{s["englishName"]} ({s["name"]})') for s in data]
    else:
        st.error("Erreur lors du chargement des sourates.")
        return []

# Sélection utilisateur
sourates = get_sourates()
selected_surah = st.selectbox("📚 Choisissez une sourate :", sourates)
verse_number = st.number_input("📌 Choisir le numéro du verset :", min_value=1, value=1)

# Choix de la langue de traduction
langues = {
    "Français (Hamidullah)": "fr.hamidullah",
    "Anglais (Sahih)": "en.sahih",
    "Espagnol": "es.cortes",
}
selected_lang_label = st.selectbox("🌐 Choisir la langue de traduction :", list(langues.keys()))
translation_id = langues[selected_lang_label]

# Récupérer le verset
def get_verse(surah, verse, translation):
    res_ar = requests.get(f"{API_URL}/ayah/{surah}:{verse}")
    res_tr = requests.get(f"{API_URL}/ayah/{surah}:{verse}/{translation}")
    
    if res_ar.status_code == 200 and res_tr.status_code == 200:
        arabic = res_ar.json()["data"]["text"]
        translated = res_tr.json()["data"]["text"]
        return arabic, translated
    else:
        return None, None

arabic_text, translation = get_verse(selected_surah[0], verse_number, translation_id)

st.markdown("### 🕋 Verset en arabe")
st.markdown(f"**{arabic_text}**")

st.markdown(f"### 🌐 Traduction ({selected_lang_label})")
st.markdown(f"> {translation}")

# Affichage du Tafsir
st.markdown("### 📘 Tafsir As-Saadi")
tafsir_key = f"{selected_surah[0]}:{verse_number}"
tafsir_text = tafsir_data.get(tafsir_key, "⚠️ Tafsir non disponible pour ce verset.")
st.info(tafsir_text)

# Recherche sémantique
st.markdown("---")
st.subheader("🔍 Recherche sémantique dans le Tafsir")

user_query = st.text_input("Posez une question ou entrez un mot-clé (en français, arabe ou anglais)")

if user_query:
    query_embedding = model.encode([user_query])
    distances, indices = index.kneighbors(query_embedding, n_neighbors=3)

    st.markdown("#### Résultats les plus pertinents :")
    for i in indices[0]:
        key = keys[i]
        result_text = tafsir_data.get(key, "")
        st.markdown(f"**Verset {key}** : {result_text}")
