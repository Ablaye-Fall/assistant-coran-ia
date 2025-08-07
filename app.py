import streamlit as st
import requests
import json
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# --- Configuration de la page
st.set_page_config(page_title="Assistant Coran IA", layout="centered")
st.title("📖 Assistant Coran - Tafsir & Traduction multilingue")
st.markdown("---")

# --- Charger les fichiers locaux
with open("sq-saadi.json", "r", encoding="utf-8") as f:
    tafsir_data = json.load(f)

embeddings = np.load("tafsir_embeddings.npy")
keys = json.load(open("tafsir_keys.json", encoding="utf-8"))
index = joblib.load("tafsir_index_sklearn.joblib")
model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")

# --- API config
API_URL = "https://api.alquran.cloud/v1"

# --- Récupérer les sourates depuis l'API
@st.cache_data
def get_sourates():
    response = requests.get(f"{API_URL}/surah")
    if response.status_code == 200:
        data = response.json()["data"]
        return [(s["number"], f'{s["englishName"]} ({s["name"]})') for s in data]
    else:
        st.error("Erreur lors du chargement des sourates.")
        return []

# --- Sélection utilisateur
sourates = get_sourates()
selected_surah = st.selectbox("📚 Choisissez une sourate :", sourates)
verse_number = st.number_input("📌 Choisir le numéro du verset :", min_value=1, value=1)

# --- Choix langue
langues = {
    "Français": "fr",
    "Anglais": "en",
    "Espagnol": "es",
    "Allemand": "de"
}
selected_lang = st.selectbox("🌐 Choisissez la langue de traduction :", list(langues.keys()))
lang_code = langues[selected_lang]
translation_api_id = {
    "fr": "fr.hamidullah",
    "en": "en.sahih",
    "es": "es.cortes",
    "de": "de.bubenheim"
}.get(lang_code, "fr.hamidullah")

# --- Obtenir verset et traduction depuis API Quran
def get_verse_and_translation(surah, verse, translation_id):
    res_ar = requests.get(f"{API_URL}/ayah/{surah}:{verse}")
    res_tr = requests.get(f"{API_URL}/ayah/{surah}:{verse}/{translation_id}")
    
    if res_ar.status_code == 200 and res_tr.status_code == 200:
        arabic = res_ar.json()["data"]["text"]
        translated = res_tr.json()["data"]["text"]
        audio_url = res_ar.json()["data"]["audio"]
        return arabic, translated, audio_url
    else:
        return None, None, None

arabic_text, translated_text, audio_url = get_verse_and_translation(selected_surah[0], verse_number, translation_api_id)

# --- Affichage du verset
if arabic_text:
    st.markdown("### 🕋 Verset en arabe")
    st.markdown(f"<div style='font-size:28px; direction:rtl;'>{arabic_text}</div>", unsafe_allow_html=True)
    
    st.markdown(f"### 🌐 Traduction du verset ({selected_lang})")
    st.markdown(f"> {translated_text}")
    
    # --- Lecture audio
    st.audio(audio_url)

# --- Tafsir traduit automatiquement
tafsir_key = f"{selected_surah[0]}:{verse_number}"
original_tafsir = tafsir_data.get(tafsir_key, None)

if original_tafsir:
    try:
        translated_tafsir = GoogleTranslator(source='auto', target=lang_code).translate(original_tafsir[:4500])  # limite de caractères
        st.markdown("### 📘 Tafsir As-Saadi (Traduit automatiquement)")
        st.success(translated_tafsir)
    except Exception as e:
        st.warning("Erreur lors de la traduction automatique du tafsir.")
else:
    st.info("⚠️ Tafsir non disponible pour ce verset.")

# --- Recherche sémantique dans le tafsir
st.markdown("---")
st.subheader("🔍 Recherche sémantique dans le Tafsir")

user_query = st.text_input("Posez une question ou entrez un mot-clé (en arabe, français ou anglais)")

if user_query:
    try:
        query_embedding = model.encode([user_query])
        distances, indices = index.kneighbors(query_embedding, n_neighbors=3)

        st.markdown("#### Résultats les plus pertinents (traduits automatiquement):")
        for i in indices[0]:
            key = keys[i]
            original_text = tafsir_data.get(key, "")
            if original_text:
                translated_text = GoogleTranslator(source='auto', target=lang_code).translate(original_text[:4500])
                st.markdown(f"**Verset {key}** : {translated_text}")
    except Exception as e:
        st.error("Erreur dans la recherche sémantique ou la traduction.")

