import streamlit as st
import requests
import json
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator

# Chargement fichiers
with open("tafsir_keys.json", "r", encoding="utf-8") as f:
    tafsir_keys = json.load(f)
with open("sq-saadi.json", "r", encoding="utf-8") as f:
    tafsir_data = json.load(f)

embeddings = np.load("tafsir_embeddings.npy")
index = joblib.load("tafsir_index_sklearn.joblib")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Langues de traduction disponibles via l'API Quran
translation_options = {
    "Français (Hamidullah)": "fr.hamidullah",
    "Anglais (Sahih)": "en.sahih",
    "Arabe (original)": "ar",
    "Indonésien": "id.indonesian"
}

# Fonction pour récupérer les données du verset
def get_verse_text(surah, verse, lang_code):
    try:
        url = f"https://api.quran.com/v4/quran/verses/{lang_code}?verse_key={surah}:{verse}"
        response = requests.get(url)
        data = response.json()
        return data["verses"][0]["text"]
    except:
        return "❌ Erreur de récupération du verset"

# Fonction pour l’audio
def get_audio_url(surah, verse):
    try:
        audio_url = f"https://verses.quran.com/AbdulBaset/Mujawwad/mp3/{int(surah):03d}{int(verse):03d}.mp3"
        return audio_url
    except:
        return None

# Interface Streamlit
st.title("📖 Assistant Coran IA")

st.sidebar.header("Options")

# Sélection de la sourate et verset
surah_number = st.sidebar.number_input("Numéro de sourate", min_value=1, max_value=114, value=1)
verse_number = st.sidebar.number_input("Numéro de verset", min_value=1, value=1)

# Choix de la langue
translation_name = st.sidebar.selectbox("Langue de traduction", list(translation_options.keys()))
translation_code = translation_options[translation_name]

st.markdown(f"## 🕋 Verset {surah_number}:{verse_number}")

# Texte en arabe
ar_text = get_verse_text(surah_number, verse_number, "ar")
st.markdown(f"**🗣️ Arabe :**\n\n> {ar_text}")

# Traduction choisie
translated_text = get_verse_text(surah_number, verse_number, translation_code)
st.markdown(f"**🌐 Traduction ({translation_name}) :**\n\n> {translated_text}")

# Audio
audio_url = get_audio_url(surah_number, verse_number)
if audio_url:
    st.audio(audio_url, format="audio/mp3")

# Tafsir local
tafsir_key = f"{surah_number}:{verse_number}"
tafsir_text = tafsir_data.get(tafsir_key, "❌ Tafsir non disponible pour ce verset.")
st.markdown("## 🧠 Tafsir As-Saadi")
st.write(tafsir_text)

# Recherche sémantique
st.markdown("## 🔍 Recherche sémantique dans le tafsir")
query = st.text_input("Entrez une question ou un mot-clé", "")

if query:
    q_emb = model.encode([query])
    distances, indices = index.kneighbors(q_emb, n_neighbors=3)
    st.markdown("### Résultats similaires :")
    for idx in indices[0]:
        ref = tafsir_keys[idx]
        extrait = tafsir_data.get(ref, "[Non trouvé]")
        st.markdown(f"**{ref}** : {extrait[:300]}...")
