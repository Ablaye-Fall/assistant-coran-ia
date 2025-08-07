import streamlit as st
import json
import numpy as np
import joblib
import requests
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator

# Chargement des données
tafsir_data = json.load(open("sq-saadi.json", encoding="utf-8"))
embeddings = np.load("tafsir_embeddings.npy")
tafsir_keys = json.load(open("tafsir_keys.json", encoding="utf-8"))
index = joblib.load("tafsir_index_sklearn.joblib")

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# API pour récupérer les versets et l'audio
QURAN_API_URL = "https://api.alquran.cloud/v1/ayah/"

def get_ayah(surah: int, ayah: int, translation: str = "fr.hamidullah"):
    res = requests.get(f"{QURAN_API_URL}{surah}:{ayah}/{translation}")
    if res.status_code == 200:
        data = res.json()
        return data["data"]
    return None

def get_tafsir(surah: int, ayah: int):
    key = f"{surah}:{ayah}"
    return tafsir_data.get(key, "Tafsir non disponible.")

def semantic_search(query: str, top_k=3):
    query_embedding = model.encode([query])[0]
    D, I = index.kneighbors([query_embedding], n_neighbors=top_k)
    results = []
    for idx in I[0]:
        key = tafsir_keys[idx]
        tafsir_text = tafsir_data.get(key, "Tafsir introuvable.")
        results.append((key, tafsir_text))
    return results

def play_audio(audio_url):
    st.audio(audio_url, format='audio/mp3')

# Interface Streamlit
st.title("🕌 Assistant Coran IA")

# Choix de la sourate et du verset
surah_number = st.number_input("📖 Numéro de la sourate", min_value=1, max_value=114, value=1)
verse_number = st.number_input("📌 Numéro du verset", min_value=1, value=1)

# Choix de la langue de traduction
langue = st.selectbox("🌐 Choisir la langue de traduction", ["fr.hamidullah", "en.sahih", "ar.alafasy"])

if st.button("Afficher le verset et le tafsir"):
    ayah_data = get_ayah(surah=surah_number, ayah=verse_number, translation=langue)

    if ayah_data:
        st.markdown(f"### 🕋 Verset en arabe\n{ayah_data['text']}")
        st.markdown(f"### 🌍 Traduction\n{ayah_data['edition']['name']}:\n> {ayah_data['text']}")

        tafsir_text = get_tafsir(surah=surah_number, ayah=verse_number)
        st.markdown(f"### 📚 Tafsir (Saadi)\n{tafsir_text}")

        if 'audio' in ayah_data:
            st.markdown("### 🔊 Audio du verset")
            play_audio(ayah_data['audio'])
    else:
        st.error("❌ Verset non trouvé via l'API Quran.")

st.markdown("---")
st.markdown("### 🔍 Recherche sémantique dans le tafsir")
query = st.text_input("Entrez une question ou un mot-clé")
if st.button("Rechercher") and query:
    results = semantic_search(query)
    for key, tafsir in results:
        st.markdown(f"**{key}**: {tafsir}")
