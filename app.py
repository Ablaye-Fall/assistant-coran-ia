import streamlit as st
import json
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
import requests

# === Chargement des fichiers ===
@st.cache_resource
def load_data():
    with open("tafsir_fr_complet.json", "r", encoding="utf-8") as f:
        tafsir_data = json.load(f)
    with open("tafsir_keys.json", "r", encoding="utf-8") as f:
        tafsir_keys = json.load(f)
    embeddings = np.load("tafsir_embeddings.npy")
    index = joblib.load("tafsir_index_sklearn.joblib")
    return tafsir_data, tafsir_keys, embeddings, index

tafsir_data, tafsir_keys, embeddings, index = load_data()

# === Modèle d'embedding ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Fonctions ===
def search_tafsir(query, k=3):
    query_embedding = model.encode([query])
    scores, indices = index.kneighbors(query_embedding, n_neighbors=k)
    results = []
    for idx in indices[0]:
        key = tafsir_keys[idx]
        tafsir_text = tafsir_data.get(key, "Tafsir non disponible.")
        results.append((key, tafsir_text))
    return results

def get_quran_verse(surah_number, verse_number):
    url = f"https://api.quran.com/v4/quran/verses/uthmani?verse_key={surah_number}:{verse_number}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        verse_text = data["data"]["verses"][0]["text_uthmani"]
        return verse_text
    return "Verset non trouvé."

def get_translation(surah_number, verse_number):
    url = f"https://api.quran.com/v4/quran/translations/33?verse_key={surah_number}:{verse_number}"  # 33 = Hamidullah
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()["data"]["translations"][0]["text"]
    return "Traduction non trouvée."

def get_audio_url(surah_number, verse_number, reciter_id=7):  # Mishary Rashid Alafasy
    url = f"https://api.quran.com/v4/quran/recitations/{reciter_id}/by_ayah/{surah_number}:{verse_number}"
    response = requests.get(url)
    if response.status_code == 200:
        audio_url = response.json()["audio"]["url"]
        return audio_url
    return None

# === Interface Streamlit ===
st.title("📖 Assistant Coran IA")

# Sélection Sourate / Verset
surah_number = st.number_input("📌 Numéro de la sourate :", min_value=1, max_value=114, value=1)
verse_number = st.number_input("📌 Numéro du verset :", min_value=1, value=1)

# Chargement du verset
verse_text = get_quran_verse(surah_number, verse_number)
translation = get_translation(surah_number, verse_number)
tafsir_key = f"{surah_number}:{verse_number}"
tafsir_text = tafsir_data.get(tafsir_key, "Tafsir non disponible.")

# Affichage
st.markdown("### 🕋 Verset en arabe")
st.markdown(f"<div style='font-size:24px; direction: rtl;'>{verse_text}</div>", unsafe_allow_html=True)

st.markdown("### 🌍 Traduction (Hamidullah)")
st.write(translation)

st.markdown("### 📚 Tafsir As-Saadi (Français)")
st.write(tafsir_text)

# Audio
audio_url = get_audio_url(surah_number, verse_number)
if audio_url:
    st.markdown("### 🔊 Écoute du verset")
    st.audio(audio_url)

# Recherche sémantique
st.markdown("---")
st.markdown("## 🔍 Recherche sémantique dans le tafsir")
query = st.text_input("Entrez votre question ou mot-clé...")
if query:
    results = search_tafsir(query, k=3)
    st.markdown("### ✨ Résultats similaires :")
    for key, tafsir in results:
        st.markdown(f"**{key}** : {tafsir}")
