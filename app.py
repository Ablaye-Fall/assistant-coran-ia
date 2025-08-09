import json
import numpy as np
import streamlit as st
import requests
import re
import tempfile
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from sklearn.neighbors import NearestNeighbors
from gtts import gTTS

DetectorFactory.seed = 0  # stabilité détection langue

# --- UTILITAIRES ---
def nettoyer_html(texte):
    texte = re.sub(r'<b><i>.*?</i></b>', '', texte, flags=re.DOTALL)
    return re.sub(r'<[^>]+>', '', texte)

def reformulate_text(text, target_lang):
    try:
        temp_en = GoogleTranslator(source='auto', target='en').translate(text)
        refined = GoogleTranslator(source='en', target=target_lang).translate(temp_en)
        return refined
    except Exception:
        return text

# --- CHARGEMENTS ---
@st.cache_resource(show_spinner=False)
def load_tafsir_resources():
    with open("sq-saadi.json", "r", encoding="utf-8") as f:
        tafsir_data = json.load(f)
    with open("tafsir_keys.json", "r", encoding="utf-8") as f:
        tafsir_keys = json.load(f)
    embeddings = np.load("tafsir_embeddings.npy")
    nn_index = NearestNeighbors(n_neighbors=5, metric='cosine')
    nn_index.fit(embeddings)
    return tafsir_data, tafsir_keys, embeddings, nn_index

@st.cache_resource(show_spinner=False)
def load_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

tafsir_data, tafsir_keys, tafsir_embeddings, tafsir_index = load_tafsir_resources()
model = load_model()

# --- API QURAN ---
@st.cache_data(ttl=86400)
def get_surahs():
    try:
        return requests.get("http://api.alquran.cloud/v1/surah").json()["data"]
    except:
        return []

@st.cache_data(ttl=86400)
def get_verses(surah_num, translation_code="en.asad"):
    try:
        url = f"http://api.alquran.cloud/v1/surah/{surah_num}/editions/quran-simple,{translation_code}"
        data = requests.get(url).json()["data"]
        return data
    except:
        return []

@st.cache_data(ttl=86400)
def get_audio_ayah(surah_num, ayah_num, reciter="ar.alafasy"):
    try:
        data = requests.get(f"http://api.alquran.cloud/v1/ayah/{surah_num}:{ayah_num}/{reciter}").json()
        return data["data"]["audio"]
    except:
        return None

# --- FONCTIONS PRINCIPALES ---
def search_tafsir(query_albanian, top_k=3):
    query_embed = model.encode([query_albanian], convert_to_tensor=False)
    distances, indices = tafsir_index.kneighbors(query_embed, n_neighbors=top_k)
    results = []
    for idx in indices[0]:
        key = tafsir_keys[idx]
        tafsir_text = tafsir_data.get(key, "")
        results.append({"key": key, "tafsir": tafsir_text})
    return results

def qa_multilang(user_question):
    # 1️⃣ Détection de la langue
    lang_detected = detect(user_question)

    # 2️⃣ Traduction en albanais si nécessaire
    if lang_detected != "sq":
        question_albanian = GoogleTranslator(s_
