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
    lang = detect(user_question)
    try:
        if lang != "sq":
            question_sq = GoogleTranslator(source='auto', target='sq').translate(user_question)
        else:
            question_sq = user_question
    except:
        question_sq = user_question

    results = search_tafsir(question_sq, top_k=3)
    if not results:
        return "Aucune réponse trouvée.", lang

    combined = " ".join([r['tafsir'] for r in results if r['tafsir']])

    try:
        if lang != "sq":
            answer = GoogleTranslator(source='sq', target=lang).translate(combined)
        else:
            answer = combined
    except:
        answer = combined

    answer_refined = reformulate_text(answer, lang)
    return answer_refined, lang

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="Assistant Coran IA", page_icon="📖", layout="centered")
st.title("📖 Assistant Coran IA - Lecture, Tafsir & Q&A Multilingue")

# 1. Choix sourate
surahs = get_surahs()
surah_names = [f"{s['number']}. {s['englishName']} ({s['name']})" for s in surahs]
choix_surah = st.selectbox("📚 Choisissez une sourate :", surah_names)
num_surah = int(choix_surah.split(".")[0]) if choix_surah else 1

# 2. Choix traduction
translations = {
    "🇫🇷 Français (Hamidullah)": "fr.hamidullah",
    "🇬🇧 Anglais (Muhammad Asad)": "en.asad",
    "🇮🇩 Indonésien": "id.indonesian",
    "🇹🇷 Turc": "tr.translator",
    "🇺🇿 Ouzbek": "uz.sodik"
}
traduction_choisie = st.selectbox("🌐 Choisissez une traduction :", list(translations.keys()))
code_trad = translations.get(traduction_choisie, "en.asad")

# 3. Récupération versets
verses_data = get_verses(num_surah, code_trad)
verses_ar = verses_data[0]["ayahs"] if len(verses_data) > 0 else []
verses_trad = verses_data[1]["ayahs"] if len(verses_data) > 1 else []

# 4. Choix verset
max_verset = len(verses_ar) if verses_ar else 1
verset_num = st.number_input("📌 Choisissez un verset :", min_value=1, max_value=max_verset, value=1)

verset_ar = verses_ar[verset_num - 1]["text"] if verses_ar else ""
verset_trad = verses_trad[verset_num - 1]["text"] if verses_trad else ""

# 5. Affichage verset + traduction
zoom = st.slider("🔍 Zoom texte arabe", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
st.markdown(f"<p style='font-size:{zoom}em; direction:rtl; text-align:right; font-weight:bold;'>{verset_ar}</p>", unsafe_allow_html=True)
st.subheader(f"🌍 Traduction ({traduction_choisie})")
st.write(f"*{verset_trad}*")

# 6. Audio verset
reciters = {
    "🎙 Mishary Rashid Alafasy": "ar.alafasy",
    "🎙 Abdul Basit": "ar.abdulbasitmurattal",
    "🎙 Saad Al-Ghamdi": "ar.saoodshuraim"
}
reciter_choice = st.selectbox("🎧 Choisissez un récitateur :", list_
