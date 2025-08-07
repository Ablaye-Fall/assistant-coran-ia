import streamlit as st
import json
import requests
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator

st.set_page_config(page_title="Assistant Coran IA", layout="wide")
st.title("📖 Assistant Coran IA")

# -------------------- Chargement des données --------------------

@st.cache_data

def load_tafsir():
    with open("sq-saadi.json", "r", encoding="utf-8") as f:
        return json.load(f)

tafsir_data = load_tafsir()
embeddings = np.load("tafsir_embeddings.npy")
index = joblib.load("tafsir_index_sklearn.joblib")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# -------------------- Sélection sourate et verset --------------------

response = requests.get("https://api.quran.com/v4/chapters")
surahs = response.json()["data"]
surah_names = [f"{s['id']:>3} - {s['name_arabic']} ({s['name_simple']})" for s in surahs]
selected_surah = st.selectbox("📚 Choisissez une sourate :", surah_names)
surah_id = int(selected_surah.split(" - ")[0])

verse_num = st.number_input("📌 Choisissez le numéro du verset :", min_value=1, value=1, step=1)

# -------------------- Choix des langues --------------------

langue_verset = st.selectbox("🌐 Traduction du verset :", ["fr", "en"])
langue_tafsir = st.selectbox("🧠 Langue de traduction du tafsir :", ["fr", "en", "ar", "id"])

# -------------------- Affichage du verset + audio --------------------

verset_url = f"https://api.quran.com/v4/quran/verses/{langue_verset}?chapter_number={surah_id}&verse_number={verse_num}"
verse_data = requests.get(verset_url).json()
arabic = verse_data['data']['verses'][0]['text_uthmani']
traduction = verse_data['data']['verses'][0]['translations'][0]['text'] if 'translations' in verse_data['data']['verses'][0] else ""

st.markdown("### 🕋 Verset en arabe")
st.markdown(f"<div style='font-size:28px; direction:rtl;'>{arabic}</div>", unsafe_allow_html=True)

st.markdown("### 🌍 Traduction du verset")
st.markdown(f"{traduction}")

# Audio
recitation_url = f"https://verses.quran.com/{surah_id:03d}{verse_num:03d}.mp3"
st.audio(recitation_url)

# -------------------- Affichage du tafsir traduit --------------------

key = f"{surah_id}:{verse_num}"
tafsir_original = tafsir_data.get(key, "Aucun tafsir trouvé.")
tafsir_translated = GoogleTranslator(source="auto", target=langue_tafsir).translate(tafsir_original)

st.markdown("### 🧠 Tafsir traduit")
st.markdown(tafsir_translated)

# -------------------- Recherche sémantique --------------------

st.markdown("---")
st.markdown("## 🔍 Recherche sémantique dans le tafsir")
query = st.text_input("Entrez un mot ou une question (en français, anglais, etc.)")

if query:
    query_embedding = model.encode(query, convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]
    top_k = np.argsort(-scores)[:3]

    st.markdown("### Résultats les plus pertinents :")
    for idx in top_k:
        key = list(tafsir_data.keys())[idx]
        s_id, v_id = key.split(":")
        s_id, v_id = int(s_id), int(v_id)
        verse_ar = requests.get(f"https://api.quran.com/v4/quran/verses/ar?chapter_number={s_id}&verse_number={v_id}").json()['data']['verses'][0]['text_uthmani']
        tafsir_text = tafsir_data[key]
        tafsir_trans = GoogleTranslator(source="auto", target=langue_tafsir).translate(tafsir_text)

        st.markdown(f"**{s_id}:{v_id}**")
        st.markdown(f"<div style='direction:rtl; font-size:20px'>{verse_ar}</div>", unsafe_allow_html=True)
        st.markdown(f"{tafsir_trans}")
        st.markdown("---")
