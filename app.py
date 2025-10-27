# app.py (version corrigée)
import json
import io
import re
import logging
import numpy as np
import streamlit as st
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from sklearn.neighbors import NearestNeighbors
from gtts import gTTS

# ---------------------------
# --- CONFIG / LOGGING ---
# ---------------------------
DetectorFactory.seed = 0  # stabilité détection langue
logging.basicConfig(level=logging.INFO)

# ---------------------------
# --- UTILITAIRES ---
# ---------------------------
DIACRITICS_RE = re.compile(r'[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]')

def remove_diacritics(text: str) -> str:
    """Supprime les harakât/diacritiques arabes."""
    return DIACRITICS_RE.sub('', text) if text else text

def nettoyer_html(texte):
    """Supprime les balises HTML du texte tafsir."""
    if not texte:
        return ""
    texte = re.sub(r'<b><i>.*?</i></b>', '', texte, flags=re.DOTALL)
    return re.sub(r'<[^>]+>', '', texte)

def safe_translate(text, src='auto', target='en'):
    """Traduction robuste (catch des erreurs)."""
    if not text:
        return ""
    try:
        return GoogleTranslator(source=src, target=target).translate(text)
    except Exception as e:
        logging.warning(f"[Traduction] {src}->{target} échouée : {e}")
        return text

def reformulate_text(text, target_lang):
    """Double traduction pour fluidité."""
    if not text:
        return ""
    try:
        temp_en = safe_translate(text, src='auto', target='en')
        refined = safe_translate(temp_en, src='en', target=target_lang)
        return refined
    except Exception as e:
        logging.info(f"[Reformulation] échouée : {e}")
        return text

def detect_language_safe(text):
    """Détecte la langue, même si caractères arabes."""
    if not text or not text.strip():
        return "unknown"
    if re.search(r'[\u0600-\u06FF]', text):
        return "ar"
    try:
        return detect(text)
    except Exception:
        return "unknown"

def _limit_text(text, max_chars=1200, max_sentences=6):
    """Nettoie et limite le texte."""
    if not text:
        return ""
    t = re.sub(r"<[^>]+>", "", text)
    sents = re.split(r'(?<=[.!?])\s+', t)
    limited = " ".join(sents[:max_sentences]).strip()
    if len(limited) > max_chars:
        limited = limited[:max_chars].rsplit(" ", 1)[0] + "..."
    return limited

# ---------------------------
# SIDEBAR (ARABE)
# ---------------------------
with st.sidebar:
    st.header("🕌 Affichage arabe")
    if "hide_harakat" not in st.session_state:
        st.session_state.hide_harakat = False
    if "zoom_em" not in st.session_state:
        st.session_state.zoom_em = 1.5

    st.session_state.hide_harakat = st.checkbox("Masquer les harakât (diacritiques)",
                                                value=st.session_state.hide_harakat)
    st.session_state.zoom_em = st.slider("🔍 Zoom texte arabe (em)",
                                         1.0, 3.0, st.session_state.zoom_em, 0.1)

def render_verset(verset_ar, weight=600):
    """Affiche le verset arabe avec options de style."""
    display_text = remove_diacritics(verset_ar) if st.session_state.hide_harakat else verset_ar
    if not display_text:
        return
    st.markdown(
        f"<p style='font-size:{st.session_state.zoom_em}em; direction:rtl; text-align:right; font-weight:{weight};'>{display_text}</p>",
        unsafe_allow_html=True
    )

# ---------------------------
# CHARGEMENT DES RESSOURCES
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_tafsir_resources():
    try:
        with open("sq-saadi.json", "r", encoding="utf-8") as f:
            tafsir_data = json.load(f)
    except Exception as e:
        logging.error(f"Erreur chargement sq-saadi.json : {e}")
        tafsir_data = {}

    try:
        with open("tafsir_keys.json", "r", encoding="utf-8") as f:
            tafsir_keys = json.load(f)
    except Exception:
        tafsir_keys = []

    try:
        embeddings = np.load("tafsir_embeddings.npy")
    except Exception:
        embeddings = np.zeros((0, 384))

    nn_index = None
    if embeddings.size > 0:
        try:
            n_neighbors = min(10, embeddings.shape[0])
            nn_index = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
            nn_index.fit(embeddings)
        except Exception as e:
            logging.error(f"Erreur index embeddings : {e}")

    return tafsir_data, tafsir_keys, embeddings, nn_index

@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2", show_progress_bar=False)

@st.cache_resource
def load_reranker_model():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

tafsir_data, tafsir_keys, tafsir_embeddings, tafsir_index = load_tafsir_resources()
model = load_sentence_model()
reranker = load_reranker_model()

# ---------------------------
# API ALQURAN.CLOUD
# ---------------------------
@st.cache_data(ttl=86400)
def get_surahs():
    try:
        r = requests.get("https://api.alquran.cloud/v1/surah", timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        logging.error(f"get_surahs error: {e}")
        return []

@st.cache_data(ttl=86400)
def get_verses(surah_num, translation_code="en.asad"):
    try:
        url = f"https://api.alquran.cloud/v1/surah/{surah_num}/editions/quran-simple,{translation_code}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])
        return data if isinstance(data, list) else [data]
    except Exception as e:
        logging.error(f"get_verses error: {e}")
        return []

@st.cache_data(ttl=86400)
def get_audio_ayah(surah_num, ayah_num, reciter="ar.alafasy"):
    try:
        r = requests.get(f"https://api.alquran.cloud/v1/ayah/{surah_num}:{ayah_num}/{reciter}", timeout=10)
        r.raise_for_status()
        d = r.json().get("data", {})
        return d.get("audio") or d.get("Audio")
    except Exception as e:
        logging.warning(f"get_audio_ayah error: {e}")
        return None

# ---------------------------
# RECHERCHE / RERANKING
# ---------------------------
def search_tafsir(query_translated, top_k=10):
    if tafsir_index is None or not tafsir_keys:
        return []
    try:
        query_embed = model.encode([query_translated], convert_to_tensor=False, show_progress_bar=False)
        n_neighbors = min(top_k, tafsir_embeddings.shape[0])
        distances, indices = tafsir_index.kneighbors(query_embed, n_neighbors=n_neighbors)
    except Exception as e:
        logging.error(f"Erreur recherche : {e}")
        return []

    results = []
    for idx in indices[0]:
        if idx >= len(tafsir_keys):
            continue
        key = tafsir_keys[idx]
        tafsir_value = tafsir_data.get(key, "")
        tafsir_text = tafsir_value.get("text") if isinstance(tafsir_value, dict) else str(tafsir_value)
        if tafsir_text.strip():
            results.append({"key": key, "tafsir": tafsir_text.strip()})
    return results

def rerank_results(question, passages):
    if not passages:
        return []
    try:
        pairs = [(question, p) for p in passages]
        scores = reranker.predict(pairs)
        norm_scores = (scores - np.min(scores)) / (np.ptp(scores) + 1e-9)
        return sorted(zip(passages, norm_scores.tolist()), key=lambda x: x[1], reverse=True)
    except Exception as e:
        logging.error(f"Reranking error: {e}")
        return [(p, 0.0) for p in passages]

# ---------------------------
# PIPELINE Q&A MULTILINGUE
# ---------------------------
def qa_multilang(user_question):
    lang_detected = detect_language_safe(user_question)
    if lang_detected == "unknown":
        return "Impossible de détecter la langue.", lang_detected

    # Traduction vers albanais
    question_sq = safe_translate(user_question, src=lang_detected, target="sq") if lang_detected != "sq" else user_question

    results = search_tafsir(question_sq, top_k=10)
    passages = [r["tafsir"] for r in results if r.get("tafsir")]
    if not passages:
        return "Aucune réponse trouvée dans le tafsir albanais.", lang_detected

    ranked = rerank_results(question_sq, passages)
    combined = " ".join([p for p, _ in ranked[:3]])
    combined_clean = _limit_text(combined)

    # Retraduction vers langue d’origine
    answer_translated = safe_translate(combined_clean, src="sq", target=lang_detected) if lang_detected != "sq" else combined_clean
    return reformulate_text(answer_translated, lang_detected), lang_detected

# ---------------------------
# --- INTERFACE STREAMLIT ---
# ---------------------------
st.set_page_config(
    page_title="Assistant Coran IA",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📖 Assistant Coran IA - Lecture, Tafsir & Q&A Multilingue")

# 1. Choix sourate
surahs = get_surahs()
surah_names = [f"{s['number']}. {s.get('englishName','')} ({s.get('name','')})" for s in surahs] if surahs else []
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
arabic_edition = None
trans_edition = None
if isinstance(verses_data, list) and len(verses_data) > 0:
    arabic_edition = next((d for d in verses_data if d.get("edition", {}).get("identifier") in ["quran-simple", "quran-simple-qdc", "quran-simple-text"]), None)
    trans_edition = next((d for d in verses_data if d.get("edition", {}).get("identifier") == code_trad), None)
    if not arabic_edition and len(verses_data) >= 1:
        arabic_edition = verses_data[0]
    if not trans_edition and len(verses_data) >= 2:
        trans_edition = verses_data[1]

verses_ar = arabic_edition.get("ayahs", []) if arabic_edition else []
verses_trad = trans_edition.get("ayahs", []) if trans_edition else []

# 4. Choix verset
max_verset = len(verses_ar) if verses_ar else 1
verset_num = st.number_input("📌 Choisissez un verset :", min_value=1, max_value=max_verset, value=1)

verset_ar = verses_ar[verset_num - 1]["text"] if verses_ar and len(verses_ar) >= verset_num else ""
verset_trad = verses_trad[verset_num - 1]["text"] if verses_trad and len(verses_trad) >= verset_num else ""

render_verset(verset_ar, weight=700)

st.subheader(f"🌍 Traduction ({traduction_choisie})")
st.write(f"*{verset_trad}*")

# Audio verset
reciters = {
    "🎙 Mishary Rashid Alafasy": "ar.alafasy",
    "🎙 Abdul Basit": "ar.abdulbasitmurattal",
    "🎙 Saad Al-Ghamdi": "ar.saoodshuraim"
}
reciter_choice = st.selectbox("🎧 Choisissez un récitateur :", list(reciters.keys()))
audio_url = get_audio_ayah(num_surah, verset_num, reciters.get(reciter_choice, "ar.alafasy"))
if audio_url:
    try:
        st.audio(audio_url, format="audio/mp3")
    except Exception as e:
        logging.warning(f"Audio widget error: {e}")

# Tafsir affichage
cle_verset = f"{num_surah}:{verset_num}"
tafsir_entry = tafsir_data.get(cle_verset, {})
tafsir_raw = tafsir_entry.get("text", "") if isinstance(tafsir_entry, dict) else str(tafsir_entry or "")
tafsir_clean = nettoyer_html(tafsir_raw)

lang_tafsir = st.selectbox("🌐 Langue traduction du tafsir :", ["fr", "en", "ar", "es", "wo"])
traduction_tafsir = safe_translate(tafsir_clean, src='auto', target=lang_tafsir) if tafsir_clean else ""

st.subheader("📜 Tafsir")
st.write(traduction_tafsir)

# Audio tafsir gTTS
tts_langs = ["fr", "en", "ar", "es"]
if lang_tafsir in tts_langs and traduction_tafsir:
    try:
        tts = gTTS(traduction_tafsir, lang=lang_tafsir)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        st.audio(audio_buffer.read(), format="audio/mp3")
    except Exception as e:
        st.warning(f"Audio tafsir non disponible : {e}")

# Q&A multilingue
st.markdown("---")
st.subheader("❓ Posez une question au sujet du Coran (toutes langues)")

if "history" not in st.session_state:
    st.session_state.history = []

for chat in st.session_state.history:
    st.markdown(f"🧑‍💻 **Vous :** {chat['question']}")
    st.info(chat['answer'])

user_q = st.text_input("💬 Posez votre question :")

if st.button("Envoyer"):
    if user_q.strip():
        with st.spinner("Recherche de la réponse..."):
            answer, lang_used = qa_multilang(user_q)
        st.session_state.history.append({"question": user_q, "answer": answer})
        st.rerun()
    else:
        st.warning("Veuillez entrer une question.")

if st.button("🗑 Effacer la conversation"):
    st.session_state.history = []
    st.success("Conversation réinitialisée.")
