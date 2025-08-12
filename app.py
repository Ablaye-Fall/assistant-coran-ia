# app_final.py — Qalam-Quran-AI Optimisé
import streamlit as st
import json
import re
import html
import requests
from io import BytesIO
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import MarianTokenizer, MarianMTModel, pipeline
import joblib
import numpy as np

# Modules optionnels
try:
    import language_tool_python
    _LANG_TOOL_AVAILABLE = True
except ImportError:
    _LANG_TOOL_AVAILABLE = False

try:
    from gtts import gTTS
    _GTTS_AVAILABLE = True
except ImportError:
    _GTTS_AVAILABLE = False

DetectorFactory.seed = 0
st.set_page_config(page_title="Qalam-Quran-AI (Final)", page_icon="🕋", layout="centered")

# ---------------- CONFIG ----------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SUMMARY_MODEL = "csebuetnlp/mT5_multilingual_XLSum"
TOP_K = 8
RERANK_THRESHOLD = 0.35
TAFSIR_JSON = "sq-saadi.json"
TAFSIR_KEYS = "tafsir_keys.json"
TAFSIR_EMB = "tafsir_embeddings.npy"
TAFSIR_INDEX = "tafsir_index_sklearn.joblib"

# ---------------- UTILITAIRES ----------------
def nettoyer_html(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def safe_detect(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"

# ---------------- CHARGEMENT RESSOURCES ----------------
@st.cache_resource(show_spinner=False)
def load_tafsir_resources():
    try:
        with open(TAFSIR_JSON, "r", encoding="utf-8") as f:
            tafsir_data = json.load(f)
    except FileNotFoundError:
        tafsir_data = {}

    try:
        with open(TAFSIR_KEYS, "r", encoding="utf-8") as f:
            tafsir_keys = json.load(f)
    except FileNotFoundError:
        tafsir_keys = []

    try:
        embeddings = np.load(TAFSIR_EMB)
    except Exception:
        embeddings = None

    try:
        index = joblib.load(TAFSIR_INDEX)
    except Exception:
        index = None

    return tafsir_data, tafsir_keys, embeddings, index

tafsir_data, tafsir_keys, tafsir_embeddings, tafsir_index = load_tafsir_resources()

# ---------------- MODELS ----------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

embedder = load_embedder()
_reranker = None
_summarizer = None
_lang_tool = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        _summarizer = pipeline("summarization", model=SUMMARY_MODEL, tokenizer=SUMMARY_MODEL)
    return _summarizer

def get_lang_tool():
    global _lang_tool
    if _lang_tool is None and _LANG_TOOL_AVAILABLE:
        _lang_tool = language_tool_python.LanguageTool('fr')
    return _lang_tool

# ---------------- TRADUCTION ----------------
@st.cache_resource
def load_marian(src: str, tgt: str):
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def marian_translate(text: str, src: str, tgt: str) -> str:
    if src == tgt:
        return text
    try:
        tokenizer, model = load_marian(src, tgt)
        batch = tokenizer([text], return_tensors="pt", padding=True)
        gen = model.generate(**batch)
        return tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
    except Exception:
        try:
            from deep_translator import GoogleTranslator
            return GoogleTranslator(source=src, target=tgt).translate(text)
        except Exception:
            return text

# ---------------- RECHERCHE & RERANK ----------------
def semantic_search_and_rerank(question_text, question_emb, corpus_texts, corpus_embeddings, top_k=TOP_K):
    hits = util.semantic_search(question_emb, corpus_embeddings, top_k=top_k)[0]
    passages = [corpus_texts[h['corpus_id']] for h in hits]
    reranker = get_reranker()
    scores = reranker.predict([[question_text, p] for p in passages])
    return sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)

# ---------------- RESUME & CORRECTION ----------------
def summarize_text(text: str, max_words=150):
    try:
        summarizer = get_summarizer()
        out = summarizer(text, max_length=max_words, min_length=40, do_sample=False)
        return out[0]['summary_text']
    except Exception:
        return " ".join(text.split()[:max_words])

def correct_text_language_tool(text: str, lang_code='fr'):
    if not _LANG_TOOL_AVAILABLE:
        return text
    tool = get_lang_tool()
    try:
        return tool.correct(text)
    except Exception:
        return text

# ---------------- QA PIPELINE ----------------
def qa_pipeline(user_question: str):
    lang = safe_detect(user_question)
    if lang == "unknown":
        return "Impossible de détecter la langue.", lang

    question_sq = marian_translate(user_question, lang, "sq")
    q_emb = embedder.encode(question_sq, convert_to_tensor=True)

    corpus_texts = [tk.get("tafsir", "") for tk in tafsir_keys]
    corpus_embeddings = tafsir_embeddings

    if not corpus_texts or corpus_embeddings is None:
        return "Base de données indisponible.", lang

    scored = semantic_search_and_rerank(question_sq, q_emb, corpus_texts, corpus_embeddings)
    best_passage, best_score = scored[0]

    if best_score < RERANK_THRESHOLD:
        return "Aucune réponse pertinente trouvée.", lang

    cleaned = nettoyer_html(best_passage)
    summarized = summarize_text(cleaned)
    corrected = correct_text_language_tool(summarized, lang_code=lang if lang in ['fr','en','es'] else 'fr')
    final = marian_translate(corrected, "sq", lang) if lang != "sq" else corrected
    return final, lang

# ---------------- API ALQURAN ----------------
@st.cache_data(ttl=86400)
def get_surahs():
    try:
        return requests.get("http://api.alquran.cloud/v1/surah", timeout=6).json().get("data", [])
    except Exception:
        return []

@st.cache_data(ttl=86400)
def get_surah_editions(surah_num, translation_code="en.asad"):
    try:
        url = f"http://api.alquran.cloud/v1/surah/{surah_num}/editions/quran-simple,{translation_code}"
        return requests.get(url, timeout=6).json().get("data", [])
    except Exception:
        return []

@st.cache_data(ttl=86400)
def get_ayah_audio(surah_num, ayah_num, reciter="ar.alafasy"):
    try:
        data = requests.get(f"http://api.alquran.cloud/v1/ayah/{surah_num}:{ayah_num}/{reciter}", timeout=6).json().get("data", {})
        return data.get("audio", None)
    except Exception:
        return None

# ---------------- INTERFACE ----------------
st.title("📖 Qalam-Quran-AI — Final Optimisé")
st.markdown("Lecture, tafsir et Q&A multilingue avec embeddings, rerank, résumé et correction.")

surahs = get_surahs()
surah_names = [f"{s['number']}. {s['englishName']} ({s['name']})" for s in surahs]
choix_surah = st.selectbox("📚 Choisissez une sourate :", surah_names)
num_surah = int(choix_surah.split(".")[0]) if choix_surah else 1

translations = {
    "🇫🇷 Français (Hamidullah)": "fr.hamidullah",
    "🇬🇧 Anglais (Muhammad Asad)": "en.asad",
    "🇮🇩 Indonésien": "id.indonesian"
}
traduction_choisie = st.selectbox("🌐 Choisissez une traduction :", list(translations.keys()))
code_trad = translations.get(traduction_choisie, "en.asad")

verses_data = get_surah_editions(num_surah, code_trad)
verses_ar = verses_data[0]["ayahs"] if verses_data else []
verses_trad = verses_data[1]["ayahs"] if len(verses_data) > 1 else []

max_verset = len(verses_ar) if verses_ar else 1
verset_num = st.number_input("📌 Choisissez un verset :", 1, max_verset, 1)

verset_ar = verses_ar[verset_num - 1]["text"] if verses_ar else ""
verset_trad = verses_trad[verset_num - 1]["text"] if verses_trad else ""

zoom = st.slider("🔍 Zoom texte arabe", 1.0, 3.0, 1.6, 0.1)
st.markdown(f"<div style='font-size:{zoom}em; direction:rtl; text-align:right; font-weight:600;'>{verset_ar}</div>", unsafe_allow_html=True)
st.write(f"*{verset_trad}*")

reciters = {
    "🎙 Mishary Rashid Alafasy": "ar.alafasy",
    "🎙 Abdul Basit": "ar.abdulbasitmurattal",
    "🎙 Saad Al-Ghamdi": "ar.saoodshuraim"
}
reciter_choice = st.selectbox("🎧 Choisissez un récitateur :", list(reciters.keys()))
audio_url = get_ayah_audio(num_surah, verset_num, reciters.get(reciter_choice, "ar.alafasy"))
if audio_url:
    st.audio(audio_url)

cle = f"{num_surah}:{verset_num}"
tafsir_text = tafsir_data.get(cle, {}).get("tafsir", "")
tafsir_clean = nettoyer_html(tafsir_text)
st.subheader("📜 Tafsir (local)")
if tafsir_clean:
    lang_tafsir_view = st.selectbox("🌐 Voir le tafsir en :", ["fr", "en", "ar", "es"])
    src_lang = safe_detect(tafsir_clean)
    tafsir_trans = marian_translate(tafsir_clean, src_lang, lang_tafsir_view)
    st.write(tafsir_trans)
else:
    st.info("Aucun tafsir local trouvé pour ce verset.")

# Q&A
st.markdown("---")
st.subheader("❓ Q&A multilingue")
if "history" not in st.session_state:
    st.session_state.history = []

user_q = st.text_input("💬 Pose ta question :", key="q_in")
if st.button("Envoyer"):
    if user_q.strip():
        with st.spinner("Recherche..."):
            answer, lang_detected = qa_pipeline(user_q)
        st.session_state.history.insert(0, {"q": user_q, "a": answer, "lang": lang_detected})
    else:
        st.warning("Veuillez saisir une question.")

for chat in st.session_state.history:
    st.markdown(f"**🧑‍💻 Vous ({chat['lang']}):** {chat['q']}")
    st.markdown(f"**🤖 Réponse :** {chat['a']}")
    if _GTTS_AVAILABLE:
        try:
            tts_lang = chat['lang'] if chat['lang'] in ['fr','en','es','ar'] else 'fr'
            tts = gTTS(chat['a'], lang=tts_lang)
            bio = BytesIO()
            tts.write_to_fp(bio)
            st.audio(bio.getvalue())
        except Exception:
            pass

if st.button("🗑 Effacer l'historique"):
    st.session_state.history.clear()
    st.success("Historique vidé.")

