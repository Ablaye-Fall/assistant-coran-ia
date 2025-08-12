# app_final.py
import streamlit as st
import json
import re
import html
import requests
from io import BytesIO
import tempfile
from langdetect import detect, DetectorFactory

# ML libs
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import MarianTokenizer, MarianMTModel, pipeline
import joblib

# Optional niceties (fallbacks handled)
try:
    import language_tool_python
    _LANG_TOOL_AVAILABLE = True
except Exception:
    _LANG_TOOL_AVAILABLE = False

try:
    from gtts import gTTS
    _GTTS_AVAILABLE = True
except Exception:
    _GTTS_AVAILABLE = False

DetectorFactory.seed = 0
st.set_page_config(page_title="Qalam-Quran-AI (final)", page_icon="🕋", layout="centered")

# -------------------------
# Config (modifie si besoin)
# -------------------------
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SUMMARY_MODEL = "csebuetnlp/mT5_multilingual_XLSum"  # ou une alternative disponible
TOP_K = 8
RERANK_THRESHOLD = 0.35  # ajuster selon qualité
TAFSIR_JSON = "sq-saadi.json"
TAFSIR_KEYS = "tafsir_keys.json"
TAFSIR_EMB = "tafsir_embeddings.npy"
TAFSIR_INDEX = "tafsir_index_sklearn.joblib"

# -------------------------
# Utilitaires texte & HTML
# -------------------------
def nettoyer_html(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def safe_detect(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"

# -------------------------
# Chargement des ressources tafsir & embeddings (une seule fois)
# -------------------------
@st.cache_resource(show_spinner=False)
def load_tafsir_resources():
    # charge tafsir complet + keys + embeddings + index sklearn/joblib
    try:
        with open(TAFSIR_JSON, "r", encoding="utf-8") as f:
            tafsir_data = json.load(f)
    except FileNotFoundError:
        tafsir_data = {}

    try:
        with open(TAFSIR_KEYS, "r", encoding="utf-8") as f:
            tafsir_keys = json.load(f)  # expects list of dicts with {"key":"s:v","tafsir": "..."}
    except FileNotFoundError:
        tafsir_keys = []

    try:
        embeddings = __import__("numpy").load(TAFSIR_EMB)
    except Exception:
        embeddings = None

    try:
        index = joblib.load(TAFSIR_INDEX)
    except Exception:
        index = None

    return tafsir_data, tafsir_keys, embeddings, index

tafsir_data, tafsir_keys, tafsir_embeddings, tafsir_index = load_tafsir_resources()

# -------------------------
# Models: embedder (cache), reranker & summarizer lazy-loaded
# -------------------------
@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

# Lazy loads for heavy models (loaded on first use)
_reranker = None
_summarizer = None
_lang_tools = None

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker

def get_summarizer():
    global _summarizer
    if _summarizer is None:
        # instantiate a summarization pipeline; may be heavy
        _summarizer = pipeline("summarization", model=SUMMARY_MODEL, tokenizer=SUMMARY_MODEL)
    return _summarizer

def get_lang_tool():
    global _lang_tools
    if _lang_tools is None and _LANG_TOOL_AVAILABLE:
        _lang_tools = language_tool_python.LanguageTool('fr')  # default FR; on peut paramétrer dynamiquement
    return _lang_tools

embedder = load_embedder()

# -------------------------
# MarianMT translation cache (par paire src-tgt)
# -------------------------
@st.cache_resource
def load_marian(src: str, tgt: str):
    model_name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    return tokenizer, model

def marian_translate(text: str, src: str, tgt: str) -> str:
    """
    Traduction via Marian (cached per paire). Si modèle introuvable,
    on renvoie le texte d'origine.
    """
    try:
        tokenizer, model = load_marian(src, tgt)
        batch = tokenizer([text], return_tensors="pt", padding=True)
        gen = model.generate(**batch)
        out = tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
        return out
    except Exception:
        # fallback to deep_translator (GoogleTranslator) si Marian indisponible
        try:
            from deep_translator import GoogleTranslator
            return GoogleTranslator(source=src, target=tgt).translate(text)
        except Exception:
            return text

# -------------------------
# Recherche sémantique + reranking
# -------------------------
def semantic_search_and_rerank(question_emb, corpus_texts, corpus_embeddings, top_k=TOP_K):
    """
    Utilise sentence-transformers util.semantic_search puis CrossEncoder reranker.
    corpus_embeddings can be numpy array or torch tensor.
    Returns list of (passage, score) sorted desc.
    """
    hits = util.semantic_search(question_emb, corpus_embeddings, top_k=top_k)[0]
    passages = [corpus_texts[h['corpus_id']] for h in hits]
    reranker = get_reranker()
    pairs = [[_q_text, p] for p in passages for _q_text in [question_alb]]  # question_alb defined in outer scope when called
    scores = reranker.predict(pairs)
    # pairwise mapping (scores in order)
    scored = list(zip(passages, scores))
    scored_sorted = sorted(scored, key=lambda x: x[1], reverse=True)
    return scored_sorted

# -------------------------
# Résumé local (via summarizer) + correction optionnelle
# -------------------------
def summarize_text(text: str, max_words=150):
    try:
        summarizer = get_summarizer()
        out = summarizer(text, max_length=max_words, min_length=40, do_sample=False)
        return out[0]['summary_text']
    except Exception:
        # fallback simple truncation
        words = text.split()
        return " ".join(words[:max_words])

def correct_text_language_tool(text: str, lang_code='fr'):
    if not _LANG_TOOL_AVAILABLE:
        return text
    tool = get_lang_tool()
    if not tool:
        return text
    try:
        return tool.correct(text)
    except Exception:
        return text

# -------------------------
# QA pipeline centralisée
# -------------------------
def qa_pipeline(user_question: str):
    """
    1) detect language
    2) translate -> albanais (sq) if needed for embeddings
    3) embed question & semantic_search
    4) rerank with CrossEncoder and threshold
    5) cleanup, summarize, correct
    6) translate back to original language
    Returns (answer_text, detected_lang)
    """
    lang = safe_detect(user_question)
    if lang == "unknown":
        return "Impossible de détecter la langue. Peux-tu reformuler ?", lang

    # translate to Albanian for embedding-space (if desired); if not present, we can embed in original lang
    # note: if you don't have Marian for src->sq, fallback to GoogleTranslator
    try:
        # try to detect and translate via Marian if possible, else fallback to deep_translator
        question_sq = marian_translate(user_question, lang, "sq")
    except Exception:
        from deep_translator import GoogleTranslator
        try:
            question_sq = GoogleTranslator(source='auto', target='sq').translate(user_question)
        except Exception:
            question_sq = user_question  # fallback: no translation

    # embed
    q_emb = embedder.encode(question_sq, convert_to_tensor=True)

    # prepare corpus_texts and embeddings from tafsir_keys (we expect tafsir_keys to be list of dicts with 'tafsir')
    corpus_texts = [tk.get("tafsir", "") for tk in tafsir_keys]
    corpus_embeddings = tafsir_embeddings if tafsir_embeddings is not None else None

    if not corpus_texts or corpus_embeddings is None:
        return "La base de tafsir / embeddings n'est pas disponible sur le serveur.", lang

    # semantic search (use util)
    hits = util.semantic_search(q_emb, corpus_embeddings, top_k=TOP_K)[0]
    passages = [corpus_texts[h['corpus_id']] for h in hits]

    # rerank
    reranker = get_reranker()
    pairs = [[question_sq, p] for p in passages]
    try:
        scores = reranker.predict(pairs)
    except Exception:
        scores = [0.0] * len(passages)

    scored = sorted(list(zip(passages, scores)), key=lambda x: x[1], reverse=True)
    best_passage, best_score = scored[0]

    if best_score < RERANK_THRESHOLD:
        return "Aucune réponse pertinente trouvée (score trop faible).", lang

    cleaned = nettoyer_html(best_passage)

    # summarize and correct
    summarized = summarize_text(cleaned, max_words=150)
    # prefer correction in detected language if tool available
    corrected = correct_text_language_tool(summarized, lang_code=lang if lang in ['fr','en','es'] else 'fr')

    # translate back to original language if we forced sq
    try:
        # attempt marian back translation sq -> lang
        final = marian_translate(corrected, "sq", lang) if lang != "sq" else corrected
    except Exception:
        from deep_translator import GoogleTranslator
        try:
            final = GoogleTranslator(source='sq', target=lang).translate(corrected) if lang != "sq" else corrected
        except Exception:
            final = corrected

    return final, lang

# -------------------------
# API Quran helpers (safe)
# -------------------------
@st.cache_data(ttl=86400)
def get_surahs():
    try:
        resp = requests.get("http://api.alquran.cloud/v1/surah", timeout=6)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception:
        return []

@st.cache_data(ttl=86400)
def get_surah_editions(surah_num, translation_code="en.asad"):
    try:
        url = f"http://api.alquran.cloud/v1/surah/{surah_num}/editions/quran-simple,{translation_code}"
        resp = requests.get(url, timeout=6)
        resp.raise_for_status()
        return resp.json().get("data", [])
    except Exception:
        return []

@st.cache_data(ttl=86400)
def get_ayah_audio(surah_num, ayah_num, reciter="ar.alafasy"):
    try:
        resp = requests.get(f"http://api.alquran.cloud/v1/ayah/{surah_num}:{ayah_num}/{reciter}", timeout=6)
        resp.raise_for_status()
        data = resp.json().get("data", {})
        return data.get("audio", None)
    except Exception:
        return None

# -------------------------
# Interface Streamlit
# -------------------------
st.title("📖 Qalam-Quran-AI — Final optimisé")
st.markdown("Lecture, tafsir et Q&A multilingue — embeddings + rerank + résumé & correction.")

# Surah selector
surahs = get_surahs()
surah_names = [f"{s['number']}. {s['englishName']} ({s['name']})" for s in surahs] if surahs else []
choix_surah = st.selectbox("📚 Choisissez une sourate :", ["1. Al-Fatihah"] + surah_names)
num_surah = int(choix_surah.split(".")[0]) if choix_surah else 1

# translation choice
translations = {
    "🇫🇷 Français (Hamidullah)": "fr.hamidullah",
    "🇬🇧 Anglais (Muhammad Asad)": "en.asad",
    "🇮🇩 Indonésien": "id.indonesian"
}
traduction_choisie = st.selectbox("🌐 Choisissez une traduction :", list(translations.keys()))
code_trad = translations.get(traduction_choisie, "en.asad")

# fetch verses for UI (safe)
verses_data = get_surah_editions(num_surah, code_trad)
verses_ar = verses_data[0]["ayahs"] if len(verses_data) > 0 else []
verses_trad = verses_data[1]["ayahs"] if len(verses_data) > 1 else []

max_verset = len(verses_ar) if verses_ar else 1
verset_num = st.number_input("📌 Choisissez un verset :", min_value=1, max_value=max_verset, value=1)

verset_ar = verses_ar[verset_num - 1]["text"] if verses_ar else ""
verset_trad = verses_trad[verset_num - 1]["text"] if verses_trad else ""

# Verset display with zoom + optional popup-like behaviour
zoom = st.slider("🔍 Zoom texte arabe", min_value=1.0, max_value=3.0, value=1.6, step=0.1)
st.markdown(f"<div style='font-size:{zoom}em; direction:rtl; text-align:right; font-weight:600;'>{verset_ar}</div>", unsafe_allow_html=True)
st.write(f"*{verset_trad}*")

# audio verse
reciters = {
    "🎙 Mishary Rashid Alafasy": "ar.alafasy",
    "🎙 Abdul Basit": "ar.abdulbasitmurattal",
    "🎙 Saad Al-Ghamdi": "ar.saoodshuraim"
}
reciter_choice = st.selectbox("🎧 Choisissez un récitateur :", list(reciters.keys()))
audio_url = get_ayah_audio(num_surah, verset_num, reciters.get(reciter_choice, "ar.alafasy"))
if audio_url:
    st.audio(audio_url, format="audio/mp3")

# Tafsir display (from tafsir_data or tafsir_keys)
cle = f"{num_surah}:{verset_num}"
tafsir_entry = tafsir_data.get(cle) if isinstance(tafsir_data, dict) else None
tafsir_text = ""
if tafsir_entry:
    # if structure is dict with 'text'
    if isinstance(tafsir_entry, dict):
        tafsir_text = tafsir_entry.get("text") or tafsir_entry.get("tafsir") or ""
    else:
        tafsir_text = str(tafsir_entry)
else:
    # fallback look in tafsir_keys list
    for tk in tafsir_keys:
        if tk.get("key") == cle:
            tafsir_text = tk.get("tafsir", "")
            break

tafsir_clean = nettoyer_html(tafsir_text)
st.subheader("📜 Tafsir (local)")
if tafsir_clean:
    # small UI: language for tafsir translation
    lang_tafsir_view = st.selectbox("🌐 Voir le tafsir en :", ["fr", "en", "ar", "es"])
    # attempt marian or fallback translation
    if lang_tafsir_view != "auto" and lang_tafsir_view != "ar" and tafsir_clean:
        try:
            tafsir_trans = marian_translate(tafsir_clean, "sq", lang_tafsir_view) if tafsir_clean else ""
        except Exception:
            from deep_translator import GoogleTranslator
            tafsir_trans = GoogleTranslator(source='auto', target=lang_tafsir_view).translate(tafsir_clean)
    else:
        tafsir_trans = tafsir_clean
    st.write(tafsir_trans)
else:
    st.info("Aucun tafsir local trouvé pour ce verset.")

# -------------------------
# Q&A section
# -------------------------
st.markdown("---")
st.subheader("❓ Q&A multilingue (embed → rerank → résumé → correction → audio)")

if "history" not in st.session_state:
    st.session_state.history = []

col1, col2 = st.columns([4,1])
with col1:
    user_q = st.text_input("💬 Pose ta question (toute langue) :", key="q_in")
with col2:
    if st.button("Envoyer"):
        if user_q and user_q.strip():
            with st.spinner("Recherche..."):
                answer, lang_detected = qa_pipeline(user_q)
            st.session_state.history.insert(0, {"q": user_q, "a": answer, "lang": lang_detected})
        else:
            st.warning("Veuillez saisir une question.")

# display history
for chat in st.session_state.history:
    st.markdown(f"**🧑‍💻 Vous ({chat['lang']}):** {chat['q']}")
    st.markdown(f"**🤖 Réponse :** {chat['a']}")
    # audio generation for the answer (gTTS) — avoid if unavailable
    if _GTTS_AVAILABLE:
        try:
            tts = gTTS(chat['a'], lang=chat['lang'] if chat['lang'] in ['fr','en','es','ar'] else 'fr')
            bio = BytesIO()
            tts.write_to_fp(bio)
            st.audio(bio.getvalue(), format="audio/mp3")
        except Exception:
            # non bloquant
            pass

# Reset history
if st.button("🗑 Effacer l'historique"):
    st.session_state.history = []
    st.success("Historique vidé.")

