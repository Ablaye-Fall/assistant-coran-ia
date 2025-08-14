# app_final.py — Qalam-Quran-AI Optimisé (stable DOM, sans joblib)
import streamlit as st
import json
import re
import html
import requests
from io import BytesIO
from langdetect import detect, DetectorFactory
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import MarianTokenizer, MarianMTModel, pipeline
import numpy as np

# Optional modules
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

# Basic config
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

# ---------------- FONT IMPORT (single injection) ----------------
FONT_IMPORT_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Amiri:wght@400;700&family=Scheherazade+New:wght@400;700&family=Noto+Naskh+Arabic:wght@400;700&display=swap');
.arabic-verse {
    direction: rtl;
    -webkit-font-smoothing: antialiased;
    font-variant-ligatures: normal;
}
</style>
"""
if "fonts_injected" not in st.session_state:
    st.markdown(FONT_IMPORT_CSS, unsafe_allow_html=True)
    st.session_state["fonts_injected"] = True

# ---------------- FONT PRESETS ----------------
FONT_PRESETS = {
    "Amiri (classique)": ("'Amiri','Scheherazade New','Noto Naskh Arabic',serif"),
    "Scheherazade New": ("'Scheherazade New','Amiri','Noto Naskh Arabic',serif"),
    "Noto Naskh Arabic": ("'Noto Naskh Arabic','Amiri','Scheherazade New',serif"),
    "Noto Kufi Arabic": ("'Noto Kufi Arabic','Noto Naskh Arabic','Amiri',serif"),
    "Lateef (calligraphie)": ("'Lateef','Scheherazade New','Noto Naskh Arabic',serif"),
}

# Harakat toggle
_HARAKAT_RE = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
def toggle_harakat(txt: str, show: bool) -> str:
    return txt if show else _HARAKAT_RE.sub("", txt)

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

    return tafsir_data, tafsir_keys, embeddings

tafsir_data, tafsir_keys, tafsir_embeddings = load_tafsir_resources()

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
    if not text:
        return text
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
    if tool is None:
        return text
    try:
        return tool.correct(text) if lang_code == 'fr' else text
    except Exception:
        return text

# ---------------- QA PIPELINE ----------------
def qa_pipeline(user_question: str):
    lang = safe_detect(user_question)
    if lang == "unknown":
        return "Impossible de détecter la langue.", lang

    # Traduire vers le corpus (sq)
    question_sq = marian_translate(user_question, lang, "sq")
    q_emb = embedder.encode(question_sq, convert_to_tensor=True)

    # Construire corpus + embeddings
    corpus_texts = []
    for tk in tafsir_keys:
        entry = tafsir_data.get(tk, {})
        if isinstance(entry, dict):
            txt = entry.get("tafsir") or entry.get("text") or ""
        else:
            txt = str(entry) if entry else ""
        corpus_texts.append(txt)

    corpus_embeddings = tafsir_embeddings
    if not corpus_texts or corpus_embeddings is None:
        return "Base de données indisponible.", lang

    scored = semantic_search_and_rerank(question_sq, q_emb, corpus_texts, corpus_embeddings)
    if not scored:
        return "Aucune réponse trouvée.", lang

    best_passage, best_score = scored[0]
    if best_score < RERANK_THRESHOLD:
        return "Aucune réponse pertinente trouvée.", lang

    cleaned = nettoyer_html(best_passage)
    summarized = summarize_text(cleaned)
    output_in_user_lang = marian_translate(summarized, "sq", lang) if lang != "sq" else summarized
    corrected = correct_text_language_tool(output_in_user_lang, lang_code=lang if lang in ['fr','en','es'] else 'fr')
    return corrected, lang

# ---------------- API ALQURAN ----------------
@st.cache_data(ttl=86400, show_spinner=False)
def get_surahs():
    try:
        return requests.get("http://api.alquran.cloud/v1/surah", timeout=6).json().get("data", [])
    except Exception:
        return []

@st.cache_data(ttl=86400, show_spinner=False)
def get_surah_editions(surah_num, translation_code="en.asad"):
    try:
        url = f"http://api.alquran.cloud/v1/surah/{surah_num}/editions/quran-simple,{translation_code}"
        return requests.get(url, timeout=6).json().get("data", [])
    except Exception:
        return []

@st.cache_data(ttl=86400, show_spinner=False)
def get_ayah_audio(surah_num, ayah_num, reciter="ar.alafasy"):
    try:
        data = requests.get(f"http://api.alquran.cloud/v1/ayah/{surah_num}:{ayah_num}/{reciter}", timeout=6).json().get("data", {})
        return data.get("audio", None)
    except Exception:
        return None

# ---------------- INTERFACE ----------------
st.title("📖 Qalam-Quran-AI — Final Optimisé")
st.markdown("Lecture, tafsir et Q&A multilingue avec embeddings, rerank, résumé et correction.")

# Sélecteurs sourate / traduction
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

# Récupérations éditions (robuste)
verses_data = get_surah_editions(num_surah, code_trad)
verses_ar, verses_trad = [], []
if verses_data:
    ed_by_id = {}
    for ed in verses_data:
        ident = (ed.get("edition") or {}).get("identifier", "")
        ed_by_id[ident] = ed
    if "quran-simple" in ed_by_id:
        verses_ar = ed_by_id["quran-simple"].get("ayahs", [])
    if code_trad in ed_by_id:
        verses_trad = ed_by_id[code_trad].get("ayahs", [])
    else:
        for k, v in ed_by_id.items():
            if k.endswith(code_trad.split(".")[-1]):
                verses_trad = v.get("ayahs", [])
                break

max_verset = len(verses_ar) if verses_ar else 1
verset_num = st.number_input("📌 Choisissez un verset :", 1, max_verset if max_verset > 0 else 1, 1)

verset_ar = verses_ar[verset_num - 1]["text"] if verses_ar else ""
verset_trad = verses_trad[verset_num - 1]["text"] if verses_trad else ""

# Préparer tafsir local (texte) — avant placeholders pour pouvoir choisir la langue TTS
cle = f"{num_surah}:{verset_num}"
_t = tafsir_data.get(cle, {})
if isinstance(_t, dict):
    tafsir_text = _t.get("tafsir") or _t.get("text") or ""
else:
    tafsir_text = str(_t) if _t else ""
tafsir_clean = nettoyer_html(tafsir_text)

# ----- Options d'affichage du texte arabe (UI) -----
with st.expander("🧰 Options d’affichage du texte arabe", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        font_label = st.selectbox("Police", list(FONT_PRESETS.keys()), index=2)
        font_size = st.slider("Taille (px)", 18, 96, 42, step=2)
        line_height = st.slider("Interligne (em)", 1.0, 3.0, 1.9, step=0.1)
        align = st.selectbox("Alignement", ["right", "justify", "center"], index=0)
    with col2:
        text_color = st.color_picker("Couleur du texte", "#111111")
        bg_color = st.color_picker("Fond", "#FFFFFF")
        letter_spacing = st.slider("Espacement lettres (px)", 0.0, 3.0, 0.0, step=0.1)
        word_spacing = st.slider("Espacement mots (px)", 0.0, 8.0, 0.0, step=0.5)
    col3, col4 = st.columns(2)
    with col3:
        show_shadow = st.checkbox("Ombre légère", value=False)
    with col4:
        show_harakat = st.checkbox("Afficher les harakât (voyelles)", value=True)

# Choix langue tafsir (avant affichage)
st.subheader("📜 Tafsir (local)")
lang_tafsir_view = st.selectbox("🌐 Voir le tafsir en :", ["fr", "en", "ar", "es"])
if tafsir_clean:
    src_lang = safe_detect(tafsir_clean)
    try:
        tafsir_trans = marian_translate(tafsir_clean, src_lang, lang_tafsir_view)
    except Exception:
        tafsir_trans = tafsir_clean
else:
    tafsir_trans = ""

# ---------------- Stable placeholders for verse / translation / tafsir ----------------
if "placeholders_created" not in st.session_state:
    st.session_state["ph_verse"] = st.empty()
    st.session_state["ph_trad"] = st.empty()
    st.session_state["ph_tafsir"] = st.empty()
    st.session_state["ph_tafsir_audio"] = st.empty()
    st.session_state["placeholders_created"] = True

ph_verse = st.session_state["ph_verse"]
ph_trad = st.session_state["ph_trad"]
ph_tafsir = st.session_state["ph_tafsir"]
ph_tafsir_audio = st.session_state["ph_tafsir_audio"]

# Prepare verse display values
font_stack = FONT_PRESETS.get(font_label, FONT_PRESETS["Noto Naskh Arabic"])
verset_ar_display = toggle_harakat(verset_ar or "", show_harakat)
text_shadow_css = "text-shadow: 0 1px 1px rgba(0,0,0,.25);" if show_shadow else ""

verse_html = f"""
<div class="arabic-verse" style="
    text-align:{align};
    font-family:{font_stack};
    font-size:{font_size}px;
    line-height:{line_height}em;
    color:{text_color};
    background:{bg_color};
    padding:16px 20px;
    border-radius:12px;
    letter-spacing:{letter_spacing}px;
    word-spacing:{word_spacing}px;
    {text_shadow_css}
    font-weight:600;
">
{verset_ar_display}
</div>
"""

# Render verse (safe)
try:
    ph_verse.markdown(verse_html, unsafe_allow_html=True)
except Exception:
    ph_verse.write(verset_ar_display)

# Render translation
ph_trad.write(f"*{verset_trad}*")

# Render tafsir text and audio (in dedicated stable placeholders)
if tafsir_clean:
    try:
        ph_tafsir.write(tafsir_trans)
    except Exception:
        ph_tafsir.write(tafsir_clean)

    # Clear previous audio then set new audio if available
    ph_tafsir_audio.empty()
    if _GTTS_AVAILABLE and isinstance(tafsir_trans, str) and len(tafsir_trans) > 0:
        try:
            tts_lang = lang_tafsir_view if lang_tafsir_view in ['fr', 'en', 'es', 'ar'] else 'fr'
            tts = gTTS(tafsir_trans, lang=tts_lang)
            bio = BytesIO()
            tts.write_to_fp(bio)
            bio.seek(0)
            ph_tafsir_audio.audio(bio.read(), format="audio/mp3")
        except Exception as e:
            ph_tafsir_audio.info("Lecture audio non disponible pour le tafsir.")
            # log for dev
            st.exception(e)
else:
    ph_tafsir.info("Aucun tafsir local trouvé pour ce verset.")
    ph_tafsir_audio.empty()

# Audio du verset (récitateur)
reciters = {
    "🎙 Mishary Rashid Alafasy": "ar.alafasy",
    "🎙 Abdul Basit": "ar.abdulbasitmurattal",
    "🎙 Saad Al-Ghamdi": "ar.saoodshuraim"
}
reciter_choice = st.selectbox("🎧 Choisissez un récitateur :", list(reciters.keys()))
audio_url = get_ayah_audio(num_surah, verset_num, reciters.get(reciter_choice, "ar.alafasy"))
if audio_url:
    try:
        st.audio(audio_url)
    except Exception:
        # fallback silent
        st.info("Audio verset indisponible.")

# ---------------- Q&A (robust with placeholders & form) ----------------
st.markdown("---")
st.subheader("❓ Q&A multilingue")

if "history" not in st.session_state:
    st.session_state.history = []

if "ph_qa_list" not in st.session_state:
    st.session_state["ph_qa_list"] = st.empty()

with st.form(key="qa_form_v2", clear_on_submit=False):
    user_q = st.text_input("💬 Pose ta question :", key="q_in_form_v2")
    submit = st.form_submit_button("Envoyer")

if submit:
    if user_q and user_q.strip():
        with st.spinner("Recherche..."):
            try:
                answer, lang_detected = qa_pipeline(user_q.strip())
            except Exception as e:
                st.error("Erreur lors du traitement de la question.")
                st.exception(e)
                answer, lang_detected = "Erreur interne.", "unknown"
        st.session_state.history.insert(0, {"q": user_q.strip(), "a": answer, "lang": lang_detected})
    else:
        st.warning("Veuillez saisir une question.")

# Render history atomically in the placeholder container
ph_list = st.session_state["ph_qa_list"]
with ph_list.container():
    for idx, chat in enumerate(st.session_state.history):
        q_text = chat.get("q", "")
        a_text = chat.get("a", "")
        lang = chat.get("lang", "unknown")
        st.markdown(f"**🧑‍💻 Vous ({lang}):** {q_text}")
        st.markdown(f"**🤖 Réponse :** {a_text}")

        # TTS for each answer if available
        if _GTTS_AVAILABLE and isinstance(a_text, str) and len(a_text) > 0:
            try:
                tts_lang = lang if lang in ['fr', 'en', 'es', 'ar'] else 'fr'
                tts = gTTS(a_text, lang=tts_lang)
                bio = BytesIO()
                tts.write_to_fp(bio)
                bio.seek(0)
                st.audio(bio.read(), format="audio/mp3")
            except Exception:
                st.info("Lecture audio non disponible pour cette réponse.")

if st.button("🗑 Effacer l'historique"):
    st.session_state.history.clear()
    st.success("Historique vidé.")

