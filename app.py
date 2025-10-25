import json
import io
import re
import tempfile
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
    """Enlève les balises HTML basiques que l'on trouve parfois dans le tafsir."""
    if not texte:
        return ""
    texte = re.sub(r'<b><i>.*?</i></b>', '', texte, flags=re.DOTALL)
    return re.sub(r'<[^>]+>', '', texte)

def safe_translate(text, target):
    """Traduction avec catch des exceptions."""
    if not text:
        return ""
    try:
        return GoogleTranslator(source='auto', target=target).translate(text)
    except Exception as e:
        logging.warning(f"Traduction échouée vers {target} : {e}")
        return text

def reformulate_text(text, target_lang):
    """Double traduction pour reformulation (fallback -> renvoie texte original si erreur)."""
    if not text:
        return ""
    try:
        temp_en = GoogleTranslator(source='auto', target='en').translate(text)
        refined = GoogleTranslator(source='en', target=target_lang).translate(temp_en)
        return refined
    except Exception as e:
        logging.info(f"Reformulation échouée : {e}")
        return text

def detect_language_safe(text):
    """Détection de langue robuste : priorise la présence d'alphabets (ex : arabe)."""
    if not text or not text.strip():
        return "unknown"
    # Si du texte contient des caractères arabes, renvoie 'ar'
    if re.search(r'[\u0600-\u06FF]', text):
        return "ar"
    try:
        lang = detect(text)
    except Exception:
        lang = "unknown"
    return lang

# ---------------------------
# --- SIDEBAR ARABE (TOGGLE/ZOOM) ---
# ---------------------------
with st.sidebar:
    st.header("Affichage arabe")
    if "hide_harakat" not in st.session_state:
        st.session_state.hide_harakat = False
    if "zoom_em" not in st.session_state:
        st.session_state.zoom_em = 1.5

    st.session_state.hide_harakat = st.checkbox(
        "Masquer les harakât (diacritiques)", value=st.session_state.hide_harakat
    )
    st.session_state.zoom_em = st.slider(
        "🔍 Zoom texte arabe (em)", 1.0, 3.0, st.session_state.zoom_em, 0.1
    )

def render_verset(verset_ar: str, weight: int = 600):
    """Affiche un verset arabe en appliquant toggle + zoom.
       -> UTILISER UNIQUEMENT POUR LE VERSET ARABE."""
    display_text = remove_diacritics(verset_ar) if st.session_state.hide_harakat else verset_ar
    if not display_text:
        st.write("")  # rien à afficher
        return
    st.markdown(
        f"<p style='font-size:{st.session_state.zoom_em}em; direction:rtl; text-align:right; font-weight:{weight};'>{display_text}</p>",
        unsafe_allow_html=True
    )

# ---------------------------
# --- CHARGEMENTS (MODELES + RESSOURCES) ---
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_tafsir_resources():
    """Charge tafsir JSON, clés et embeddings + index NearestNeighbors."""
    try:
        with open("sq-saadi.json", "r", encoding="utf-8") as f:
            tafsir_data = json.load(f)
    except Exception as e:
        logging.error(f"Impossible de charger sq-saadi.json : {e}")
        tafsir_data = {}

    try:
        with open("tafsir_keys.json", "r", encoding="utf-8") as f:
            tafsir_keys = json.load(f)
    except Exception as e:
        logging.error(f"Impossible de charger tafsir_keys.json : {e}")
        tafsir_keys = []

    try:
        embeddings = np.load("tafsir_embeddings.npy")
    except Exception as e:
        logging.error(f"Impossible de charger tafsir_embeddings.npy : {e}")
        embeddings = np.zeros((0, 384))  # fallback

    nn_index = None
    if embeddings.size > 0:
        try:
            nn_index = NearestNeighbors(n_neighbors=10, metric='cosine')
            nn_index.fit(embeddings)
        except Exception as e:
            logging.error(f"Erreur création index NearestNeighbors : {e}")
            nn_index = None

    return tafsir_data, tafsir_keys, embeddings, nn_index

@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def load_reranker_model():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

tafsir_data, tafsir_keys, tafsir_embeddings, tafsir_index = load_tafsir_resources()
model = load_sentence_model()
reranker = load_reranker_model()

# ---------------------------
# --- API QURAN (stable wrappers) ---
# ---------------------------
@st.cache_data(ttl=86400)
def get_surahs():
    try:
        r = requests.get("http://api.alquran.cloud/v1/surah", timeout=10)
        r.raise_for_status()
        return r.json().get("data", [])
    except Exception as e:
        logging.error(f"get_surahs error: {e}")
        return []

@st.cache_data(ttl=86400)
def get_verses(surah_num, translation_code="en.asad"):
    """
    Récupère éditions pour une sourate et retourne la liste d'éditions (le format varie parfois).
    On renverra une liste d'éditions (chaque édition possède "edition" et "ayahs").
    """
    try:
        url = f"http://api.alquran.cloud/v1/surah/{surah_num}/editions/quran-simple,{translation_code}"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json().get("data", [])
        return data
    except Exception as e:
        logging.error(f"get_verses error: {e}")
        return []

@st.cache_data(ttl=86400)
def get_audio_ayah(surah_num, ayah_num, reciter="ar.alafasy"):
    """Retourne l'URL audio si disponible."""
    try:
        r = requests.get(f"http://api.alquran.cloud/v1/ayah/{surah_num}:{ayah_num}/{reciter}", timeout=10)
        r.raise_for_status()
        d = r.json().get("data", {})
        # la clé audio peut être dans data ou data['audio']
        if isinstance(d, dict):
            return d.get("audio") or d.get("Audio") or None
        return None
    except Exception as e:
        logging.warning(f"get_audio_ayah error: {e}")
        return None

# ---------------------------
# --- RECHERCHE / RERANKER ---
# ---------------------------
def search_tafsir(query_translated, top_k=10):
    """
    Recherche par nearest neighbours dans les embeddings du tafsir.
    query_translated est déjà la question traduite dans la langue des embeddings (ici SQ selon ton dataset).
    """
    if tafsir_index is None or len(tafsir_keys) == 0:
        return []

    try:
        query_embed = model.encode([query_translated], convert_to_tensor=False)
        distances, indices = tafsir_index.kneighbors(query_embed, n_neighbors=min(top_k, len(tafsir_keys)))
    except Exception as e:
        logging.error(f"search_tafsir error (kneighbors): {e}")
        return []

    results = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(tafsir_keys):
            continue
        key = tafsir_keys[idx]
        tafsir_value = tafsir_data.get(key, "")
        if isinstance(tafsir_value, dict):
            tafsir_text = tafsir_value.get('text') or tafsir_value.get('tafsir') or ""
        elif isinstance(tafsir_value, str):
            tafsir_text = tafsir_value
        else:
            tafsir_text = ""
        if tafsir_text and tafsir_text.strip():
            results.append({"key": key, "tafsir": tafsir_text.strip()})
    return results

def rerank_results(question, passages):
    """Reranker avec CrossEncoder et normaliser les scores."""
    if not passages:
        return []
    try:
        pairs = [(question, p) for p in passages]
        scores = reranker.predict(pairs)
        # normalisation 0..1
        min_s, max_s = float(np.min(scores)), float(np.max(scores))
        if max_s - min_s > 1e-9:
            norm_scores = (scores - min_s) / (max_s - min_s)
        else:
            norm_scores = np.ones_like(scores)
        ranked = sorted(zip(passages, norm_scores), key=lambda x: x[1], reverse=True)
        return ranked
    except Exception as e:
        logging.error(f"rerank_results error: {e}")
        # fallback : retourner passages sans score
        return [(p, 0.0) for p in passages]

# ---------------------------
# --- PIPELINE QA MULTILINGUE ---
# ---------------------------
def qa_multilang(user_question):
    """Pipeline : détecte la langue, traduit si besoin, recherche, rerank, retraduit et reformule."""
    lang_detected = detect_language_safe(user_question)

    if lang_detected == "unknown":
        return "Impossible de détecter la langue, veuillez reformuler.", lang_detected

    # -> traduire dans la langue de ton tafsir embeddings (ici je garde la logique d'origine : sq)
    target_embedding_lang = "sq"
    if lang_detected != target_embedding_lang:
        try:
            question_translated = GoogleTranslator(source='auto', target=target_embedding_lang).translate(user_question)
        except Exception as e:
            logging.warning(f"Traduction vers {target_embedding_lang} échouée : {e}")
            return "Erreur lors de la traduction de la question.", lang_detected
    else:
        question_translated = user_question

    # Recherche
    results = search_tafsir(question_translated, top_k=10)
    passages = [r['tafsir'] for r in results if isinstance(r['tafsir'], str) and r['tafsir'].strip()]
    if not passages:
        return "Aucune réponse trouvée.", lang_detected

    # Rerank (on reranker le texte dans la langue des passages)
    ranked = rerank_results(question_translated, passages)
    best_passages = [p[0] for p in ranked[:3]]  # on prend top 3
    combined = " ".join(best_passages)

    # Retraduire la réponse dans la langue originale de l'utilisateur si nécessaire
    if lang_detected != target_embedding_lang:
        try:
            answer_translated = GoogleTranslator(source=target_embedding_lang, target=lang_detected).translate(combined)
        except Exception as e:
            logging.warning(f"Retraduction échouée : {e}")
            answer_translated = combined
    else:
        answer_translated = combined

    # Reformule légèrement pour fluidité
    answer_refined = reformulate_text(answer_translated, lang_detected)
    return answer_refined, lang_detected

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

# 3. Récupération versets (gestion robuste de l'ordre des éditions)
verses_data = get_verses(num_surah, code_trad)
arabic_edition = None
trans_edition = None
if isinstance(verses_data, list) and len(verses_data) > 0:
    # edition identifier pour quran-simple (arabe simple), et pour code_trad
    arabic_edition = next((d for d in verses_data if d.get("edition", {}).get("identifier") in ["quran-simple", "quran-simple-qdc", "quran-simple-text"]), None)
    trans_edition = next((d for d in verses_data if d.get("edition", {}).get("identifier") == code_trad), None)
    # fallback : si non trouvé, utiliser positions 0/1 si présent
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

# 5. Affichage verset arabe AVEC toggle + zoom (SEULEMENT ICI)
render_verset(verset_ar, weight=700)

# 6. Traduction (affichée normalement, sans zoom/diacritiques appliqués)
st.subheader(f"🌍 Traduction ({traduction_choisie})")
st.write(f"*{verset_trad}*")

# 7. Audio verset
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

# 8. Affichage Tafsir (AFFICHÉ SANS toggle/zoom)
cle_verset = f"{num_surah}:{verset_num}"
tafsir_entry = tafsir_data.get(cle_verset, {})
tafsir_raw = tafsir_entry.get("text", "") if isinstance(tafsir_entry, dict) else str(tafsir_entry or "")
tafsir_clean = nettoyer_html(tafsir_raw)

lang_tafsir = st.selectbox("🌐 Langue traduction du tafsir :", ["fr", "en", "ar", "es", "wo"])
traduction_tafsir = safe_translate(tafsir_clean, lang_tafsir) if tafsir_clean else ""

st.subheader("📜 Tafsir")
st.write(traduction_tafsir)

# Audio tafsir gTTS (en mémoire pour éviter problèmes de fichiers temporaires)
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

# 9. Q&A multilingue avec tafsir complet
st.markdown("---")
st.subheader("❓ Posez une question au sujet du Coran (toutes langues)")

if "history" not in st.session_state:
    st.session_state.history = []

# Affichage de l'historique (plus lisible)
for chat in st.session_state.history:
    st.markdown(f"🧑‍💻 **Vous :** {chat['question']}")
    st.info(chat['answer'])

# Champ de saisie
user_q = st.text_input("💬 Posez votre question :")

# Bouton envoyer
if st.button("Envoyer"):
    if user_q.strip():
        with st.spinner("Recherche de la réponse..."):
            answer, lang_used = qa_multilang(user_q)
        st.session_state.history.append({"question": user_q, "answer": answer})
        # rerun pour afficher l'historique mis à jour
        st.experimental_rerun()
    else:
        st.warning("Veuillez entrer une question.")

# Bouton pour effacer la conversation
if st.button("🗑 Effacer la conversation"):
    st.session_state.history = []
    st.success("Conversation réinitialisée.")
