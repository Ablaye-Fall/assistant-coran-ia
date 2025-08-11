import json
import numpy as np
import streamlit as st
import requests
import re
import tempfile
from sentence_transformers import SentenceTransformer, CrossEncoder
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
from sklearn.neighbors import NearestNeighbors
from gtts import gTTS

DetectorFactory.seed = 0  # stabilité détection langue

# --- UTILITAIRES ---
def nettoyer_html(texte: str) -> str:
    if not isinstance(texte, str):
        texte = str(texte)
    texte = re.sub(r'<b><i>.*?</i></b>', '', texte, flags=re.DOTALL)
    texte = re.sub(r'<[^>]+>', '', texte)
    texte = re.sub(r'\s+', ' ', texte)
    return texte.strip()

def safe_detect(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"

def translate_text(text: str, target: str, src: str = "auto") -> str:
    try:
        return GoogleTranslator(source=src, target=target).translate(text)
    except Exception:
        return text

def reformulate_local(text: str, target_lang: str) -> str:
    """
    Fallback local "reformulation" using double translation to smooth style.
    """
    try:
        en = GoogleTranslator(source='auto', target='en').translate(text)
        return GoogleTranslator(source='en', target=target_lang).translate(en)
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

@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def load_reranker_model():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

tafsir_data, tafsir_keys, tafsir_embeddings, tafsir_index = load_tafsir_resources()
model = load_sentence_model()
reranker = load_reranker_model()

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

# --- RECHERCHE VECTORIELLE ---
def search_tafsir(query_albanian, top_k=10):
    query_embed = model.encode([query_albanian], convert_to_tensor=False)
    distances, indices = tafsir_index.kneighbors(query_embed, n_neighbors=top_k)
    results = []
    for idx in indices[0]:
        key = tafsir_keys[idx]
        tafsir_value = tafsir_data.get(key, "")
        if isinstance(tafsir_value, dict):
            tafsir_text = tafsir_value.get('text') or tafsir_value.get('tafsir') or ""
        elif isinstance(tafsir_value, str):
            tafsir_text = tafsir_value
        else:
            tafsir_text = ""
        if tafsir_text.strip():
            results.append({"key": key, "tafsir": tafsir_text.strip()})
    return results

# --- RERANKER ---
def rerank_results(question: str, passages: list):
    """
    passages: list[str]
    retourne: list[str] triés par score décroissant
    """
    if not passages:
        return []
    pairs = [(question, p) for p in passages]
    try:
        scores = reranker.predict(pairs)
    except Exception:
        # si reranker échoue, renvoie passages originaux
        return passages
    ranked = sorted(zip(passages, scores), key=lambda x: x[1], reverse=True)
    return [p for p, s in ranked]
# -------------------------
# FILTRAGE / CLEAN
# -------------------------
def filter_passages(passages, min_len=50, max_len=1500):
    out = []
    for p in passages:
        p_clean = nettoyer_html(p)
        # supprimer courts snippets non informatifs
        if len(p_clean) < min_len:
            continue
        # couper très long en paragraphes et garder premier paragraphe utile
        if len(p_clean) > max_len:
            parts = re.split(r'\n{2,}|\. ', p_clean)
            # garder paragraphe le plus long mais < max_len
            candidates = [s.strip() for s in parts if 30 < len(s.strip()) < max_len]
            p_clean = candidates[0] if candidates else p_clean[:max_len]
        out.append(p_clean.strip())
    return out

# --- PIPELINE QA MULTILINGUE ---
def qa_pipeline(user_question: str, target_lang: str = "fr"):
        # 1. détection langue
    lang = safe_detect(user_question)
    # 2. traduction en albanais si ton tafsir est en albanais (adaptable)
    # ici on suppose tafsir_data est en albanais (sq), si non adapter.
    if lang != "sq":
        question_for_search = translate_text(user_question, "sq", src=lang)
    else:
        question_for_search = user_question

    # 3. recherche initiale (top_k)
    initial = search_tafsir(question_for_search, top_k=10)
    if not initial:
        return "Aucune réponse trouvée dans la base tafsir."

    # collect passages
    passages = [r['tafsir'] for r in initial if isinstance(r.get('tafsir', ""), str)]

    # 4. rerank (utilise question_for_search - en albanais)
    ranked = rerank_results(question_for_search, passages)

    # 5. filtrage anti-bruit
    filtered = filter_passages(ranked, min_len=60, max_len=1500)
    if not filtered:
        return "Aucune réponse utile après filtrage."

    # 6. prendre top passage(s)
    top_passage = filtered[0]
    # si tu veux, combiner top 2 ou 3 :
    # top_passage = " ".join(filtered[:3])

    # 8. final polishing (reformulation fluide si besoin)
    polished = reformulate_local(summarized, target_lang)

    # 9. retourner
    return polished


    # Traduction vers langue originale si nécessaire
    if lang_detected != "sq":
        try:
            answer_translated = GoogleTranslator(source='sq', target=lang_detected).translate(summarized)
        except Exception:
            answer_translated = summarized
    else:
        answer_translated = summarized

    return answer_translated, lang_detected

# --- INTERFACE STREAMLIT ---
st.set_page_config(page_title="Assistant Coran IA", page_icon="📖", layout="centered")
st.title("📖 Assistant Coran IA - Lecture, Tafsir & Q&A Multilingue")

# Choix sourate
surahs = get_surahs()
surah_names = [f"{s['number']}. {s['englishName']} ({s['name']})" for s in surahs]
choix_surah = st.selectbox("📚 Choisissez une sourate :", surah_names)
num_surah = int(choix_surah.split(".")[0]) if choix_surah else 1

# Choix traduction
translations = {
    "🇫🇷 Français (Hamidullah)": "fr.hamidullah",
    "🇬🇧 Anglais (Muhammad Asad)": "en.asad",
    "🇮🇩 Indonésien": "id.indonesian",
    "🇹🇷 Turc": "tr.translator",
    "🇺🇿 Ouzbek": "uz.sodik"
}
traduction_choisie = st.selectbox("🌐 Choisissez une traduction :", list(translations.keys()))
code_trad = translations.get(traduction_choisie, "en.asad")

# Récupération versets
verses_data = get_verses(num_surah, code_trad)
verses_ar = verses_data[0]["ayahs"] if len(verses_data) > 0 else []
verses_trad = verses_data[1]["ayahs"] if len(verses_data) > 1 else []

# Choix verset
max_verset = len(verses_ar) if verses_ar else 1
verset_num = st.number_input("📌 Choisissez un verset :", min_value=1, max_value=max_verset, value=1)

verset_ar = verses_ar[verset_num - 1]["text"] if verses_ar else ""
verset_trad = verses_trad[verset_num - 1]["text"] if verses_trad else ""

# Affichage verset + traduction avec zoom
zoom = st.slider("🔍 Zoom texte arabe", min_value=1.0, max_value=3.0, value=1.5, step=0.1)
st.markdown(f"<p style='font-size:{zoom}em; direction:rtl; text-align:right; font-weight:bold;'>{verset_ar}</p>", unsafe_allow_html=True)
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
    st.audio(audio_url, format="audio/mp3")

# Affichage Tafsir
cle_verset = f"{num_surah}:{verset_num}"
tafsir_entry = tafsir_data.get(cle_verset, {})
tafsir_raw = tafsir_entry.get("text", "") if isinstance(tafsir_entry, dict) else str(tafsir_entry)
tafsir_clean = nettoyer_html(tafsir_raw)

lang_tafsir = st.selectbox("🌐 Langue traduction du tafsir :", ["fr", "en", "ar", "es", "wo"])
traduction_tafsir = GoogleTranslator(source='auto', target=lang_tafsir).translate(tafsir_clean) if tafsir_clean else ""

st.subheader("📜 Tafsir")
st.write(traduction_tafsir)

# Audio tafsir gTTS
tts_langs = ["fr", "en", "ar", "es"]
if lang_tafsir in tts_langs and traduction_tafsir:
    try:
        tts = gTTS(traduction_tafsir, lang=lang_tafsir)
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(tmp_file.name)
        st.audio(tmp_file.name, format="audio/mp3")
    except Exception as e:
        st.warning(f"Audio tafsir non disponible : {e}")

# Q&A multilingue avec tafsir complet
st.markdown("---")
st.subheader("❓ Posez une question au sujet du Coran (toutes langues)")

if "history" not in st.session_state:
    st.session_state.history = []

# Affichage historique
for chat in st.session_state.history:
    st.markdown(f"**🧑‍💻 Vous :** {chat['question']}")
    st.markdown(f"**🤖 Assistant :** {chat['answer']}")

# Champ saisie question
user_q = st.text_input("💬 Posez votre question :")

if st.button("Envoyer"):
    if user_q.strip():
        with st.spinner("Recherche de la réponse..."):
            answer, lang_used = qa_pipeline(user_q)
        st.session_state.history.append({"question": user_q, "answer": answer})
        st.rerun()
    else:
        st.warning("Veuillez entrer une question.")

if st.button("🗑 Effacer la conversation"):
    st.session_state.history = []
    st.success("Conversation réinitialisée.")
