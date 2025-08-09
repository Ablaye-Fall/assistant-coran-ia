import streamlit as st
import json
import numpy as np
import requests
import re
import tempfile
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
from langdetect import detect, DetectorFactory
from gtts import gTTS

DetectorFactory.seed = 0  # stabilité détection langue

# --- Fonctions utilitaires ---
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def nettoyer_html(texte):
    """Supprime balises HTML + nettoyage BI"""
    texte = re.sub(r'<b><i>.*?</i></b>', '', texte, flags=re.DOTALL)
    return re.sub(r'<[^>]+>', '', texte)

def traduire_texte(texte, langue_cible):
    if not texte.strip():
        return ""
    try:
        return GoogleTranslator(source='auto', target=langue_cible).translate(texte)
    except Exception as e:
        return f"Erreur de traduction : {e}"

def translate_text(text, src_lang, tgt_lang):
    if src_lang == tgt_lang:
        return text
    try:
        return GoogleTranslator(source=src_lang, target=tgt_lang).translate(text)
    except Exception as e:
        st.warning(f"Erreur traduction : {e}")
        return text

# --- Chargement ressources ---
@st.cache_resource
def load_resources():
    with open("sq-saadi.json", "r", encoding="utf-8") as f:
        tafsir_data = json.load(f)
    with open("tafsir_keys.json", "r", encoding="utf-8") as f:
        tafsir_keys = json.load(f)
    embeddings = np.load("tafsir_embeddings.npy")
    return tafsir_data, tafsir_keys, embeddings

tafsir_data, tafsir_keys, tafsir_embeddings_np = load_resources()

@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

model = load_sentence_model()

# Index NearestNeighbors
nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
nn_model.fit(tafsir_embeddings_np)

# Recherche dans le tafsir
def search_tafsir(query_albanian, top_k=3):
    query_embed = model.encode(query_albanian, convert_to_tensor=False)
    query_embed = np.array(query_embed, dtype=np.float32).reshape(1, -1)  # Format correct pour kneighbors
    distances, indices = nn_model.kneighbors(query_embed, n_neighbors=top_k)
    results = []
    for idx in indices[0]:
        tafsir_entry = tafsir_data.get(tafsir_keys[idx], {})
        tafsir_text = tafsir_entry.get("text", "") if isinstance(tafsir_entry, dict) else str(tafsir_entry)
        results.append({
            "key": tafsir_keys[idx],
            "tafsir": tafsir_text
        })
    return results

# Reformulation via traduction intermédiaire
def reformulate_text(text, target_lang):
    try:
        temp_en = GoogleTranslator(source='auto', target='en').translate(text)
        return GoogleTranslator(source='en', target=target_lang).translate(temp_en)
    except Exception:
        return text

# QA multilingue avec contexte
def qa_multilang(user_question, history):
    lang_detected = detect_language(user_question)
    full_question = " ".join([h["question"] for h in history]) + " " + user_question if history else user_question

    if lang_detected != "sq":
        question_albanian = GoogleTranslator(source='auto', target='sq').translate(full_question)
    else:
        question_albanian = full_question

    tafsir_results = search_tafsir(question_albanian, top_k=3)
    if not tafsir_results:
        return "Aucune réponse trouvée.", lang_detected

    combined_albanian = " ".join(str(res['tafsir']) for res in tafsir_results if res['tafsir'])

    if lang_detected != "sq":
        answer_translated = GoogleTranslator(source='sq', target=lang_detected).translate(combined_albanian)
    else:
        answer_translated = combined_albanian

    return reformulate_text(answer_translated, lang_detected), lang_detected

# --- API Quran ---
@st.cache_data(ttl=86400)
def obtenir_la_liste_des_surahs():
    try:
        return requests.get("http://api.alquran.cloud/v1/surah").json()["data"]
    except Exception as e:
        st.warning(f"Erreur récupération sourates : {e}")
        return []

def obtenir_vers(surah_number, translation_code="en.asad"):
    try:
        url = f"http://api.alquran.cloud/v1/surah/{surah_number}/editions/quran-simple,{translation_code}"
        return requests.get(url).json()["data"]
    except Exception as e:
        st.warning(f"Erreur récupération versets : {e}")
        return []

@st.cache_data
def obtenir_audio_verset(surah_num, ayah_num, recitateur="ar.alafasy"):
    try:
        data = requests.get(f"http://api.alquran.cloud/v1/ayah/{surah_num}:{ayah_num}/{recitateur}").json()
        return data["data"]["audio"]
    except Exception as e:
        st.warning(f"Impossible de récupérer l'audio : {e}")
        return None

# --- Interface ---
st.set_page_config(page_title="Assistant Coran IA", layout="centered")
st.title("📖 Assistant Coran avec IA (optimisé)")

# Choix sourate
sourates = obtenir_la_liste_des_surahs()
sourate_noms = [f"{s['number']}. {s['englishName']} ({s['name']})" for s in sourates]
choix_sourate = st.selectbox("📚 Choisissez une sourate :", sourate_noms)
num_sourate = int(choix_sourate.split(".")[0]) if choix_sourate else 1

# Traductions
traduction_options = {
    "🇫🇷 Français (Hamidullah)": "fr.hamidullah",
    "🇬🇧 Anglais (Muhammad Asad)": "en.asad",
    "🇮🇩 Indonésien": "id.indonesian",
    "🇹🇷 Turc": "tr.translator",
    "🇺🇿 Ouzbek": "uz.sodik"
}
traduction_label = st.selectbox("🌐 Traduction :", list(traduction_options.keys()))
code_traduction = traduction_options.get(traduction_label, "en.asad")

versets_data = obtenir_vers(num_sourate, code_traduction)
versets_ar = versets_data[0]["ayahs"] if versets_data else []
versets_trad = versets_data[1]["ayahs"] if len(versets_data) > 1 else []

verset_num = st.number_input("📌 Choisir le verset :", 1, len(versets_ar) if versets_ar else 1, 1)
verset_sel = versets_ar[verset_num - 1] if versets_ar else {"text": ""}
verset_trad = versets_trad[verset_num - 1] if versets_trad else {"text": ""}

# Affichage verset
zoom = st.slider("🔍 Zoom du verset", 1.0, 3.0, 1.5, 0.1)
st.markdown(f"<p style='font-size:{zoom}em; direction:rtl; text-align:right; font-weight:bold;'>{verset_sel['text']}</p>", unsafe_allow_html=True)
st.subheader(f"🌍 Traduction ({traduction_label})")
st.write(f"*{verset_trad['text']}*")

# Audio
recitateurs = {
    "🎙 Mishary Rashid Alafasy": "ar.alafasy",
    "🎙 Abdul Basit": "ar.abdulbasitmurattal",
    "🎙 Saad Al-Ghamdi": "ar.saoodshuraim"
}
choix_recitateur = st.selectbox("Choisissez un récitateur :", list(recitateurs.keys()))
url_audio = obtenir_audio_verset(num_sourate, verset_num, recitateurs.get(choix_recitateur, "ar.alafasy"))
if url_audio:
    st.audio(url_audio, format="audio/mp3")

# Tafsir
cle_exacte = f"{num_sourate}:{verset_num}"
tafsir_entry = tafsir_data.get(cle_exacte, {})
tafsir_text_raw = tafsir_entry.get("text", "") if isinstance(tafsir_entry, dict) else str(tafsir_entry)
tafsir_clean = nettoyer_html(tafsir_text_raw)
langue_trad = st.selectbox("Langue traduction tafsir :", ["fr", "en", "ar", "es", "wo"])
traduction_tafsir = traduire_texte(tafsir_clean, langue_trad)
st.write(traduction_tafsir)

# Audio tafsir (si supporté par gTTS)
langues_support_tts = ["fr", "en", "ar", "es"]
if langue_trad in langues_support_tts and traduction_tafsir and not traduction_tafsir.startswith("Erreur"):
    try:
        tts = gTTS(traduction_tafsir, lang=langue_trad)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        st.audio(temp_file.name, format="audio/mp3")
    except Exception as e:
        st.warning(f"Audio non disponible : {e}")

# Q&A multilingue
st.subheader("❓ Pose ta question (n'importe quelle langue)")
if "history" not in st.session_state:
    st.session_state.history = []
for chat in st.session_state.history:
    st.markdown(f"**🧑‍💻 Vous :** {chat['question']}")
    st.markdown(f"**🤖 Assistant :** {chat['answer']}")

user_q = st.text_input("💬 Posez votre question :")
if st.button("Envoyer"):
    if user_q.strip():
        with st.spinner("Recherche..."):
            answer, lang_used = qa_multilang(user_q, st.session_state.history)
        st.session_state.history.append({"question": user_q, "answer": answer})
        st.rerun()
    else:
        st.warning("Veuillez entrer une question.")

if st.button("🗑 Effacer la conversation"):
    st.session_state.history.clear()
    st.success("Conversation réinitialisée.")
