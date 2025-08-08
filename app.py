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
from transformers import pipeline
from gtts import gTTS

DetectorFactory.seed = 0  # Pour stabilité détection langue
# Fonction de détection de langue
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# Chargement des ressources encodées
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
    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    return model
# Ensuite, appelle-la pour charger le modèle
model = load_sentence_model()

@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

qa_model = load_qa_model()

# Fonction détection langue
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

# Fonction traduction vers albanais
def translate_to_albanian(text, src_lang):
    if src_lang == "sq":
        return text
    try:
        return GoogleTranslator(source='auto', target='sq').translate(text)
    except Exception as e:
        st.warning(f"Erreur traduction : {e}")
        return text

# Construire l’index
nn_model = NearestNeighbors(n_neighbors=3, metric='cosine')
nn_model.fit(tafsir_embeddings_np)

def nettoyer_html(texte):
    return re.sub(r'<[^>]+>', '', texte)

def supprimer_blocs_balises_bi(texte):
    return re.sub(r'<b><i>.*?</i></b>', '', texte, flags=re.DOTALL)

def traduire_texte(texte, langue_cible):
    try:
        return GoogleTranslator(source='auto', target=langue_cible).translate(texte)
    except Exception as e:
        return f"Erreur de traduction : {e}"

@st.cache_data(ttl=86400)
def obtenir_la_liste_des_surahs():
    url = "http://api.alquran.cloud/v1/surah"
    return requests.get(url).json()["data"]

def obtenir_vers(surah_number, translation_code="en.asad"):
    url = f"http://api.alquran.cloud/v1/surah/{surah_number}/editions/quran-simple,{translation_code}"
    return requests.get(url).json()["data"]

@st.cache_data
def obtenir_audio_verset(surah_num, ayah_num, recitateur="ar.alafasy"):
    try:
        url = f"http://api.alquran.cloud/v1/ayah/{surah_num}:{ayah_num}/{recitateur}"
        data = requests.get(url).json()
        return data["data"]["audio"]  # lien mp3
    except Exception as e:
        st.warning(f"Impossible de récupérer l'audio : {e}")
        return None

# Interface
st.set_page_config(page_title="Assistant Coran IA", layout="centered")
st.title("📖 Assistant Coran avec IA (optimisé)")

sourates = obtenir_la_liste_des_surahs()
sourate_noms = [f"{s['number']}. {s['englishName']} ({s['name']})" for s in sourates]
choix_sourate = st.selectbox("📚 Choisissez une sourate :", sourate_noms)
num_sourate = int(choix_sourate.split(".")[0])

traduction_options = {
    "🇫🇷 Français (Hamidullah)": "fr.hamidullah",
    "🇬🇧 Anglais (Muhammad Asad)": "en.asad",
    "🇮🇩 Indonésien": "id.indonesian",
    "🇹🇷 Turc": "tr.translator",
    "🇺🇿 Ouzbek": "uz.sodik"

}
traduction_label = st.selectbox("🌐 Traduction :", list(traduction_options.keys()))
code_traduction = traduction_options[traduction_label]

versets_data = obtenir_vers(num_sourate, code_traduction)
versets_ar = versets_data[0]["ayahs"]
versets_trad = versets_data[1]["ayahs"]

verset_num = st.number_input("📌 Choisir le verset :", 1, len(versets_ar), 1)
verset_sel = versets_ar[verset_num - 1]
verset_trad = versets_trad[verset_num - 1]

# Option de zoom
zoom = st.slider("🔍 Zoom du verset", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

# Affichage avec zoom
st.subheader("🕋 Verset en arabe")
st.markdown(
    f"<p style='font-size:{zoom}em; direction:rtl; text-align:right;'>"
    f"**{verset_sel['text']}**</p>",
    unsafe_allow_html=True
)

st.subheader(f"🌍 Traduction ({traduction_label})")
st.write(f"*{verset_trad['text']}*")

# Audio avec choix du récitateur
recitateurs = {
    "🎙 Mishary Rashid Alafasy": "ar.alafasy",
    "🎙 Abdul Basit": "ar.abdulbasitmurattal",
    "🎙 Saad Al-Ghamdi": "ar.saoodshuraim"
}
choix_recitateur = st.selectbox("Choisissez un récitateur :", list(recitateurs.keys()))
code_recitateur = recitateurs[choix_recitateur]

url_audio = obtenir_audio_verset(num_sourate, verset_num, recitateur=code_recitateur)
st.subheader("🎧 Écouter le verset")
if url_audio:
    st.audio(url_audio, format="audio/mp3")
else:
    st.info("Audio non disponible pour ce verset.")

cle_exacte = f"{num_sourate}:{verset_num}"
tafsir = tafsir_data.get(cle_exacte, {}).get("text", "")
tafsir_sans_bi = supprimer_blocs_balises_bi(tafsir)
tafsir_clean = nettoyer_html(tafsir_sans_bi)

# 🔥 AFFICHAGE UNIQUEMENT DE LA TRADUCTION
st.subheader("🌐 TAFSIR DU VERSET")
langue_trad = st.selectbox("Choisir la langue de traduction :", ["fr", "en", "ar", "es", "wolof"])
traduction_tafsir = traduire_texte(tafsir_clean, langue_trad)
st.markdown(f"**Traduction du tafsir en {langue_trad.upper()} :**")
st.write(traduction_tafsir)

# Audio du tafsir traduit
if traduction_tafsir:
    try:
        tts = gTTS(traduction_tafsir, lang=langue_trad)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        st.subheader("🔊 Écouter le tafsir traduit")
        st.audio(temp_file.name, format="audio/mp3")
    except Exception as e:
        st.warning(f"Audio non disponible : {e}")

# ----------------- Q&A -----------------
st.markdown("---")
st.subheader("❓ Pose ta question (arabe/français/anglais)")

question = st.text_input("Entrez votre question :")
if question:
    question_clean = question.strip()
    langue_question = detect_language(question_clean)
    st.write(f"Langue détectée : {langue_question}")

    # Traduction question vers albanais (langue du tafsir)
    question_albanais = translate_text(question_clean, langue_question, "sq")
    st.write(f"Question traduite en albanais : {question_albanais}")

    # Recherche des passages proches dans le tafsir albanais
    q_embed = sentence_model.encode([question_albanais])
    q_embed = np.array(q_embed)

    distances, indices = nn_model.kneighbors(q_embed, n_neighbors=5)

    candidats = []
    for idx in indices[0]:
        key = tafsir_keys[idx]
        contexte_brut = tafsir_data.get(key, {}).get("text", "")
        contexte = nettoyer_html(contexte_brut)

        if not contexte:
            st.warning(f"Contexte vide pour la clé {key}, passage ignoré.")
            continue

        try:
            result = qa_model(question=question_albanais, context=contexte)
            candidats.append((result["answer"], result["score"], key))
        except Exception as e:
            st.error(f"Erreur QA sur contexte {key} : {e}")
            continue

    candidats = sorted(candidats, key=lambda x: x[1], reverse=True)

    if candidats:
        st.markdown("### Réponses proposées :")
        for i, (rep, score, key) in enumerate(candidats, start=1):
            # Traduire réponse QA vers la langue de la question
            rep_trad = translate_text(rep, "sq", langue_question)
            st.markdown(f"**Réponse {i}** (score: {score:.2f}, source: {key})")
            st.write(rep_trad)
    else:
        st.warning("Aucune réponse trouvée pour cette question.")

