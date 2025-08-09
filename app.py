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

# --- Fonctions utilitaires ---

def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def nettoyer_html(texte):
    return re.sub(r'<[^>]+>', '', texte)

def supprimer_blocs_balises_bi(texte):
    return re.sub(r'<b><i>.*?</i></b>', '', texte, flags=re.DOTALL)

def traduire_texte(texte, langue_cible):
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

# --- Chargement des ressources ---

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
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

model = load_sentence_model()

# Création index NearestNeighbors pour recherche dans embeddings
nn_model = NearestNeighbors(n_neighbors=5, metric='cosine')
nn_model.fit(tafsir_embeddings_np)

# Chargement modèle de reformulation (text2text generation)
@st.cache_resource
def load_gen_model():
    return pipeline("text2text-generation", model="google/mt5-small")

gen_model = load_gen_model()

# --- Fonctions QA améliorées ---

def search_tafsir(question_albanais, top_k=5):
    q_embed = model.encode([question_albanais], convert_to_tensor=False)
    distances, indices = nn_model.kneighbors(q_embed, n_neighbors=top_k)  # corrigé ici, enlever crochet supplémentaire
    results = []
    for idx in indices[0]:
        key = tafsir_keys[idx]
        tafsir_txt = tafsir_data.get(key, "")
        results.append({"key": key, "tafsir": tafsir_txt})
    return results

def qa_multilang(user_question, history):
    try:
        lang_detected = detect(user_question)
    except:
        lang_detected = "fr"  # fallback

    # Ajouter historique contexte
    if history:
        full_question = " ".join([h["question"] for h in history]) + " " + user_question
    else:
        full_question = user_question

    # Traduire en albanais si nécessaire
    if lang_detected != "sq":
        question_albanais = translate_text(full_question, lang_detected, "sq")
    else:
        question_albanais = full_question

    # Recherche passages pertinents dans tafsir
    tafsir_results = search_tafsir(question_albanais, top_k=5)
    if not tafsir_results:
        return "Aucune réponse trouvée.", lang_detected

    # Concaténer tafsir albanais
    combined_albanian = " ".join([r['tafsir'] for r in tafsir_results])

    # Nettoyage HTML et balises
    combined_albanian = nettoyer_html(supprimer_blocs_balises_bi(combined_albanian))

    # Traduction contexte vers langue d'origine
    if lang_detected != "sq":
        context_translated = translate_text(combined_albanian, "sq", lang_detected)
    else:
        context_translated = combined_albanian

    # Génération réponse reformulée naturelle
    prompt = (
        f"Réponds clairement à la question suivante en te basant sur le contexte donné, "
        f"en restant fidèle au texte.\n\nQuestion: {user_question}\nContexte: {context_translated}\nRéponse:"
    )
    generated = gen_model(prompt, max_length=250, num_return_sequences=1)
    refined_answer = generated[0].get("generated_text") or generated[0].get("text", "")

    return refined_answer, lang_detected

# --- Fonctions API Quran (affichage verset + audio) ---

@st.cache_data(ttl=86400)
def obtenir_la_liste_des_surahs():
    url = "http://api.alquran.cloud/v1/surah"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()["data"]
    except Exception as e:
        st.warning(f"Erreur lors de la récupération des sourates : {e}")
        return []

def obtenir_vers(surah_number, translation_code="en.asad"):
    url = f"http://api.alquran.cloud/v1/surah/{surah_number}/editions/quran-simple,{translation_code}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()["data"]
    except Exception as e:
        st.warning(f"Erreur lors de la récupération des versets : {e}")
        return []

@st.cache_data
def obtenir_audio_verset(surah_num, ayah_num, recitateur="ar.alafasy"):
    try:
        url = f"http://api.alquran.cloud/v1/ayah/{surah_num}:{ayah_num}/{recitateur}"
        data = requests.get(url).json()
        return data["data"]["audio"]  # lien mp3
    except Exception as e:
        st.warning(f"Impossible de récupérer l'audio : {e}")
        return None

# --- Interface Streamlit ---

st.set_page_config(page_title="Assistant Coran IA", layout="centered")
st.title("📖 Assistant Coran avec IA (optimisé)")

# Affichage sourates
sourates = obtenir_la_liste_des_surahs()
sourate_noms = [f"{s['number']}. {s['englishName']} ({s['name']})" for s in sourates]
choix_sourate = st.selectbox("📚 Choisissez une sourate :", sourate_noms)
num_sourate = int(choix_sourate.split(".")[0]) if choix_sourate else 1

# Choix traduction
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

# Zoom verset
zoom = st.slider("🔍 Zoom du verset", min_value=1.0, max_value=3.0, value=1.5, step=0.1)

st.subheader("🕋 Verset en arabe")
st.markdown(
    f"<p style='font-size:{zoom}em; direction:rtl; text-align:right; font-weight:bold;'>"
    f"{verset_sel['text']}</p>",
    unsafe_allow_html=True
)

st.subheader(f"🌍 Traduction ({traduction_label})")
st.write(f"*{verset_trad['text']}*")

# Audio verset
recitateurs = {
    "🎙 Mishary Rashid Alafasy": "ar.alafasy",
    "🎙 Abdul Basit": "ar.abdulbasitmurattal",
    "🎙 Saad Al-Ghamdi": "ar.saoodshuraim"
}
choix_recitateur = st.selectbox("Choisissez un récitateur :", list(recitateurs.keys()))
code_recitateur = recitateurs.get(choix_recitateur, "ar.alafasy")

url_audio = obtenir_audio_verset(num_sourate, verset_num, recitateur=code_recitateur)
st.subheader("🎧 Écouter le verset")
if url_audio:
    st.audio(url_audio, format="audio/mp3")
else:
    st.info("Audio non disponible pour ce verset.")

# Affichage tafsir + traduction
cle_exacte = f"{num_sourate}:{verset_num}"
tafsir = tafsir_data.get(cle_exacte, {}).get("text", "")
tafsir_sans_bi = supprimer_blocs_balises_bi(tafsir)
tafsir_clean = nettoyer_html(tafsir_sans_bi)

st.subheader("🌐 TAFSIR DU VERSET")
langue_trad = st.selectbox("Choisir la langue de traduction :", ["fr", "en", "ar", "es", "wolof"])
traduction_tafsir = traduire_texte(tafsir_clean, langue_trad)
st.markdown(f"**Traduction du tafsir en {langue_trad.upper()} :**")
st.write(traduction_tafsir)

# Audio tafsir traduit
if traduction_tafsir and not traduction_tafsir.startswith("Erreur de traduction"):
    try:
        tts = gTTS(traduction_tafsir, lang=langue_trad)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        st.subheader("🔊 Écouter le tafsir traduit")
        st.audio(temp_file.name, format="audio/mp3")
    except Exception as e:
        st.warning(f"Audio non disponible : {e}")

# ----------------- Q&A Multilingue amélioré -----------------
st.markdown("---")
st.subheader("❓ Pose ta question (n'importe quelle langue)")

if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Entrez votre question :")

if st.button("Envoyer"):
    if question and question.strip():
        with st.spinner("Recherche et génération de la réponse..."):
            answer, lang_used = qa_multilang(question.strip(), st.session_state.history)
        st.session_state.history.append({"question": question.strip(), "answer": answer})
        st.experimental_rerun()
    else:
        st.warning("Veuillez entrer une question.")

# Afficher historique
for chat in st.session_state.history:
    st.markdown(f"**🧑‍💻 Vous :** {chat['question']}")
    st.markdown(f"**🤖 Assistant :** {chat['answer']}")

# Synthèse vocale de la dernière réponse
if st.session_state.history:
    try:
        last_answer = st.session_state.history[-1]["answer"]
        last_lang = detect(last_answer)
        tts = gTTS(text=last_answer, lang=last_lang)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        st.audio(temp_file.name, format="audio/mp3")
    except Exception as e:
        st.warning(f"Lecture audio indisponible : {e}")

# Bouton réinitialiser conversation
if st.button("🗑 Effacer la conversation"):
    st.session_state.history = []
    st.success("Conversation réinitialisée.")
