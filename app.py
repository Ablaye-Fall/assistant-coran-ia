# assistant-coran-ia/app.py
import streamlit as st
import requests
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer

# === Chargement modèle sémantique ===
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# Charger le modèle léger
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Chargement du fichier tafsir ===
with open("tafsir_fr_complet.json", "r", encoding="utf-8") as f:
    tafsir_data = json.load(f)
# Préparer les clés et textes
tafsir_keys = list(tafsir_data.keys())
tafsir_texts = list(tafsir_data.values())

# Encoder tous les tafsir (à faire une seule fois)
tafsir_embeddings = model.encode(tafsir_texts, convert_to_tensor=True)

def trouver_tafsir_semantique(texte_verset):
    """
    Trouve le tafsir le plus proche sémantiquement du texte du verset donné.
    """
    # Encoder la requête
    query_embedding = model.encode(texte_verset, convert_to_tensor=True)
    
    # Calculer la similarité
    scores = util.pytorch_cos_sim(query_embedding, tafsir_embeddings)[0]
    
    # Trouver le meilleur score
    best_idx = scores.argmax()
    meilleur_tafsir = tafsir_texts[best_idx]
    cle_associee = tafsir_keys[best_idx]
    
    return cle_associee, meilleur_tafsir
    
def traduire_texte(texte, langue_cible):
    """
    Traduit le texte donné vers la langue cible (ex: 'en', 'ar', 'fr', 'wolof')
    """
    try:
        traduction = GoogleTranslator(source='auto', target=langue_cible).translate(texte)
        return traduction
    except Exception as e:
        return f"Erreur de traduction : {e}"    

# === Construction index sémantique FAISS ===
tafsir_texts = []
tafsir_keys = []
for surah, verses in tafsir_data.items():
    for ayah, text in verses.items():
        tafsir_texts.append(text)
        tafsir_keys.append((surah, ayah))

embeddings = model.encode(tafsir_texts)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# === Fonctions API AlQuran.cloud ===
def get_surah_list():
    url = "http://api.alquran.cloud/v1/surah"
    response = requests.get(url)
    return response.json()["data"]

def get_verses(surah_number, translation_code="en.asad"):
    url = f"http://api.alquran.cloud/v1/surah/{surah_number}/editions/quran-simple,{translation_code}"
    response = requests.get(url)
    return response.json()["data"]

# === Interface Streamlit ===
st.set_page_config(page_title="Assistant Coran IA", layout="centered")
st.title("📖 Assistant Coran avec NLP")

# Sélection de la sourate
surahs = get_surah_list()
surah_names = [f"{s['number']}. {s['englishName']} ({s['name']})" for s in surahs]
surah_choice = st.selectbox("📚 Choisissez une sourate :", surah_names)
surah_number = int(surah_choice.split('.')[0])

# Sélection de la langue
translation_options = {
    "🇫🇷 Français (Hamidullah)": "fr.hamidullah",
    "🇬🇧 English (Muhammad Asad)": "en.asad",
    "🇮🇩 Indonésien": "id.indonesian",
    "🇹🇷 Turc": "tr.translator",
    "🇺🇿 Ouzbek": "uz.sodik"
}
translation_label = st.selectbox("🌐 Choisir une langue de traduction :", list(translation_options.keys()))
translation_code = translation_options[translation_label]

# Récupération des versets
verses_data = get_verses(surah_number, translation_code)
versets = verses_data[0]['ayahs']
traductions = verses_data[1]['ayahs']

# Sélection du verset
verse_index = st.number_input("📌 Choisir le numéro du verset :", min_value=1, max_value=len(versets), value=1)
selected_verse = versets[verse_index - 1]
translated_verse = traductions[verse_index - 1]

# Affichage
st.markdown("### 🕋 Verset en arabe")
st.markdown(f"**{selected_verse['text']}**")

st.markdown(f"### 🌍 Traduction ({translation_label})")
st.markdown(f"*{translated_verse['text']}*")

# Tafsir classique (local)
st.markdown("### 📖 Tafsir classique (extrait)")
tafsir_key = f"{int(surah_number)}:{int(verse_index)}"
tafsir_entry = tafsir_data.get(tafsir_key)
if tafsir_entry:
    st.markdown(f"*Source : {tafsir_entry.get('source', 'non spécifiée')}*")
    st.write(tafsir_entry.get("tafsir", "Contenu non disponible."))
else:
    st.warning("⚠️ Aucun tafsir trouvé pour ce verset.")

# Recherche sémantique
st.markdown("---")
st.markdown("## 🔍 Recherche sémantique dans le Tafsir")
query = st.text_input("Entrez un mot-clé ou une question :")
if query:
    query_vector = model.encode([query])
    D, I = index.search(np.array(query_vector), k=3)
    for idx in I[0]:
        surah, ayah = tafsir_keys[idx]
        st.markdown(f"**Sourate {surah}, Verset {ayah}**")
        st.write(tafsir_data[surah][ayah])

# Question-Réponse (placeholder à compléter avec un modèle local ou GPT)
st.markdown("---")
st.markdown("## ❓ Posez une question sur un verset ou tafsir")
question = st.text_input("Votre question :")
if question:
    st.info("🔧 Fonction de réponse à la question à intégrer avec modèle local ou GPT.")


