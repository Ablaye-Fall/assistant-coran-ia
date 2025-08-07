import streamlit as st
import requests
import json
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from sklearn.neighbors import NearestNeighbors
import re

# === Mise en cache du modèle ===
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# === Chargement du fichier tafsir ===
try:
    with open("sq-saadi.json", "r", encoding="utf-8") as f:
        tafsir_data = json.load(f)
except Exception as e:
    st.error(f"❌ Erreur de chargement du fichier tafsir : {e}")
    st.stop()

# Extraction des textes depuis la clé "text"
textes_tafsir = []
tafsir_keys = []
for key, value in tafsir_data.items():
    if isinstance(value, dict):
        tafsir_texte = value.get("text", "").strip()
        if isinstance(tafsir_texte, str) and tafsir_texte:
            textes_tafsir.append(tafsir_texte)
            tafsir_keys.append(key)

if not textes_tafsir:
    st.error("❌ Aucun texte de tafsir valide trouvé dans sq-saadi.json.")
    st.stop()

# Encodage mis en cache
@st.cache_data
def get_encoded_tafsir(textes):
    embeddings = model.encode(textes, convert_to_tensor=True)
    return embeddings.cpu().numpy()

tafsir_embeddings_np = get_encoded_tafsir(textes_tafsir)

# Recherche sémantique
nn_model = NearestNeighbors(n_neighbors=3, metric='cosine')
nn_model.fit(tafsir_embeddings_np)

def nettoyer_html(texte):
    return re.sub(r'<[^>]+>', '', texte)

def traduire_texte(texte, langue_cible):
    try:
        return GoogleTranslator(source='auto', target=langue_cible).translate(texte)
    except Exception as e:
        return f"Erreur de traduction : {e}"

@st.cache_data(ttl=86400)
def obtenir_la_liste_des_surahs():
    url = "http://api.alquran.cloud/v1/surah"
    response = requests.get(url)
    return response.json()["data"]

def obtenir_vers(surah_number, translation_code="en.asad"):
    url = f"http://api.alquran.cloud/v1/surah/{surah_number}/editions/quran-simple,{translation_code}"
    response = requests.get(url)
    return response.json()["data"]

# === Interface utilisateur ===
st.set_page_config(page_title="Assistant Coran IA", layout="centered")
st.title("📖 Assistant Coran avec IA (Tafsir As-Saadi - sq-saadi.json)")

# Choix de la sourate
sourates = obtenir_la_liste_des_surahs()
sourate_noms = [f"{s['number']}. {s['englishName']} ({s['name']})" for s in sourates]
choix_sourate = st.selectbox("📚 Choisissez une sourate :", sourate_noms)
num_sourate = int(choix_sourate.split(".")[0])

# Choix langue de traduction
traduction_options = {
    "🇫🇷 Français (Hamidullah)": "fr.hamidullah",
    "🇬🇧 Anglais (Muhammad Asad)": "en.asad",
    "🇮🇩 Indonésien": "id.indonesian",
    "🇹🇷 Turc": "tr.translator",
    "🇺🇿 Ouzbek": "uz.sodik"
}
traduction_label = st.selectbox("🌐 Choisir une langue de traduction :", list(traduction_options.keys()))
code_traduction = traduction_options[traduction_label]

# Récupération des versets
versets_data = obtenir_vers(num_sourate, code_traduction)
versets_ar = versets_data[0]["ayahs"]
versets_trad = versets_data[1]["ayahs"]

# Choix du verset
verset_num = st.number_input("📌 Choisir le numéro du verset :", min_value=1, max_value=len(versets_ar), value=1)
verset_sel = versets_ar[verset_num - 1]
verset_trad = versets_trad[verset_num - 1]

# Affichage du verset
st.subheader("🕋 Verset en arabe")
st.write(f"**{verset_sel['text']}**")

st.subheader(f"🌍 Traduction ({traduction_label})")
st.write(f"*{verset_trad['text']}*")

# Affichage du tafsir exact par clé
cle_exacte = f"{num_sourate}:{verset_num}"
tafsir = tafsir_data.get(cle_exacte, {}).get("text", "❌ Aucun tafsir disponible pour ce verset.")
tafsir_clean = nettoyer_html(tafsir)

st.subheader("📖 Tafsir du verset")
st.write(tafsir_clean)

# Traduction du tafsir
langue_trad = st.selectbox("🌐 Traduire le tafsir en :", ["fr", "en", "ar", "es", "wolof"])
traduction_tafsir = traduire_texte(tafsir_clean, langue_trad)
st.markdown(f"**Traduction du tafsir en {langue_trad.upper()} :**")
st.write(traduction_tafsir)

# Bloc Q&A
st.markdown("---")
st.subheader("❓ Posez une question sur un verset ou tafsir")
question = st.text_input("Votre question :")

if question:
    st.info("🔍 Recherche de la réponse la plus proche dans le tafsir...")

    # Encodage de la question
    question_embedding = model.encode([question], convert_to_tensor=True).cpu().numpy()

    # Recherche des réponses les plus proches
    distances, indices = nn_model.kneighbors(question_embedding, n_neighbors=3)

    st.markdown("### 🧠 Réponses suggérées à partir du Tafsir As-Saadi :")

    for i, idx in enumerate(indices[0]):
        key = tafsir_keys[idx]
        texte = nettoyer_html(tafsir_data.get(key, {}).get("text", ""))
        st.markdown(f"**Résultat {i+1} — Verset {key} :**")
        st.write(texte)

        if langue_trad:
            traduction = traduire_texte(texte, langue_trad)
            st.markdown(f"🔁 *Traduction en {langue_trad.upper()}* :")
            st.write(traduction)
