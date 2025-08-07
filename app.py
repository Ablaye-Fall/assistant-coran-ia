import streamlit as st
import json
import numpy as np
import requests
import re
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors

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

# Charger le modèle uniquement pour Q&A
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# Construire l’index
nn_model = NearestNeighbors(n_neighbors=3, metric='cosine')
nn_model.fit(tafsir_embeddings_np)

def nettoyer_html(texte):
    return re.sub(r'<[^>]+>', '', texte)

def traduire_texte(texte, langue_cible):
    try:
        return GoogleTranslator(source='auto', target=langue_cible).translate(texte)
    except Exception as e:
        return f"Erreur traduction : {e}"

@st.cache_data(ttl=86400)
def obtenir_la_liste_des_surahs():
    url = "http://api.alquran.cloud/v1/surah"
    return requests.get(url).json()["data"]

def obtenir_vers(surah_number, translation_code="en.asad"):
    url = f"http://api.alquran.cloud/v1/surah/{surah_number}/editions/quran-simple,{translation_code}"
    return requests.get(url).json()["data"]

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
}
traduction_label = st.selectbox("🌐 Traduction :", list(traduction_options.keys()))
code_traduction = traduction_options[traduction_label]

versets_data = obtenir_vers(num_sourate, code_traduction)
versets_ar = versets_data[0]["ayahs"]
versets_trad = versets_data[1]["ayahs"]

verset_num = st.number_input("📌 Choisir le verset :", 1, len(versets_ar), 1)
verset_sel = versets_ar[verset_num - 1]
verset_trad = versets_trad[verset_num - 1]

st.subheader("🕋 Verset en arabe")
st.write(f"**{verset_sel['text']}**")

st.subheader(f"🌍 Traduction ({traduction_label})")
st.write(f"*{verset_trad['text']}*")

cle_exacte = f"{num_sourate}:{verset_num}"
tafsir = tafsir_data.get(cle_exacte, {}).get("text", None)

if tafsir:
    tafsir_clean = nettoyer_html(tafsir)
    st.subheader("📖 Tafsir du verset")
    st.write(tafsir_clean)

    langue_trad = st.selectbox("🌐 Traduire le tafsir en :", ["fr", "en", "ar", "es"])
    trad = traduire_texte(tafsir_clean, langue_trad)
    st.markdown(f"**Traduction du tafsir en {langue_trad.upper()} :**")
    st.write(trad)
else:
    st.warning("❌ Aucun tafsir trouvé pour ce verset.")

# Q&A locale
st.markdown("---")
st.subheader("❓ Question sur un verset ou tafsir")

question = st.text_input("Entrez votre question :")
if question:
    query_embed = model.encode([question], convert_to_numpy=True)
    distances, indices = nn_model.kneighbors(query_embed)
    for idx in indices[0]:
        texte = nettoyer_html(tafsir_data[tafsir_keys[idx]]["text"])
        st.markdown(f"**📌 Résultat {tafsir_keys[idx]}**")
        st.write(texte)
