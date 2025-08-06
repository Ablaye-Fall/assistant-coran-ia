import streamlit as st
import requests
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from deep_translator import GoogleTranslator
from sklearn.neighbors import NearestNeighbors

# === Chargement du modèle sémantique ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Chargement du fichier tafsir sq-saadi.json ===
try:
    with open("sq-saadi.json", "r", encoding="utf-8") as f:
        tafsir_data = json.load(f)
except Exception as e:
    st.error(f"❌ Erreur de chargement du fichier tafsir : {e}")
    st.stop()

# Extraction des textes de tafsir
textes_tafsir = []
tafsir_keys = []
for key, value in tafsir_data.items():
    tafsir_texte = value.get("tafsir", "").strip()
    if tafsir_texte:
        textes_tafsir.append(tafsir_texte)
        tafsir_keys.append(key)

if not textes_tafsir:
    st.error("❌ Aucun texte de tafsir valide trouvé dans sq-saadi.json.")
    st.stop()

# Encodage sémantique
tafsir_embeddings = model.encode(textes_tafsir, convert_to_tensor=True)
tafsir_embeddings_np = tafsir_embeddings.cpu().numpy()

if tafsir_embeddings_np.ndim != 2 or tafsir_embeddings_np.shape[0] == 0:
    st.error(f"❌ Encodage invalide des vecteurs. Dimensions : {tafsir_embeddings_np.shape}")
    st.stop()

# Modèle Nearest Neighbors
nn_model = NearestNeighbors(n_neighbors=3, metric='cosine')
nn_model.fit(tafsir_embeddings_np)

def trouver_tafsir_semantique(texte_verset):
    query_embedding = model.encode([texte_verset], convert_to_tensor=True).cpu().numpy()
    distances, indices = nn_model.kneighbors(query_embedding)
    best_idx = indices[0][0]
    return tafsir_keys[best_idx], textes_tafsir[best_idx]

def traduire_texte(texte, langue_cible):
    try:
        return GoogleTranslator(source='auto', target=langue_cible).translate(texte)
    except Exception as e:
        return f"Erreur de traduction : {e}"

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
st.title("📖 Assistant Coran avec IA (Tafsir Saadi - ES)")

# Choix sourate
sourates = obtenir_la_liste_des_surahs()
sourate_noms = [f"{s['number']}. {s['englishName']} ({s['name']})" for s in sourates]

with st.container():
    choix_sourate = st.selectbox("📚 Choisissez une sourate :", sourate_noms, key="choix_sourate")
    num_sourate = int(choix_sourate.split(".")[0])

    traduction_options = {
        "🇫🇷 Français (Hamidullah)": "fr.hamidullah",
        "🇬🇧 Anglais (Muhammad Asad)": "en.asad",
        "🇮🇩 Indonésien": "id.indonesian",
        "🇹🇷 Turc": "tr.translator",
        "🇺🇿 Ouzbek": "uz.sodik"
    }
    traduction_label = st.selectbox("🌐 Choisir une langue de traduction :", list(traduction_options.keys()), key="choix_langue")
    code_traduction = traduction_options[traduction_label]

# Mémo des versets
if (
    "versets_data" not in st.session_state
    or st.session_state.get("last_sourate") != num_sourate
    or st.session_state.get("last_langue") != code_traduction
):
    st.session_state.versets_data = obtenir_vers(num_sourate, code_traduction)
    st.session_state.last_sourate = num_sourate
    st.session_state.last_langue = code_traduction

versets_data = st.session_state.versets_data

# Affichage verset et tafsir
try:
    versets_ar = versets_data[0]["ayahs"]
    versets_trad = versets_data[1]["ayahs"]

    verset_num = st.number_input("📌 Choisir le numéro du verset :", min_value=1, max_value=len(versets_ar), value=1)
    verset_sel = versets_ar[verset_num - 1]
    verset_trad = versets_trad[verset_num - 1]

    st.subheader("🕋 Verset en arabe")
    st.write(f"**{verset_sel['text']}**")

    st.subheader(f"🌍 Traduction ({traduction_label})")
    st.write(f"*{verset_trad['text']}*")

    st.subheader("📖 Tafsir extrait (selon le sens du verset)")
    cle, tafsir = trouver_tafsir_semantique(verset_sel['text'])
    st.write(tafsir)

    langue_trad = st.selectbox("🌐 Traduire le tafsir en :", ["es", "fr", "en", "ar", "wolof"], key="langue_tafsir")
    traduction_tafsir = traduire_texte(tafsir, langue_trad)
    st.markdown(f"**Traduction du tafsir en {langue_trad.upper()} :**")
    st.write(traduction_tafsir)

except Exception as e:
    st.error(f"❌ Erreur lors de l'affichage du verset ou tafsir : {e}")
    st.stop()

# Recherche sémantique
st.markdown("---")
st.subheader("🔍 Recherche sémantique dans le Tafsir")
requete = st.text_input("Entrez un mot-clé ou une question :")
if requete:
    try:
        req_embed = model.encode([requete], convert_to_tensor=True).cpu().numpy()
        distances, indices = nn_model.kneighbors(req_embed)
        for idx in indices[0]:
            st.markdown(f"**🔹 {tafsir_keys[idx]}**")
            st.write(textes_tafsir[idx])
    except Exception as e:
        st.error(f"❌ Erreur dans la recherche : {e}")

# Bloc Question-Réponse (à venir)
st.markdown("---")
st.subheader("❓ Posez une question sur un verset ou tafsir")
question = st.text_input("Votre question :")
if question:
    st.info("🔧 Fonction de réponse à intégrer avec modèle local ou GPT.")
