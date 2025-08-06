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
    if isinstance(value, dict):
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
    ur
