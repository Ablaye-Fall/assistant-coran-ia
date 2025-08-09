
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
    except Exception:
        return "unknown"

def nettoyer_html(texte):
    texte = re.sub(r'<b><i>.*?</i></b>', '', texte, flags=re.DOTALL)
    texte = re.sub(r'<[^>]+>', '', texte)
    texte = texte.replace('\n', ' ').strip()
    return texte

def traduire_texte(texte, langue_cible):
    if not texte.strip():
        return ""
    try:
        return GoogleTranslator(source='auto', target=langue_cible).translate(texte)
    except Exception:
        return texte

def reformulate_text(text, target_lang):
    # version simplifiée ici
    return text

# --- Chargement ressources (une seule fois) ---
@st.cache_resource
def load_tafsir_data():
    with open("sq-saadi.json", "r", encoding="utf-8") as f:
        tafsir_data = json.load(f)
    with open("tafsir_keys.json", "r", encoding="utf-8") as f:
        tafsir_keys = json.load(f)
    embeddings = np.load("tafsir_embeddings.npy")
    return tafsir_data, tafsir_keys, embeddings

@st.cache_resource
def load_sentence_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource
def build_nn_index(embeddings):
    nn = NearestNeighbors(n_neighbors=5, metric='cosine')
    nn.fit(embeddings)
    return nn

# Charger une seule fois en mémoire
tafsir_data, tafsir_keys, tafsir_embeddings_np = load_tafsir_data()
model = load_sentence_model()
nn_model = build_nn_index(tafsir_embeddings_np)

# Recherche dans le tafsir
def search_tafsir(query_albanian, top_k=3):
    query_embed = model.encode(query_albanian, convert_to_tensor=False)
    query_embed = np.array(query_embed, dtype=np.float32).reshape(1, -1)
    distances, indices = nn_model.kneighbors(query_embed, n_neighbors=top_k)
    results = []
    for idx in indices[0]:
        if idx < 0 or idx >= len(tafsir_keys):
            continue
        tafsir_entry = tafsir_data.get(tafsir_keys[idx], {})
        tafsir_text = tafsir_entry.get("text", "") if isinstance(tafsir_entry, dict) else str(tafsir_entry)
        results.append({"key": tafsir_keys[idx], "tafsir": tafsir_text})
    return results

# QA multilingue avec contexte
def qa_multilang(user_question, history):
    lang_detected = detect_language(user_question)
    full_question = " ".join([h["question"] for h in history]) + " " + user_question if history else user_question

    try:
        if lang_detected != "sq":
            question_albanian = GoogleTranslator(source='auto', target='sq').translate(full_question)
        else:
            question_albanian = full_question
    except Exception:
        question_albanian = full_question

    tafsir_results = search_tafsir(question_albanian, top_k=3)
    if not tafsir_results:
        return "Aucune réponse trouvée.", lang_detected

    combined_albanian = " ".join(str(res['tafsir']) for res in tafsir_results if res['tafsir'])
    combined_albanian = nettoyer_html(combined_albanian)
    first_paragraph = combined_albanian.split('\n')[0]

    try:
        if lang_detected != "sq":
            answer_translated = GoogleTranslator(source='sq', target=lang_detected).translate(first_paragraph)
        else:
            answer_translated = first_paragraph
    except Exception:
        answer_translated = first_paragraph

    answer_clean = nettoyer_html(reformulate_text(answer_translated, lang_detected))
    return answer_clean, lang_detected

# (Le reste de ton code Streamlit reste inchangé, c'est l'interface, appels API, etc.)

