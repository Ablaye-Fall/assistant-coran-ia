# assistant-coran-ia/app.py
import streamlit as st
import requests

st.set_page_config(page_title="Assistant Coran IA", layout="centered")
st.title("📖 Assistant Coran avec Traduction et Tafsir")

# === Fonctions pour API AlQuran.cloud ===
def get_surah_list():
    url = "http://api.alquran.cloud/v1/surah"
    response = requests.get(url)
    return response.json()["data"]

def get_verses(surah_number, translation_code="en.asad"):
    url = f"http://api.alquran.cloud/v1/surah/{surah_number}/editions/quran-simple,{translation_code}"
    response = requests.get(url)
    return response.json()["data"]

def get_tafsir(surah_number, ayah_number):
    url = f"http://api.quran-tafseer.com/quran/{surah_number}/{ayah_number}"
    try:
        response = requests.get(url)
        data = response.json()
        return data["text"] if data else "Tafsir non disponible."
    except:
        return "Tafsir non disponible."

# === Choix de la sourate ===
surahs = get_surah_list()
surah_names = [f"{s['number']}. {s['englishName']} ({s['name']})" for s in surahs]
surah_choice = st.selectbox("📚 Choisissez une sourate :", surah_names)
surah_number = int(surah_choice.split('.')[0])

# === Choix de la langue de traduction ===
translation_options = {
    "🇫🇷 Français (Hamidullah)": "fr.hamidullah",
    "🇬🇧 English (Muhammad Asad)": "en.asad",
    "🇮🇩 Indonésien": "id.indonesian",
    "🇹🇷 Turc": "tr.translator",
    "🇺🇿 Ouzbek": "uz.sodik"
}
translation_label = st.selectbox("🌐 Choisir une langue de traduction :", list(translation_options.keys()))
translation_code = translation_options[translation_label]

# === Versets ===
verses_data = get_verses(surah_number, translation_code)
versets = verses_data[0]['ayahs']  # Arabe
traductions = verses_data[1]['ayahs']  # Traductions

if versets:
    verse_index = st.number_input("Choisir numéro du verset", min_value=1, max_value=len(versets), value=1)
    selected_verse = versets[verse_index - 1]
    translated_verse = traductions[verse_index - 1]

    st.markdown("### 🕋 Verset en arabe")
    st.markdown(f"**{selected_verse['text']}**")

    st.markdown(f"### 🌍 Traduction ({translation_label})")
    st.markdown(f"*{translated_verse['text']}*")

    st.markdown("### 📖 Tafsir (en arabe uniquement pour le moment)")
    tafsir = get_tafsir(surah_number, verse_index)
    st.markdown(tafsir)
else:
    st.warning("Aucun verset disponible pour cette sourate.")
