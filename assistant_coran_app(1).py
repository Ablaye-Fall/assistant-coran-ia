# app.py
import streamlit as st
import json
import os
import requests
from datetime import datetime
import whisper
import difflib
from difflib import SequenceMatcher
import sounddevice as sd
import soundfile as sf
import tempfile

# Fonction pour charger les sourates, translittération et traduction depuis l'API
@st.cache_data
def fetch_quran():
    res_ar = requests.get("https://api.alquran.cloud/v1/quran/quran-uthmani").json()
    res_en = requests.get("https://api.alquran.cloud/v1/quran/en.sahih").json()
    res_translit = requests.get("https://api.alquran.cloud/v1/quran/en.transliteration").json()

    sourates = {}
    for i in range(len(res_ar["data"]["surahs"])):
        surah_ar = res_ar["data"]["surahs"][i]
        surah_en = res_en["data"]["surahs"][i]
        surah_translit = res_translit["data"]["surahs"][i]
        nom = f"{surah_ar['number']}. {surah_ar['englishName']}"
        versets = []
        for j in range(len(surah_ar["ayahs"])):
            versets.append({
                "ar": surah_ar["ayahs"][j]["text"],
                "translit": surah_translit["ayahs"][j]["text"],
                "trad": surah_en["ayahs"][j]["text"]
            })
        sourates[nom] = versets
    return sourates

coran_data = fetch_quran()

# Charger le modèle Whisper
@st.cache_resource
def load_model():
    return whisper.load_model("base")

model = load_model()

# Page config
st.set_page_config(page_title="📖 Assistant Coran IA")
st.title("📖 Assistant Coran IA")

# Choisir une sourate
sourate = st.selectbox("Choisir une sourate", list(coran_data.keys()))
versets = coran_data[sourate]

# Créer le dossier audio
if not os.path.exists("audio"):
    os.makedirs("audio")

# Plan de révision à générer
plan_revision = []

# Surlignage des erreurs mot à mot
def surligner_erreurs(ref, hyp):
    ref_words = ref.split()
    hyp_words = hyp.split()
    sm = SequenceMatcher(None, ref_words, hyp_words)
    result = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            result.extend([f"<span style='color:green'>{word}</span>" for word in ref_words[i1:i2]])
        elif tag in ('replace', 'delete'):
            result.extend([f"<span style='color:red'>{word}</span>" for word in ref_words[i1:i2]])
        elif tag == 'insert':
            result.extend([f"<span style='color:orange'>{word}</span>" for word in hyp_words[j1:j2]])
    return ' '.join(result)

# Affichage versets et enregistrement
for idx, verset in enumerate(versets, start=1):
    st.subheader(f"Verset {idx}")
    st.markdown(f"### :orange[{verset['ar']}]")
    st.markdown(f"**🔡 Translittération :** {verset['translit']}")
    st.markdown(f"**🇫🇷 Traduction :** {verset['trad']}")

    use_micro = st.checkbox(f"🎤 Enregistrer avec micro (verset {idx})", key=f"mic_{idx}")
    if use_micro:
        duration = st.number_input("⏱️ Durée d'enregistrement (sec)", min_value=2, max_value=30, value=6, key=f"dur_{idx}")
        if st.button(f"🎙️ Démarrer l'enregistrement {idx}"):
            st.info("📢 Parle maintenant...")
            fs = 44100
            myrecording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
            sd.wait()
            temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(temp_audio.name, myrecording, fs)
            st.success("✅ Enregistrement terminé")
            st.audio(temp_audio.name)
            file_path = temp_audio.name
        else:
            file_path = None
    else:
        uploaded_file = st.file_uploader(f"📤 Uploade ta récitation (verset {idx})", type=["mp3"], key=f"upload_{idx}")
        if uploaded_file:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            audio_path = os.path.join("audio", f"{sourate.replace('.', '').replace(' ', '_')}_v{idx}_{timestamp}.mp3")
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.read())
            st.success(f"Fichier enregistré : {audio_path}")
            st.audio(audio_path)
            file_path = audio_path
        else:
            file_path = None

    # Transcription + score si audio disponible
    if file_path:
        st.info("📥 Transcription en cours...")
        result = model.transcribe(file_path, language="ar")
        transcription = result["text"].strip()

        st.markdown("**📝 Comparaison mot à mot :**", unsafe_allow_html=True)
        highlight = surligner_erreurs(verset['ar'], transcription)
        st.markdown(f"<p style='font-size:18px'>{highlight}</p>", unsafe_allow_html=True)

        ratio = difflib.SequenceMatcher(None, verset['ar'], transcription).ratio()
        score = round(ratio * 100, 2)
        st.metric("🎯 Score de récitation", f"{score}%")

        if score < 90:
            plan_revision.append({"verset": verset['ar'], "score": score, "transcription": transcription})

# Plan de révision personnalisé
if plan_revision:
    st.subheader("🧠 Plan de travail personnalisé")
    for item in plan_revision:
        st.markdown(f"- ❗ Verset : :orange[{item['verset']}] — Score : **{item['score']}%**")
        st.markdown(f"  - 🔁 Transcription : :blue[{item['transcription']}]")

    if st.button("📄 Télécharger le plan en JSON"):
        plan_path = "plan_revision.json"
        with open(plan_path, "w", encoding="utf-8") as f:
            json.dump(plan_revision, f, ensure_ascii=False, indent=2)
        with open(plan_path, "rb") as f:
            st.download_button(label="📥 Télécharger", data=f, file_name="plan_revision.json", mime="application/json")
