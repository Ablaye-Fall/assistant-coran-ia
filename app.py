import streamlit as st
import sqlite3
import difflib
from gtts import gTTS
from fpdf import FPDF
from graphviz import Digraph
from pathlib import Path

st.set_page_config(page_title="Assistant Coran IA", layout="wide")
st.title("📖 Assistant Coran IA")

# Connexion SQLite
conn = sqlite3.connect("memorisation.db")
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS memorisation (mot TEXT, erreurs INT, date_dernier_test TEXT)")
conn.commit()

# API fictive : Texte coranique simulé
coran = {
    "Al-Fatiha": [
        "Bismi Allahi alrrahmani alrraheemi",
        "Alhamdu lillahi rabbi alAAalameena",
        "Alrrahmani alrraheemi",
        "Maliki yawmi alddeeni"
    ]
}

# Sélection de sourate
sourate = st.selectbox("Choisir une sourate", list(coran.keys()))
versets = coran[sourate]

# Récitation de l'utilisateur
for i, verset in enumerate(versets):
    st.markdown(f"### Verset {i+1}")
    user_input = st.text_area("Ta récitation :", key=f"input_{i}")
    expected = verset

    if user_input:
        ratio = difflib.SequenceMatcher(None, user_input.lower(), expected.lower()).ratio()
        st.write(f"📊 Similarité : {int(ratio * 100)}%")

        diff = []
        for w1, w2 in zip(user_input.split(), expected.split()):
            if w1 != w2:
                diff.append((w1, w2))
        if diff:
            st.error("🔍 Mots incorrects : " + ", ".join([f"{w1} → {w2}" for w1, w2 in diff]))
            for mot_err in diff:
                cursor.execute("INSERT INTO memorisation (mot, erreurs, date_dernier_test) VALUES (?, 1, date('now'))", (mot_err[1],))
            conn.commit()

        st.success("✅ Analyse terminée")

# Revoir les mots mal récités
st.header("🔁 Revoir les mots mal récités")
cursor.execute("SELECT mot, COUNT(*) as nb FROM memorisation GROUP BY mot ORDER BY nb DESC LIMIT 5")
rows = cursor.fetchall()
for mot, nb in rows:
    st.write(f"**{mot}** — erreurs : {nb}")
    if st.button(f"🔊 Écouter {mot}", key=mot):
        tts = gTTS(mot, lang='ar')
        audio_path = f"audio/{mot}.mp3"
        tts.save(audio_path)
        st.audio(audio_path)

# Génération d'un plan PDF
if st.button("📄 Générer un plan PDF personnalisé"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, "Plan de révision — Mots à revoir", ln=True, align="C")
    pdf.ln(10)
    for mot, nb in rows:
        pdf.cell(200, 10, f"{mot} — erreurs : {nb}", ln=True)
    pdf_path = "plan_revision.pdf"
    pdf.output(pdf_path)
    st.success("📄 Fichier PDF généré")
    st.download_button("⬇️ Télécharger le PDF", data=open(pdf_path, "rb"), file_name=pdf_path)

# Carte mentale avec Graphviz
st.header("🧠 Carte mentale")
dot = Digraph()
dot.attr(size='8,5')
dot.node("Révision")
for mot, _ in rows:
    dot.node(mot)
    dot.edge("Révision", mot)
st.graphviz_chart(dot.source)
