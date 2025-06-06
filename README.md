# Assistant Coran IA

Un assistant intelligent pour l'apprentissage du Coran avec analyse vocale, plan de révision, audio guidé et carte mentale.

## ✨ Fonctionnalités

- Transcription vocale avec Whisper (ou saisie manuelle)
- Détection d’erreurs dans la récitation
- Score global de similarité
- Mots mal récités enregistrés en base SQLite
- Audio de révision mot par mot
- Génération de PDF de plan de travail
- Carte mentale automatique (Graphviz)

## 🚀 Démarrage

### Prérequis

- Python 3.10+
- `ffmpeg` et `graphviz` installés localement
- Une clé API OpenAI (pour Whisper si activé)

### Installation

```bash
git clone https://github.com/Ablaye-Fall/assistant-coran-ia.git
cd assistant-coran-ia
pip install -r requirements.txt
```

Ajoute ta clé API dans `.streamlit/secrets.toml` :

```toml
[general]
OPENAI_API_KEY = "votre_clé_api"
```

### Lancement

```bash
streamlit run app.py
```

## 📁 Structure

- `app.py` — Application principale
- `audio/` — Stockage des fichiers audio
- `.streamlit/secrets.toml` — Clé API
- `memorisation.db` — Base SQLite générée automatiquement
