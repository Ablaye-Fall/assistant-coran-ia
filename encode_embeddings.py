import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import NearestNeighbors
import joblib

# === 1. Charger le fichier sq-saadi.json ===
with open("sq-saadi.json", "r", encoding="utf-8") as f:
    tafsir_data = json.load(f)

# === 2. Initialiser le modèle d'embedding léger ===
model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

# === 3. Préparer les textes et les clés ===
keys = list(tafsir_data.keys())  # ex: ['1:1', '1:2', ...]
texts = [tafsir_data[key] for key in keys]

# === 4. Générer les embeddings ===
print("🔄 Génération des embeddings...")
embeddings = model.encode(texts, show_progress_bar=True)

# === 5. Sauvegarder les fichiers ===
np.save("tafsir_embeddings.npy", embeddings)
with open("tafsir_keys.json", "w", encoding="utf-8") as f:
    json.dump(keys, f, ensure_ascii=False)

# === 6. Créer l’index avec NearestNeighbors ===
print("📦 Création de l’index sklearn...")
nn_index = NearestNeighbors(n_neighbors=5, metric="cosine")
nn_index.fit(embeddings)

# Sauvegarder l’index avec joblib
joblib.dump(nn_index, "tafsir_index_sklearn.joblib")

print("✅ Fichiers générés avec succès (sans faiss).")

