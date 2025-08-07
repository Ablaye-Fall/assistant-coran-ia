import json
import numpy as np
from sentence_transformers import SentenceTransformer

# Charger le fichier tafsir
with open("sq-saadi.json", "r", encoding="utf-8") as f:
    tafsir_data = json.load(f)

# Extraire les textes
tafsir_keys = []
tafsir_texts = []

for key, value in tafsir_data.items():
    if isinstance(value, dict):
        text = value.get("text", "").strip()
        if text:
            tafsir_keys.append(key)
            tafsir_texts.append(text)

print(f"Nombre de tafsir encodés : {len(tafsir_texts)}")

# Encoder avec le modèle
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(tafsir_texts, convert_to_numpy=True)

# Sauvegarder les fichiers
np.save("tafsir_embeddings.npy", embeddings)

with open("tafsir_keys.json", "w", encoding="utf-8") as f:
    json.dump(tafsir_keys, f, ensure_ascii=False, indent=2)

print("✅ Embeddings enregistrés dans 'tafsir_embeddings.npy'")
print("✅ Clés enregistrées dans 'tafsir_keys.json'")
