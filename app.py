import streamlit as st
import requests

# Configuration page
st.set_page_config(page_title="Assistant Coran IA avec traduction et tafsir", layout="wide")
st.title("📖 Assistant Coran IA avec traduction et tafsir")

@st.cache_data(ttl=3600)
def get_surah_list():
    url = "https://api.alquran.cloud/v1/surah"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        return [(s['number'], s['englishName'], s['name']) for s in data['data']]
    else:
        st.error("Erreur récupération sourates")
        return []

@st.cache_data(ttl=3600)
def get_verses_with_translations_and_tafsir(surah_number):
    url_ar = f"https://api.alquran.cloud/v1/surah/{surah_number}/ar.alafasy"
    url_fr = f"https://api.alquran.cloud/v1/surah/{surah_number}/fr"
    url_tafsir = f"https://api.alquran.cloud/v1/surah/{surah_number}/tafsir/fr.jalalayn"

    res_ar = requests.get(url_ar)
    res_fr = requests.get(url_fr)
    res_tafsir = requests.get(url_tafsir)

    if res_ar.status_code == 200 and res_fr.status_code == 200 and res_tafsir.status_code == 200:
        data_ar = res_ar.json()['data']['ayahs']
        data_fr = res_fr.json()['data']['ayahs']
        data_tafsir = res_tafsir.json()['data']['ayahs']

        combined = []
        for i in range(len(data_ar)):
            combined.append((
                data_ar[i]['text'],
                data_fr[i]['text'],
                data_tafsir[i]['text']
            ))
        return combined
    else:
        st.error("Erreur récupération des données")
        return []

surah_list = get_surah_list()
surah_names = [f"{num}. {eng} ({arab})" for num, eng, arab in surah_list]
sourate_choice = st.selectbox("Choisir une sourate", surah_names)
surah_number = int(sourate_choice.split(".")[0])

versets = get_verses_with_translations_and_tafsir(surah_number)

verse_index = st.number_input("Choisir numéro du verset", min_value=1, max_value=len(versets), value=1)
texte_ar, traduction_fr, tafsir_fr = versets[verse_index - 1]

st.markdown(f"### Verset {verse_index} — Arabe :")
st.write(texte_ar)

st.markdown("### Traduction française :")
st.write(traduction_fr)

st.markdown("### Tafsir Jalalayn (fr) :")
st.write(tafsir_fr)
