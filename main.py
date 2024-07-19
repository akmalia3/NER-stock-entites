import streamlit as st
import requests
import re
import numpy as np
import pandas as pd

st.title('Named Entity Recognoition 🔤')
st.write('Sistem untuk mengidentifikasi entitas saham. Ada 5 entitas yang dapat sistem identifikasi diantaranya nama perusahaan, kode saham, indeks saham, sektor industri dan sub sektor')

# input user
st.subheader('Coba disini')
text = st.text_input("🔗Input URL")

# fungsi untuk extract teks dari url
def get_url(input):
    respons = requests.get(text)
    soup = BeautifulSoup(respons.content, 'html.parser')
    data = soup.get_text(strip=True)
    return data

# fungsi untuk memvalidasi url
def validate_url(url):
    """Validates and adds schema (https) if missing."""
    if not url.startswith('http'):
        url = 'https://' + url
    return url

#menghandel input kosong
input_user = validate_url(text) if text else None

text_input = []

if input_user:
    extraxt_tetxt = get_url(input_user)
    if extraxt_tetxt:
        text_input.append(extraxt_tetxt)
    else:
        st.write("Gagal extraxt text dari URL. Coba Lagi!!")
else:
    st.write("Kosong")
    

# Input dari text
st.subheader('Atau')
txt = st.text_area("Masukkan teks disini 👇")
text_input.append(txt)

# preprocessing input
from preprocessing import *

if text_input:
    text_clean = clean(' '.join(text_input))
    #text_clean_str = ' '.join(text_clean)  # Convert the list of tokens to a string
    doc = nlp(text_clean)
    entity = [token.text for token in doc]
    #st.write(tokens)
else:
    st.write("Please enter some text to process.")


from ner_model_load import model_predict

tokens = model_predict(entity)

entity_result = []
for token, tag in tokens:
    result = (f"{token}: {tag}")
    entity_result.append(result)


from display_ner import convert_to_annotated_text
annotated_result = convert_to_annotated_text(tokens)
print(annotated_result)

# Next step for the visualization
from annotated_text import annotated_text

st.subheader('Entitas Teridentifikasi')

if annotated_result:
    annotated_text([elemen for elemen in annotated_result])
else:
    st.write("No entities found.")

