import numpy as np
import regex as re
import pandas as pd
import import_ipynb
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from map import token2idx, idx2token, tag2idx, idx2tag

import json

# Load token2idx and idx2token
with open('token2idx.json', 'r') as f:
    token2idx = json.load(f)

with open('idx2token.json', 'r') as f:
    idx2token = json.load(f)

# Load tag2idx and idx2ta
with open('tag2idx.json', 'r') as f:
    tag2idx = json.load(f)

with open('idx2tag.json', 'r') as f:
    idx2tag = json.load(f)


#token2idx = {int(k): v for k, v in token2idx.items()}
idx2token = {int(k): v for k, v in idx2token.items()}
#tag2idx = {int(k): v for k, v in tag2idx.items()}
idx2tag = {int(k): v for k, v in idx2tag.items()}


def model_predict(tokens):
    # load model
    model = load_model('NER-stock3.keras')
    # 
    if '<UNK>' not in token2idx:
        token2idx['<UNK>'] = len(token2idx)
        idx2token[len(idx2token)] = '<UNK>'

    # Mapping and padding
    maxlen = 40
    n_token = 9574
    tokens_idx = [[token2idx.get(w, token2idx['<UNK>']) for w in tokens]]
    prediksi_kata_padded = pad_sequences(sequences=tokens_idx, maxlen=maxlen, padding="post", value=n_token - 1)

    # Predict
    preds = np.argmax(model.predict(np.array(prediksi_kata_padded)), axis=-1)[0]

    # Convert predictions to tag names
    predicted_tags = [idx2tag[p] for p in preds[:len(tokens)]]  # ensure length matches tokens
    result = list(zip(tokens, predicted_tags))

    return result

from preprocessing import *

input_text = input('masukkan text: ')
input_clean = clean(input_text)
doc = nlp(input_clean)
entity = [token.text for token in doc]

tokens = model_predict(entity)

entity_result = []
for token, tag in tokens:
    result = (f"{token}: {tag}")
    entity_result.append(result)

print(entity_result)