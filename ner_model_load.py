import numpy as np
import regex as re
import pandas as pd
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
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


idx2token = {int(k): v for k, v in idx2token.items()}
idx2tag = {int(k): v for k, v in idx2tag.items()}

# Mapping
if '<UNK>' not in token2idx:
    token2idx['<UNK>'] = len(token2idx)
    idx2token[len(idx2token)] = '<UNK>'

def mapping(text, token2idx):
    return [[token2idx.get(w, token2idx['<UNK>']) for w in text]]

# Padding
def padd_inference(tokens, max_length=None):
    if max_length is None:
        # Use the built-in max function
        max_length = max(len(token) for token in tokens)  # Remove 'builtins.'

    pad_tokens = pad_sequences(tokens, maxlen=max_length, dtype='int32', padding='post')
    return pad_tokens

model = load_model('NER-BILSTM-FINAL.h5')

def model_predict(tokens):
    # Predict
    preds = np.argmax(model.predict(np.array(tokens)), axis=-1)[0]

    # Convert predictions to tag names
    predicted_tags = [idx2tag[p] for p in preds[:len(tokens)]]  # ensure length matches tokens
    result = list(zip(tokens, predicted_tags))

    return result
