# THIS CODE JUST RUN BEFORE TRAINING

import pandas as pd
import json

df = pd.read_excel('C:\Penyimpanan Utama\Document\Coding\Data\DATASET-STOCK_ENTITY_RECOGNITION.xlsx')
df = df.drop(columns=['Unnamed: 0'])

def encoding(data, token_or_tag):
    tok2idx = {}
    idx2tok = {}

    if token_or_tag == 'token':
        vocab = list(set(data['token'].tolist()))
    elif token_or_tag == 'label':
        vocab = list(set(data['label'].tolist()))

    tok2idx = {token: idx for idx, token in enumerate(vocab)}
    idx2tok = {idx: token for idx, token in enumerate(vocab)}

    return tok2idx, idx2tok

token2idx, idx2token = encoding(df, 'token')
tag2idx, idx2tag = encoding(df, 'label')

df['token_encode'] = df['token'].map(token2idx)
df['label_encode'] = df['label'].map(tag2idx)

df.to_excel('mapping.xlsx')

# Save the mappings to JSON files
with open('token2idx.json', 'w') as f:
    json.dump(token2idx, f)
with open('idx2token.json', 'w') as f:
    json.dump(idx2token, f)
with open('tag2idx.json', 'w') as f:
    json.dump(tag2idx, f)
with open('idx2tag.json', 'w') as f:
    json.dump(idx2tag, f)
