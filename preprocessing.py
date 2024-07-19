# preprocessing function
import regex as re
import pandas as pd
import spacy
from spacy.tokenizer import Tokenizer
from spacy.util import compile_infix_regex
from spacy.lang.char_classes import LIST_ELLIPSES, LIST_ICONS
from spacy.attrs import ORTH

nlp = spacy.load("en_core_web_sm")

def custom_tokenizer(nlp, phrases):

    # Create a new tokenizer with the default settings
    infix_re = compile_infix_regex(LIST_ELLIPSES + LIST_ICONS)
    tokenizer = Tokenizer(nlp.vocab, infix_finditer=infix_re.finditer)
    
    # Add custom rules to the tokenizer
    for phrase in phrases:
        tokenizer.add_special_case(phrase, [{ORTH: phrase}])
    
    return tokenizer

def clean(text):
  # hapus tanda baca
  punc = ['!', '(', ')', '[', ']', '{', '}', ';', ':', '"', "'", '\\', ',', '—', '–', '•',
          '<', '>', '.', '/', '?', '@', '#', '$', '%', '^', '&', '*', '_', '~', '|', '-', '»']

  for elemen in punc:
    text = text.replace(elemen,'')
    #text = text.lower()

  return text

data = pd.read_excel('C:\Penyimpanan Utama\Document\Coding\Data\Data Skenario II.xlsx')
data = data['Data'].apply(clean)
dictionary = data.to_list()
nlp.tokenizer = custom_tokenizer(nlp, dictionary)
