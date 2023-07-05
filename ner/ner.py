import pandas as pd
import spacy

# Load the pre-trained spaCy NER model
nlp = spacy.load("en_core_web_sm")

# Define functions for Named Entity Recognition (NER)
def ner_ORG(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ == 'ORG':
            entities.append(ent.text)
    return entities

def ner_GPE(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        if ent.label_ =='GPE':
            entities.append(ent.text)
    return entities

# Apply NER to the 'text' column of the DataFrame
def apply_ner(data):
    data['ORG_entities'] = data['preprocessed_text'].apply(lambda x: ner_ORG(' '.join(x)))
    data['GPE_entities'] = data['preprocessed_text'].apply(lambda x: ner_GPE(' '.join(x)))
    return data
