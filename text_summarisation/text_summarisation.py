import csv
import os
from transformers import pipeline

from transformers import pipeline
import pandas as pd

summarizer_pipeline = pipeline("summarization", model = "facebook/bart-large-cnn")
translation_pipeline = pipeline("translation_en_to_de", model = "t5-large")

def summarizer(text):
    max_sequence_length = 1024  # Maximum sequence length for the model
    if len(text) > max_sequence_length:
        text = text[:max_sequence_length]
    summary = summarizer_pipeline(text, 
                        max_length=50,
                        min_length=20)[0]['summary_text']
    return summary

def translator(text):
    translated_text = translation_pipeline(text)[0]['translation_text']
    return translated_text
