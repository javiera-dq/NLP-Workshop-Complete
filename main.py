import pandas as pd
import os
import json
from data_cleaning.data_cleaning import preprocess
from sentiment_analysis.sentiment_analysis import add_sentiment_columns
from ner.ner import apply_ner
from text_summarisation.text_summarisation import summarizer
from text_summarisation.text_summarisation import translator
from topic_modeling.export_topic_model_BERTopic import generate_BERTopic_cytoscape


if __name__ == '__main__':
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Define the path to the data folder
    data_folder = os.path.join(current_dir, 'data')

    # OR import the News API data
    file_path_import = os.path.join(data_folder, 'news_data.json')
    with open(file_path_import, 'r') as json_file:
        data = json.load(json_file)
    data = pd.DataFrame(data)

    # Clean and preprocess the text data
    data['preprocessed_text'] = data['text'].apply(preprocess)
    # Perform sentiment analysis
    data = add_sentiment_columns(data)
    # Perform Named Entity Recognition (NER)
    data = apply_ner(data)
    # add summarisation column
    data['summary'] = data['text'].apply(summarizer)
    # BERTopic data
    data= generate_BERTopic_cytoscape(data)
    print(data)

    # Save data and export to be analyzed in Power BI
    file_path_export = os.path.join(data_folder, 'data_to_export_final.csv')
    data.to_csv(file_path_export, encoding='utf-8', index=False)
