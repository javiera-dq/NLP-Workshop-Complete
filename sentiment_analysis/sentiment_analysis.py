import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

# Instantiate the SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Define a discrete variable to tag Tweets as 'Positive', 'Negative' or 'Neutral'
def get_sentiment_group(score):
  if score < 0:
    return "Negative"
  elif score == 0:
    return "Neutral"
  else:
    return "Positive"

# Add sentiment-related columns to the dataframe  
def add_sentiment_columns(data):
    data['sentiment_neg'] = data['preprocessed_text'].apply(lambda x: sia.polarity_scores(' '.join(x))['neg'])
    data['sentiment_pos'] = data['preprocessed_text'].apply(lambda x: sia.polarity_scores(' '.join(x))['pos'])
    data['sentiment_neutral'] = data['preprocessed_text'].apply(lambda x: sia.polarity_scores(' '.join(x))['neu'])
    data['sentiment'] = data['preprocessed_text'].apply(lambda x: sia.polarity_scores(' '.join(x))['compound'])
    data["sentiment_group"] = data["sentiment"].apply(lambda x: get_sentiment_group(x))
    return data
