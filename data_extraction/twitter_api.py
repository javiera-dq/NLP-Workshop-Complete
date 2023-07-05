import tweepy
import csv
import pandas as pd
from dotenv import load_dotenv
import os

# Get the current directory of your_file.py
current_dir = os.path.dirname(os.path.abspath(__file__))
# Define the path to the data folder
data_folder = os.path.join(current_dir, '..', 'data')

load_dotenv()

# Set up your Twitter API credentials
api_key = os.getenv('api_key')
api_secret = os.getenv('api_secret')
access_token = os.getenv('access_token')
access_token_secret = os.getenv('access_token_secret')

# Authenticate with the Twitter API
auth = tweepy.OAuthHandler(api_key, api_secret)
auth.set_access_token(access_token, access_token_secret)

# Create a Tweepy API object
api = tweepy.API(auth, wait_on_rate_limit=True) # wait_on_rate_limit needed to fetch all the needed tweets even when we exceed limit
try:
    api.verify_credentials()
    print('Successful Authentication')
except:
    print('Failed authentication')

hashtags = [
    "emobility",
    "e-mobility",
    "electric",
    "EV",
    "electricvehicle",
    "EV charging",
    "EV sustainability",
    "sustainable mobility",
    "clean energy",
    "battery tech",
    "battery life",
    "lithium ion battery",
    "renewable energy",
    "green energy",
    "smart grid",
    "energy efficiency",
    "solar power",
    "wind power",
    "energy storage",
    "electric car",
    "charge point",
    "electric bicycle",
    "e-scooter"
]

number_of_tweets = 500
tweets = []
for hashtag in hashtags:
    for i in tweepy.Cursor(api.search_tweets, 
                           q = hashtag, 
                           tweet_mode="extended",
                           result_type="recent",
                        #  result_type="popular", # to select most relevant tweets. this also reduces the number of results returned to 15 per query
                           lang="en").items(number_of_tweets):
        tweets.append({'text' : i.full_text, 
                       'retweet_count' : i.retweet_count,
                       'favorite_count' : i.favorite_count,
                       'created_at':i.created_at})
tweets = pd.DataFrame(tweets)

# Want to select most relevant tweets. 
# Drop rows where both retweet_count and favorite_count are 0
tweets = tweets[(tweets['retweet_count'] != 0) | (tweets['favorite_count'] != 0)]

# Drop duplicates based on the full_text field and keep the row with the maximum retweet_count and favorite_count
# Important to do this, otherwise retweets show in our data as unique when they are not. 
tweets = tweets.sort_values(['retweet_count', 'favorite_count'], ascending=False).drop_duplicates(subset='text')

print(len(tweets))

csv_file_path_export = os.path.join(data_folder, 'twitter_data.json')
tweets.to_csv(csv_file_path_export, encoding='utf-8', index=False)