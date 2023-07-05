import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns

# Read scraped and preprocessed data
df_data = pd.read_csv('../data/data_to_export.csv')

# Prepare the corpus of texts
corpus = df_data['text']

# Create a CountVectorizer to convert text into a matrix of token counts
# The LDA topic model algorithm requires a document word matrix as the main input.
vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10, # minimum reqd occurences of a word 
                             stop_words='english', # remove stop words           
                             lowercase=True, # convert all words to lowercase                 
                             token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3          
)
X = vectorizer.fit_transform(corpus)
print(X)

X_dense = X.toarray()
df = pd.DataFrame(X_dense, columns=vectorizer.get_feature_names_out())
print(df)

# # Create the LDA model with a specified number of topics
num_topics = 12
lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42)

# Fit the LDA model to the data
lda_model.fit(X)

# Get the topic-word distribution matrix
topic_word_matrix = lda_model.components_
print(topic_word_matrix)

feature_names = vectorizer.get_feature_names_out()
df = pd.DataFrame(topic_word_matrix, columns=feature_names)
pd.set_option('display.precision', 3)
print(df)

# Get the most probable words for each topic
feature_names = vectorizer.get_feature_names_out()
top_words_per_topic = []
for topic_idx, topic in enumerate(topic_word_matrix):
    top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]  # Get the top 5 words
    top_words_per_topic.append(top_words)

# Assign topics to the documents
document_topics = lda_model.transform(X) # this gives the document-topic distribution matrix
df_data['Topic'] = document_topics.argmax(axis=1)
# Set the style for plots
sns.set(style="whitegrid")
print(document_topics)

# Visualize the results using bar charts
fig, axs = plt.subplots(num_topics, figsize=(10, 6*num_topics))
colors = sns.color_palette("husl", num_topics)  # Choose a color palette
for topic_idx, (topic_words, color) in enumerate(zip(top_words_per_topic, colors)):
    axs[topic_idx].barh(range(5), lda_model.components_[topic_idx].argsort()[:-6:-1], color=color)
    axs[topic_idx].set_yticks(range(5))
    axs[topic_idx].set_yticklabels(topic_words)
    axs[topic_idx].set_title(f'Topic {topic_idx + 1}', fontweight='bold', color=color)
    axs[topic_idx].spines['top'].set_visible(False)
    axs[topic_idx].spines['right'].set_visible(False)
    axs[topic_idx].spines['bottom'].set_visible(False)
    axs[topic_idx].spines['left'].set_visible(False)

plt.tight_layout()
plt.show()

perplexity = []

for i in range(2,50):
    lda_model = LatentDirichletAllocation(n_components=i, random_state=42)
    lda_model.fit(X)
    perplexity.append(lda_model.score(X))

plt.plot(range(2,50), perplexity)
plt.xlabel('Perplexity')
plt.ylabel('Number of Topics')
plt.title('Perplexity Diagram')
plt.show()

# Export the results
df_data.to_csv('../data/lda_result.csv', encoding='utf-8', index=False)
