import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

# Set manual seed for reproducibility
torch.manual_seed(42)

# ===============================
# Download and Load ConvoKit Corpus
# ===============================
try:
    # Download the "conversations-gone-awry-corpus" dataset
    corpus = Corpus(filename=download("conversations-gone-awry-corpus"))
    print("Corpus successfully loaded.")
except Exception as e:
    print(f"An error occurred while loading the corpus: {e}")
    exit()


# ===============================
# Extract Utterances DataFrame
# ===============================
# Convert utterances to a DataFrame
conv_wiki = corpus.get_utterances_dataframe().reset_index()

# Add a new column to categorize comments as 'Personal Attack' or 'Normal'
conv_wiki['Comment Nature'] = conv_wiki['meta.comment_has_personal_attack'].apply(
    lambda x: 'Personal Attack' if x is True else 'Normal'
)

# Drop unnecessary columns
conv_wiki = conv_wiki.drop(
    columns=['reply_to', 'meta.is_section_header', 'meta.parsed', 'meta.comment_has_personal_attack', 'vectors']
)

# Rename columns for consistency and clarity
conv_wiki = conv_wiki.rename(
    columns={
        'id': 'Utterance-ID',
        'meta.toxicity': 'Toxicity'
    }
)

# Check the range of toxicity values
toxicity_range = conv_wiki['Toxicity'].max() - conv_wiki['Toxicity'].min()
print(f"Toxicity range: {toxicity_range}")


# ===============================
# Extract Conversations DataFrame
# ===============================
# Convert conversations to a DataFrame
conversation_df = corpus.get_conversations_dataframe().reset_index()

# Drop unnecessary columns
conversation_df = conversation_df.drop(
    columns=['meta.page_id', 'meta.page_title', 'vectors']
)

# Clean column names by removing the 'meta.' prefix
conversation_df.columns = conversation_df.columns.str.replace('meta.', '')

# ===============================
# Merge Utterances and Conversations
# ===============================
# Merge utterances and conversations data on the conversation ID
talk_df = pd.merge(conv_wiki, conversation_df, left_on='conversation_id', right_on='id')

# ================
# Data Preparation
# ================
# Load dataset
talk_df = pd.read_csv('compiled_data.csv')

talk_df['datetime'] = pd.to_datetime(talk_df['timestamp'], unit='s')
talk_df['hour'] = talk_df['datetime'].dt.hour
talk_df['day'] = talk_df['datetime'].dt.dayofweek

# Define time of day categories
time_bins = [0, 6, 12, 18, 24]
time_labels = ['Night', 'Morning', 'Afternoon', 'Evening']
talk_df['time_of_day'] = pd.cut(talk_df['hour'], bins=time_bins, labels=time_labels, right=False)

# Pre-processing and feature engineering
talk_df = talk_df.sort_values(by = ['conversation_id', 'timestamp'], ascending = True)
talk_df = talk_df.dropna(subset=['Toxicity'])

# Label encoding for 'Comment Nature'
comment_nature_one_hot = pd.get_dummies(talk_df['Comment Nature'], prefix="nature")
talk_df = pd.concat([talk_df, comment_nature_one_hot], axis=1)

# List the one-hot encoded columns
comment_nature_columns = [col for col in talk_df.columns if col.startswith('nature_')]

threshold = 0.5
high_toxicity_weight = 2.0
low_toxicity_weight = 1.0

# Assign weights: higher weight for toxicity scores greater than threshold
talk_df['weights'] = talk_df['Toxicity'].apply(lambda x: high_toxicity_weight if x > threshold else low_toxicity_weight)