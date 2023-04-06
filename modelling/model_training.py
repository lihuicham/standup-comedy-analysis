print("## Importing libraries")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *

data = pd.read_csv("transcripts.csv", index_col = 0)

with open("sample.txt", "r") as f:
    sample = "".join(f.readlines())

print("## Data Preprocessing")
inconsistent_idx = []
for row_idx in data.index:
    transcript = data.loc[row_idx, "Transcript"]
    title = data.loc[row_idx, "Title"]
    sentence_split = transcript.split("\n")
    num_brackets_per_sentence = [count_brackets(sentence) for sentence in sentence_split]

    if -1 in num_brackets_per_sentence:
        inconsistent_idx.append(row_idx)

clean_transcripts = data.drop(index = inconsistent_idx)

clean_transcripts["Clean Transcripts"] = clean_transcripts["Transcript"].apply(clean_text)
pol = lambda x: TextBlob(x).sentiment.polarity
sub = lambda x: TextBlob(x).sentiment.subjectivity

clean_transcripts["Polarity Score"] = clean_transcripts["Clean Transcripts"].apply(pol)
clean_transcripts["Subjectivity Score"] = clean_transcripts["Clean Transcripts"].apply(sub)

filtered_transcripts = filter_by_sentiment(
    clean_transcripts, sample
)

extracted_data = []
for transcript in filtered_transcripts["Transcript"]:
    extracted_features = extract_features_labels(transcript)
    extracted_data.append(extracted_features)

model_data = pd.concat(extracted_data, axis = 0, ignore_index = True)
model_data["Clean Script"] = model_data["Script"].apply(process_script)


# Training the model
print("## Model Training")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

has_script = model_data[model_data["Script"] != ""]

X = has_script["Clean Script"]
y = has_script["Label"]
X_upsampled, y_upsampled = upsample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_upsampled, y_upsampled, test_size = 0.3, random_state = 242)

vectorizer = TfidfVectorizer(min_df = 0.001)
train_vec = vectorizer.fit_transform(X_train).toarray()
test_vec = vectorizer.transform(X_test).toarray()

train_labels = pd.get_dummies(y_train)
test_labels = pd.get_dummies(y_test)

model = RandomForestClassifier()
model.fit(train_vec, y_train) 

y_prob = model.predict_proba(train_vec)
evaluate_cnfm_roc(y_train, y_prob, label = "Random Forest Training")

y_prob = model.predict_proba(test_vec)
evaluate_cnfm_roc(y_test, y_prob, label = "Random Forest Test")

import pickle
pickle.dump(model, open("trained_model", "wb"))