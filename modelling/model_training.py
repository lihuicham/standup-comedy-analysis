print("## Importing libraries")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *

data = pd.read_csv("transcripts.csv", index_col = 0)

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

chosen_comedian = "Trevor Noah"
comedian_transcripts = clean_transcripts[clean_transcripts["Comedian"] == chosen_comedian]
print(f"Number of Transcripts for {chosen_comedian}: {len(comedian_transcripts)}")

extracted_data = []
for transcript in clean_transcripts["Transcript"]:
    extracted_features = extract_features_labels(transcript)
    extracted_data.append(extracted_features)

model_data = pd.concat(extracted_data, axis = 0, ignore_index = True)
print(f"Model data shape: {model_data.shape}")


# Training the model
print("## Model Training")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

model_data["Clean Script"] = model_data["Script"].apply(process_script)
annotated = model_data[model_data["Script"] != ""]

X = annotated["Clean Script"]
y = annotated["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 242)

vectorizer = TfidfVectorizer()
train_vec = vectorizer.fit_transform(X_train).toarray()
test_vec = vectorizer.transform(X_test).toarray()
print(f"Training vector shape: {train_vec.shape}")

gnb = GaussianNB()
gnb.fit(train_vec, y_train)
y_prob = gnb.predict_proba(train_vec)
evaluate_cnfm_roc(y_train, y_prob, "All comedians NB", savefig = True)

y_prob = gnb.predict_proba(test_vec)
evaluate_cnfm_roc(y_test, y_prob, "All comedians Test NB", savefig = True)