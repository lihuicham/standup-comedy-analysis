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

if chosen_comedian == "":
    comedian_transcripts = clean_transcripts
    print("All comedians will be used for training")
else:
    comedian_transcripts = clean_transcripts[clean_transcripts["Comedian"] == chosen_comedian]
    print(f"Number of Transcripts for {chosen_comedian}: {len(comedian_transcripts)}")

extracted_data = []
for transcript in comedian_transcripts["Transcript"]:
    extracted_features = extract_features_labels(transcript)
    extracted_data.append(extracted_features)

model_data = pd.concat(extracted_data, axis = 0, ignore_index = True)
model_data["Clean Script"] = model_data["Script"].apply(process_script)


# Training the model
print("## Model Training")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import roc_auc_score
from sklearn.naive_bayes import GaussianNB
from FreqNaiveBayes import FreqNaiveBayes

has_script = model_data[model_data["Script"] != ""]

X = has_script["Clean Script"]
y = has_script["Label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 242)

scores = []
params = dict(
    freq_margins = [0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3]
)
cv_results = cross_validate(
    FreqNaiveBayes(), X_train, y_train, 
    return_train_score = True,
    fit_params = params
)

print(pd.DataFrame(cv_results))