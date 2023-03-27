import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def count_brackets(sentence):
    """
    Count the number of open and close brackets. If the number of open and close brackets doesnt match,
    we have invalid annotations, then return -1. Else, we return the number of annotations found
    """
    
    open_brackets = sentence.count("[")
    close_brackets = sentence.count("]")

    if open_brackets == close_brackets:
        return open_brackets

    return -1


def reorganise_transcript(transcript):
    """
    Reorganise the transcript such that every sentence starts with an annotation.
    Returns a list of strings, where every string is a sentence beginning with annotations
    """
    
    transcript_soup = transcript.replace("\n", " ")

    return [
        "[" + sentence.strip()                      # Strip sentence and add open bracket (which was removed from splitting)
        for sentence in transcript_soup.split("[")  # Split the transcript by open brackets (each line begins with an annotation now)
        if sentence != ""                           # The first annotation is empty, so skip that
    ]


def extract_features_labels(transcript):
    """
    Extracts annotations and scripts in the transcript, using a reorganised transcript.

    """
    
    reorged = reorganise_transcript(transcript)

    # Split each sentence of the transcript into its annotation and script
    annot_script = [sentence.split("]") for sentence in reorged]

    # [1: ] is to remove the open brackets (close brackets removed from splitting)
    annots = [pair[0][1: ] for pair in annot_script]

    # If the sentence just contains annotations, the use an empty string for the script
    script = [pair[1].strip() if len(pair) > 1 else "" for pair in annot_script]


    # Boolean array to state whether the current sentence is funny
    labels = [False for _ in range(len(annots))]

    # The last sentence cant cause laughter, so loop to 2nd last sentence
    for idx in range(len(annots) - 1):
        # For each sentence, the check if the annotations contains laugh.
        # The preceding sentence's label is whether or not the next annotation contains laugh
        laughter = "laugh" in annots[idx].lower()
        labels[idx - 1] = laughter
            

    # Return the extracted features and labels as a dataframe 
    return pd.DataFrame(zip(labels, script, annots), columns = ["Label", "Script", "Annotation"])


def process_script(script):
    """
    Given a script, perform cleaning and processing for NLP
    """
    
    punct = string.punctuation
    stop_words = stopwords.words("english")
    stemmer = PorterStemmer()

    clean1 = script.translate(str.maketrans('', '', punct))                  # Remove punctuation
    clean2 = re.sub(r'[^\x00-\x7F]+','', clean1)                             # Remove non-ASCII
    clean3 = [word for word in clean2.split(" ") if word not in stop_words]  # Remove stopwords
    clean4 = [stemmer.stem(word) for word in clean3]                         # Stem the words

    return " ".join(clean4).lower()


def evaluate_cnfm_roc(y, y_prob, label = "", savefig = False):
    y_hat = np.argmax(y_prob, axis = 1)
    cnfm = confusion_matrix(y, y_hat)
    roc_auc = roc_auc_score(y, y_hat)

    fpr, tpr, threshold = roc_curve(y, y_prob[:, 1])

    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 8))
    axs[0].set_title("Baseline model prediction")
    sns.heatmap(
        cnfm, cmap = "Blues",
        annot = True, fmt = ".0f",
        ax = axs[0]
    )

    axs[1].set_title(f"ROC Curve: AUC {roc_auc:.3f}")
    axs[1].plot(fpr, tpr)
    if savefig:
        plt.savefig(f"./images/{label} results.png")
    plt.show()