import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
import string
import re

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve


def count_brackets(sentence):
    open_brackets = sentence.count("[")
    close_brackets = sentence.count("]")

    if open_brackets == close_brackets:
        return open_brackets

    return -1


def reorganise_transcript(transcript):
    transcript_soup = transcript.replace("\n", " ")
    return ["[" + sentence.strip() for sentence in transcript_soup.split("[") if sentence != ""]


def extract_features_labels(transcript):
    reorged = reorganise_transcript(transcript)

    annot_script = [sentence.split("]") for sentence in reorged]
    annots = [pair[0][1: ] for pair in annot_script]
    script = [pair[1].strip() if len(pair) > 1 else "" for pair in annot_script]

    # The last sentence cant cause laughter
    labels = [False for _ in range(len(annots))]
    for idx in range(len(annots) - 1):
        laughter = "laugh" in annots[idx].lower()
        labels[idx - 1] = laughter
            
    return pd.DataFrame(zip(labels, script, annots), columns = ["Label", "Script", "Annotation"])


def process_script(script):
    punct = string.punctuation
    stop_words = stopwords.words("english")

    clean1 = script.translate(str.maketrans('', '', punct))                  # Remove punctuation
    clean2 = re.sub(r'[^\x00-\x7F]+','', clean1)                             # Remove non-ASCII
    clean3 = [word for word in clean2.split(" ") if word not in stop_words]  # Remove stopwords

    return " ".join(clean3).lower()


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
        plt.savefig(f"{label} results")
    plt.show()