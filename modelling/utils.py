import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from textblob import TextBlob
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


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        # As default pos in lemmatization is Noun
        return wordnet.NOUN


def pos_then_lemmatize(pos_tagged_words):
    lemmatizer = WordNetLemmatizer()

    res = []
    for pos in pos_tagged_words : 
        word = pos[0]
        pos_tag = pos[1]

        lem = lemmatizer.lemmatize(word, get_wordnet_pos(pos_tag))
        res.append(lem)
    return res


def custom_tokenizer_stop(doc): 
    punct = string.punctuation + "’“”–"
    doc = doc.translate(str.maketrans(punct, " " * len(punct)))
    words = word_tokenize(doc.lower())
    
    # add our own stop word list to the existing English stop words 
    stop_words = stopwords.words("english") + [
        "get", "go", "know", "dont", "im", "like", "say", "thats", "one", "come", "right", "think", "youre", 
        "people", "see", "look", "want", "time", "make", "na", "gon", "thing", "oh", "take", "good", "guy", 
        "fuck", "would", "yeah", "tell", "well", "he", "shit", "cause", "back", "theyre", "man", "really", "cant", "little",
        "let", "just", "okay", "ive", "♪", "–", "ta", "uh", "wan", "g", "e", "ah", "r", "mi", "le"
    ]
    
    filtered_words = [w.strip() for w in words if not w in stop_words] 
    pos_tagged_words = pos_tag(filtered_words)
    pos_lemmatized_words = pos_then_lemmatize(pos_tagged_words)
    filtered_words_2 = [w for w in pos_lemmatized_words if not w in stop_words] 
    
    return filtered_words_2


def process_script(script):
    """
    Given a script, perform cleaning and processing for NLP
    """
    
    # punct = string.punctuation
    # stop_words = stopwords.words("english") + [
    #     "get", "go", "know", "dont", "im", "like", "say", "thats", "one", "come", "right", "think", "youre", 
    #     "people", "see", "look", "want", "time", "make", "na", "gon", "thing", "oh", "take", "good", "guy", 
    #     "fuck", "would", "yeah", "tell", "well", "he", "shit", "cause", "back", "theyre", "man", "really", "cant", "little",
    #     "let", "just", "okay", "ive", "♪", "–", "ta", "uh", "wan", "g", "e", "ah", "r", "mi", "le"
    # ]
    # stemmer = PorterStemmer()

    # clean1 = script.translate(str.maketrans("", "", punct))                  # Remove punctuation
    # clean2 = re.sub(r"[^\x00-\x7F]+","", clean1)                             # Remove non-ASCII
    # clean3 = [word for word in clean2.split(" ") if word not in stop_words]  # Remove stopwords
    # clean4 = [stemmer.stem(word) for word in clean3]                         # Stem the words

    # return " ".join(clean4).lower()

    filtered_tokens = custom_tokenizer_stop(script)
    return " ".join(filtered_tokens)


def upsample(X, y):
    laugh = X[y]
    no_laugh = X[~y]

    num_to_sample = len(no_laugh) - len(laugh)
    sampled_laugh_idx = laugh.sample(max(0, num_to_sample)).index
    sampled_idx = np.concatenate([sampled_laugh_idx, laugh.index, no_laugh.index])

    X_upsampled = X[sampled_idx]
    y_upsampled = y[sampled_idx]

    return X_upsampled, y_upsampled


def clean_text(text) :
    """Make text lowercase, remove text in square brackets, remove punctuations, 
    remove quotation marks, remove words containing numbers, remove \n"""
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)   
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\w*\d\w*', '', text) 
    text = re.sub('[‘’“”…]', '', text)
    text = re.sub('\n', '', text)
    
    return text


def filter_by_sentiment(data, sample, n = 10):
    """
    Choose the `n` closest transcripts based on polarity and subjectivity of the sample.
    The `n` closest is found by finding the `n` smallest difference between polarity and subjectivity.
    """

    # Create a TextBlob from the sample
    clean_sample = clean_text(sample)
    blob = TextBlob(clean_sample)

    # Get the polarity and subjetivity of the sample
    sample_pol = blob.sentiment.polarity
    sample_sub = blob.sentiment.subjectivity

    # Find the distances of polarity and subjectivty
    pol_distance = data["Polarity Score"] - sample_pol
    sub_distance = data["Subjectivity Score"] - sample_sub

    # Sort the distances in ascending order, and choose the index of the first `n` transcripts
    pol_closest = pol_distance.sort_values().head(n).index
    sub_closest = sub_distance.sort_values().head(n).index

    # Keep adding the smallest score's index until we hit `n` transcripts
    n_closest = []
    for idx in range(n):
        pol_idx, sub_idx = pol_closest[idx], sub_closest[idx]
        if pol_idx == sub_idx:
            # If the closest polarity and subjectivity are the same transcript, add only 1
            n_closest.append(pol_idx)
        else:
            n_closest.extend([pol_idx, sub_idx])

        if len(n_closest) == n:
            break

    return data.loc[n_closest, :].reset_index(drop = True)
    

def evaluate_cnfm_roc(y, y_prob, label = "", savefig = False):
    """
    Evaluates a prediction made.
    Plots the confusion matrix and ROC curve for the prediction
    If savefig is True, save the figure with a given label
    """
    
    y_hat = np.argmax(y_prob, axis = 1)
    cnfm = confusion_matrix(y, y_hat)
    roc_auc = roc_auc_score(y, y_hat)

    fpr, tpr, threshold = roc_curve(y, y_prob[:, 1])

    fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 8))
    axs[0].set_title(f"{label} predictions")
    sns.heatmap(
        cnfm, cmap = "Blues",
        annot = True, fmt = ".0f",
        ax = axs[0]
    )
    axs[0].set_xlabel("Predicted")
    axs[0].set_ylabel("True")

    axs[1].set_title(f"ROC Curve: AUC {roc_auc:.3f}")
    axs[1].plot(fpr, tpr)
    if savefig:
        plt.savefig(f"./images/{label} results.png")
    plt.show()


def build_lstm(input_shape):
    model = keras.Sequential()

    model.add(layers.Bidirectional(
        layers.LSTM(units = 128, return_sequences = True, input_shape = input_shape, name = "LSTM-1")
    ))
    model.add(layers.Bidirectional(
        layers.LSTM(units = 128, name = "LSTM-2")
    ))

    model.add(layers.Dense(128, activation = "relu", name = "FC-1"))
    model.add(layers.Dense(256, activation = "relu", name = "FC-2"))

    model.add(layers.Dense(2, activation = "sigmoid", name = "output"))
    model.compile(
        optimizer = "adam",
        loss = "binary_crossentropy",
        metrics = [
            keras.metrics.AUC(name = "AUC"),
            keras.metrics.Accuracy(name = "Accuracy")
        ]
    )
    return model