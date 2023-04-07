from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from textblob import TextBlob
import string
import re

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
    pol_distance = abs(data["Polarity"] - sample_pol)
    sub_distance = abs(data["Subjectivity"] - sample_sub)

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

#function from utils
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

#function from utils
def pos_then_lemmatize(pos_tagged_words):
    lemmatizer = WordNetLemmatizer()

    res = []
    for pos in pos_tagged_words : 
        word = pos[0]
        pos_tag = pos[1]

        lem = lemmatizer.lemmatize(word, get_wordnet_pos(pos_tag))
        res.append(lem)
    return res

#function to preprocess transcript parts [heavily adapted from custom_tokenizer_stop(doc) in utils, and added digit removal]
def preprocess(doc):
    #remove all action text, meaning bracketed words
    doc = re.sub('\[[a-zA-Z\s\-]+\]|\([a-zA-Z\s\-]+\)', '', doc)

    #remove punctuation
    punct = string.punctuation + "’“”–"
    doc = doc.translate(str.maketrans(punct, " " * len(punct)))

    #remove digits
    doc = ''.join([i for i in doc if not i.isdigit()])

    #make doc lowercase and tokenize to words
    words = word_tokenize(doc.lower())
    
    #add our own stop word list to the existing English stop words 
    stop_words = stopwords.words("english") + [
        "get", "go", "know", "dont", "im", "like", "say", "thats", "one", "come", "right", "think", "youre", 
        "people", "see", "look", "want", "time", "make", "na", "gon", "thing", "oh", "take", "good", "guy", 
        "fuck", "would", "yeah", "tell", "well", "he", "shit", "cause", "back", "theyre", "man", "really", "cant", "little",
        "let", "just", "okay", "ive", "♪", "–", "ta", "uh", "wan", "g", "e", "ah", "r", "mi", "le"
    ]
    
    #remove stopwords before lemmatization
    filtered_words = [w.strip() for w in words if not w in stop_words] 

    #pos tag words correctly and lemmatize them according to their corrected pos tag
    pos_tagged_words = pos_tag(filtered_words)
    pos_lemmatized_words = pos_then_lemmatize(pos_tagged_words)

    #remove stopwords after lemmatization
    filtered_words_2 = [w for w in pos_lemmatized_words if not w in stop_words] 

    #join list of words back into string
    final_text = ' '.join(filtered_words_2)

    return final_text