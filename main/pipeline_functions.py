from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
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


#function to preprocess transcript parts
def preprocess(text):
    stopwords_lst = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    add_punctuation = '“”’'

    #remove punctuation
    punctuation_free = "".join([i for i in text if i not in string.punctuation])
    
    #remove digits
    digits_removed = ''.join([i for i in punctuation_free if not i.isdigit()])
    
    #make words to lowercase
    text_lower = digits_removed.lower()
    
    #tokenize to words
    tokenized = word_tokenize(text_lower)
    
    #remove stopwords
    stopwords_removed = [i for i in tokenized if i not in stopwords_lst]
    
    #lemmatize
    lemm_text = [lemmatizer.lemmatize(word) for word in stopwords_removed]
    
    #join list of words back into string
    processed_text = ' '.join(lemm_text)
    
    #remove punctuations not found in string.punctuation
    final_text = "".join([i for i in processed_text if i not in add_punctuation])
    
    return final_text
    