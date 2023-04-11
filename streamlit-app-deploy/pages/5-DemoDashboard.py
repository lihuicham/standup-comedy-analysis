import streamlit as st 
import pandas as pd
from empath import Empath
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from textblob import TextBlob
import string
import re
from sklearn.ensemble import GradientBoostingClassifier
from empath import Empath
import numpy as np
from imblearn.over_sampling import RandomOverSampler
import nltk 
nltk.download('all')

## helper functions from pipeline_functions.py
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
    
'''
# Demo 
'''
st.markdown('## Step 1 : User Input Transcript')
st.markdown('Copy sample transcript [here](https://github.com/lihuicham/standup-comedy-analysis/blob/main/main/sample.txt). You can click the copy icon to copy the text !')
user_input = st.text_area('Input the comedy transcript you would like to predict :', height=300)
user_input = ' '.join(user_input.split('\n'))
st.markdown('**Press `ctrl/cmd + enter` to process input transcript for next step.**')
st.markdown('*Note: The warning below is a caching warning from the deployment app, you can ignore it.*')
st.markdown('**Another Note : There will be an error below and you cannot see other steps of this dashboard if you do not fill in the input script above !**')
st.markdown('Another another note : This multiline text area needs ctrl/cmd + enter to run, so it can only work on laptops, not mobile phones :(')

@st.cache
def load_sent() :
    return pd.read_pickle('main/chloe_valid_transcripts_sent_df')

@st.cache
def load_df() :
    return pd.read_pickle('main/valid_transcripts_df')

valid_transcripts_sent_df = load_sent()
valid_transcripts_df = load_df()
filtered_transcripts_df = filter_by_sentiment(valid_transcripts_df, user_input)
filtered_transcripts_sent_df = valid_transcripts_sent_df[valid_transcripts_sent_df['Title'].isin(filtered_transcripts_df['Title'])]

@st.cache
def load_grad() :
    return pd.read_pickle('main/trained_grad_model')

@st.cache
def load_vectorizer() :
    return pd.read_pickle('main/full_vectorizer')

@st.cache
def load_features() :
    return pd.read_pickle('main/full_features')

full_grad_clf = load_grad()
loaded_full_vectorizer = load_vectorizer()
full_vectorizer = loaded_full_vectorizer
full_features = load_features()

filtered_tfidf_matrix = full_vectorizer.transform(filtered_transcripts_sent_df['Processed Transcript'])
X_train_filtered = pd.DataFrame(filtered_tfidf_matrix.toarray(), columns = full_vectorizer.get_feature_names_out())
X_train_filtered = X_train_filtered[full_features]
y_train_filtered = filtered_transcripts_sent_df['Funniness']

oversampler = RandomOverSampler(sampling_strategy = 'minority')
X_train_sm, y_train_sm = oversampler.fit_resample(X_train_filtered, y_train_filtered)

filtered_grad_clf = GradientBoostingClassifier(n_estimators = 40, learning_rate = 0.1, random_state = 0).fit(X_train_sm, y_train_sm)

def weighted_predictions(full_model, filtered_model, X, weight = 0.75):
    full_probs = full_model.predict_proba(X)
    filtered_probs = filtered_model.predict_proba(X)

    #determine if filtered_model has fitted to all classes
    unfitted_classes = []
    for i in range(4):
        if i not in filtered_model.classes_:
            unfitted_classes.append(i)

    #add column of zero to filtered_probs for each unfitted class
    if len(unfitted_classes) > 0:
        for c in unfitted_classes:
            filtered_probs = np.insert(filtered_probs, c, 0, axis = 1)

    adj_probs = (full_probs * weight) + (filtered_probs * (1 - weight))
    y_hat = adj_probs.argmax(axis = 1)
    return y_hat

def break_chunks(input, n_chunks):
    user_tokens = [s for s in input]
    user_token_count = len(user_tokens)
    chunk_token_count = user_token_count // n_chunks
    chunk_ranges = range(0, user_token_count, chunk_token_count)
    chunks = []

    for c in range(n_chunks):
        if c == n_chunks - 1:
            chunk = ''.join(user_tokens[chunk_ranges[c]:])
        else:
            chunk = ''.join(user_tokens[chunk_ranges[c]:chunk_ranges[c + 1]])
        chunks.append(chunk)

    return chunks

user_chunks = break_chunks(user_input, 10) 

chunk_df = pd.DataFrame({'Chunk': user_chunks})
chunk_df['Processed Chunk'] = chunk_df['Chunk'].apply(preprocess)
vec_user_input = full_vectorizer.transform(chunk_df['Processed Chunk'])
user_input_df = pd.DataFrame(vec_user_input.toarray(), columns = full_vectorizer.get_feature_names_out())
user_input_df = user_input_df[full_features]
weighted_pred = weighted_predictions(full_grad_clf, filtered_grad_clf, user_input_df)
chunk_df['Funniness'] = weighted_pred

def get_pred_output(pred):
    if pred == 0:
        return 'Neutral'
    elif pred == 1:
        return 'A little funny'
    elif pred == 2:
        return 'Moderately funny'
    elif pred == 3:
        return 'Very funny'

chunk_df['Funniness Word'] = chunk_df['Funniness'].apply(get_pred_output)

lexicon = Empath()
def get_topics(text, n_topics):
    topics_lst = []
    top_dict = lexicon.analyze(text, normalize = True)
    sorted_top_lst = sorted(top_dict.items(), key = lambda x: x[1], reverse = True)

    for top_weight in sorted_top_lst[:n_topics]:
        topics_lst.append(top_weight[0])
    return topics_lst

filtered_transcripts_df['Topics'] = filtered_transcripts_df['Cleaned Transcript'].apply(get_topics, args = (3,))

chunk_df['Topics'] = chunk_df['Processed Chunk'].apply(get_topics, args = (3,))
user_topics = get_topics(preprocess(user_input), 3)

'''
## Step 2 : Transcripts with Similar Sentiment
The dataframe below shows the transcripts that the your transcript is most similar to in terms of **sentiment** !
We filter similar transcripts by polarity and subjectivity score. 
'''

step2_df = filtered_transcripts_df.drop(columns=['Unnamed: 0', 'Date', 'Subtitle', 'Transcript', 'Cleaned Transcript', 'Topics'])


st.dataframe(step2_df)
st.write(f'Your transcript is similar to {len(step2_df.index)} transcripts from our database.')

'''
## Step 3 : Model Training and Weighted Funniness Predictions 
Before predicting the funniness level of your transcript, we need to train our model on your transcript.  
Don't worry, no actions is needed here :) !!

**What is going on behind the scene in this step ?**  
From Step 2, we have a subset of similar transcripts. We use two models for a **weighted** funniness predictions.
1. **Full model :** ML model trained on all transcripts. 
2. **Filtered model :** ML model trained on the subset of similar transcript

Finally, the models are **ensembled and weighted.** 
'''

'''
## Step 4 : Breaking into Chunks 
Okay, your input transcript may have 100000000... lines, and it's too difficult for our model to predict your funniness level accurately.  
Instead of predicting your entire transcript, why not break them into smaller chunks and we predict the funniness level of each chunk ?  
**That's much better !!** 
'''
step4_df = chunk_df.drop(columns=['Processed Chunk', 'Topics'])
st.dataframe(step4_df)


'''
## Step 5 : Topic Modelling 
We can do better than just predicting your level of funniness.  
Let's look at the top 3 topics in your 10 chunks and the transcripts you are similar to. 
'''
step5_userchunks = chunk_df.drop(columns=['Processed Chunk', 'Funniness'])

step5_similar = filtered_transcripts_df.drop(columns=['Unnamed: 0', 'Date', 'Subtitle', 'Transcript', 'Cleaned Transcript', 'Polarity', 'Subjectivity'])

st.markdown('**Top 3 topics from user input transcript chunks :**')

st.dataframe(step5_userchunks)

st.markdown('**Top 3 topics from each similar transcript  :**')

st.dataframe(step5_similar)

st.write(f'**Top 3 topics from full transcript (original, before chunking) :** {", ".join(user_topics)}')