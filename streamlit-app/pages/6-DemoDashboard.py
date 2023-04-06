import streamlit as st 
import pandas as pd
from empath import Empath
from imblearn.over_sampling import SMOTE
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import string
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import GradientBoostingClassifier

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
    
'''
# Demo 
'''
st.markdown('## Step 1 : User Input Transcript')
st.markdown('Copy sample transcript [here](https://github.com/lihuicham/standup-comedy-analysis/blob/main/main/test_pipeline_transcript.txt).')
user_input = st.text_area('Input the comedy transcript you would like to predict :', height=300)
st.markdown('Press `ctrl/cmd + enter` to process input transcript for next step.')

## Demo code from 7.5-Pipeline.ipynb 
valid_transcripts_sent_df = pd.read_pickle('/Users/lihuicham/Desktop/Y2S2/BT4222/project/standup-comedy-analysis/main/chloe_valid_transcripts_sent_df')
valid_transcripts_df = pd.read_pickle('/Users/lihuicham/Desktop/Y2S2/BT4222/project/standup-comedy-analysis/main/valid_transcripts_df')
filtered_transcripts_df = filter_by_sentiment(valid_transcripts_df, user_input)
filtered_transcripts_sent_df = valid_transcripts_sent_df[valid_transcripts_sent_df['Title'].isin(filtered_transcripts_df['Title'])]

vectorizer = TfidfVectorizer()
filtered_tfidf_matrix = vectorizer.fit_transform(filtered_transcripts_sent_df['Processed Transcript'])
X_train_filtered = pd.DataFrame(filtered_tfidf_matrix.toarray(), columns = vectorizer.get_feature_names_out())
y_train_filtered = filtered_transcripts_sent_df['Funniness']

smote = SMOTE(k_neighbors = 1)
X_train_sm, y_train_sm = smote.fit_resample(X_train_filtered, y_train_filtered)

filtered_grad_clf = GradientBoostingClassifier(n_estimators = 80, learning_rate = 0.01, random_state = 0).fit(X_train_sm, y_train_sm)

#preprocess 'user_input' and use the vectorizer to transform it
processed_user_input = preprocess(user_input)
vec_user_input = vectorizer.transform([processed_user_input])
user_input_df = pd.DataFrame(vec_user_input.toarray(), columns = vectorizer.get_feature_names_out())

#make prediction of funniness of 'user_input' 
filtered_pred = filtered_grad_clf.predict(user_input_df)[0]

#get model that is pre-trained on all valid transcripts, and also the vectorizer, and features
full_grad_clf = pickle.load(open('/Users/lihuicham/Desktop/Y2S2/BT4222/project/standup-comedy-analysis/main/trained_grad_model', 'rb'))
full_vectorizer = pickle.load(open('/Users/lihuicham/Desktop/Y2S2/BT4222/project/standup-comedy-analysis/main/full_vectorizer', 'rb'))
full_features = pickle.load(open('/Users/lihuicham/Desktop/Y2S2/BT4222/project/standup-comedy-analysis/main/full_features', 'rb'))

#transform processed 'user_input' using full_vectorizer, select the best 10000 features, and make prediction of funniness using full_grad_clf
fullvec_user_input = full_vectorizer.transform([processed_user_input])
full_user_input_df = pd.DataFrame(fullvec_user_input.toarray(), columns = full_vectorizer.get_feature_names_out())
full_user_input_df = full_user_input_df[full_features]
full_pred = full_grad_clf.predict(full_user_input_df)[0]


#function to output funniness prediction in word form
def get_pred_output(pred):
    if pred == 0:
        return 'Neutral'
    elif pred == 1:
        return 'A little funny'
    elif pred == 2:
        return 'Moderately funny'
    elif pred == 3:
        return 'Very funny'
    
st.write('Funniness based on similar transcripts: ', get_pred_output(filtered_pred))
st.write('Funniness based on all transcripts available: ', get_pred_output(full_pred))

st.markdown('## Step 2 : Transcripts of similar sentiments')
filtered_transcripts_df.rename( columns={'Unnamed: 0':'Transcript Index'}, inplace=True )
st.dataframe(filtered_transcripts_df)




'''
# ## Topic Modelling
# We are using the [**Empath**](https://github.com/Ejhfast/empath-client) library for topic modelling. 

# Let's try this out ! 
# '''

# sent = st.text_input('Here\'s a sample sentence.', 'He hit the other person.')

# lexicon = Empath()

# def demo(sent) : 
#     lex_dict_test = lexicon.analyze(sent, normalize=True)
#     topics = sorted(lex_dict_test, key=lex_dict_test.get, reverse=True)[:5]
#     return ', '.join(topics)

# st.markdown('**Top 5 Topics :**')
# st.write(demo(sent))