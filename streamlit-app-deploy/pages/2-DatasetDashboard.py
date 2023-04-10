import streamlit as st
import pandas as pd 
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
import string
import nltk
nltk.download('all')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer 
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet

'''
# Dataset Dashboard 
- Dataset description 
- Data Cleaning & Text Preprocessing
- Feature Selection 

## Data Collection 

We collected the data via web scraping. We scraped the a standup comedy transcript archive website called [Scraps From the Loft](https://scrapsfromtheloft.com/stand-up-comedy-scripts/).  

The initial columns that we obtained are `Comedian`, `Date`, `Title`, `Subtitle` and `Transcript`. All variables are of type `string`. 
'''

df = pd.read_csv('main/transcripts.csv')
df = df[df.columns[1:]]
st.dataframe(df.head())

col1, col2 = st.columns(2)
col1.metric("Number of Observations", "415")
col2.metric("Number of Variables", "5")

'''
## Data Cleaning & Text Preprocessing 

1. Removal of stop words, punctuations and etc. 
2. Word tokenization. 
3. Lemmatization using parts-of-speech tagging. 
4. Repeat step 1 to 3 for more data cleaning. 

'''

# pos tagging -> lemmitization 
def pos_then_lemmatize(pos_tagged_words) :
    res = []
    for pos in pos_tagged_words : 
        word = pos[0]
        pos_tag = pos[1]
        lemmatizer = WordNetLemmatizer()
        lem = lemmatizer.lemmatize(word, get_wordnet_pos(pos_tag))
        res.append(lem)
    return res



def filtering(sent) : 
    sent = sent.lower()
    sent = re.sub('\[.*?\]', '', sent)   
    sent = re.sub('[%s]' % re.escape(string.punctuation), '', sent)
    sent = re.sub('\w*\d\w*', '', sent) 
    sent = re.sub('[‘’“”…]', '', sent)
    sent = re.sub('\n', '', sent)
    words = word_tokenize(sent)
    stop_words = set(stopwords.words('english')) 
    filtered_words = [w for w in words if not w in stop_words]
    return filtered_words

def lemmatize(filtered_words) : 
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]
    return lemmatized_words

def stemming(filtered_words) : 
    stemmer = PorterStemmer()
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    return stemmed_words

def pos(filtered_words) : 
    pos_tagged_words = nltk.pos_tag(filtered_words)
    return pos_tagged_words

def pos_lemmatize(filtered_words) : 
    pos_tagged_words = nltk.pos_tag(filtered_words)
    pos_lemmatized_words = pos_then_lemmatize(pos_tagged_words)
    return pos_lemmatized_words

def join_words(words) : 
    return ', '.join(words)



'''
#### Extra 1 : Stemming vs Lemmatization
Why do we choose lemmitization over stemming ?  
Let's explain with one simple sentence. 
'''

st.code('Follows Papa as he shares about parenting his reliance on modern technology rescuing his pet pug and how his marriage has evolved over time.')

sentence = st.text_input('Input a sentence or copy from above.')
filtered_words = filtering(sentence)

st.markdown('**Lemmatization :**')
st.write(join_words(lemmatize(filtered_words)))

st.markdown('**Stemming :**')
st.write(join_words(stemming(filtered_words)))

st.markdown('*From above comparison betweeen lemmitization and stemming, we can clearly see that the words obtained from stemming doesn\'t make sense. This is because stemming **overtruncates** words due to its simplicity to return the root form of a word.*')
'''
#### Extra 2 : How do we lemmatize based on parts-of-speech ?  
'''
# define a helper function to map the pos tag to wordnet 
with st.echo() : 
    def get_wordnet_pos(treebank_tag) : 
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

'''
Reusing the simple sentence from Extra 1. 
'''

st.markdown('**Parts-of-Speech Tags :**')
st.write(pd.DataFrame(pos(filtered_words), columns=['word', 'pos tags']).transpose())

st.markdown('**POS then Lemmatize :**')
st.write(join_words(pos_lemmatize(filtered_words)))

'''
## Data Organization 
After thorough data cleaning, our data is organized in two ways : 
- Corpus : the "cleaned" `Transcript` column. 
- Document-Term Matrix : **Count Vectorizer** and **TF-IDF Vectorizer**

With more accurate predictions and more meaningful words, we chose the document-term matrix created with TF-IDF Vectorizer. 

'''

'''
## Exploratory Data Analysis (EDA) 

1. Most Common Words in all transcripts. (frequency >= 150)
2. Word Cloud for all transcripts (cleaned data)

'''
# EDA 1
st.markdown('#### Most Common Words')

most_common_words_complete = pd.read_pickle('st-files-dashboard/mostcommonwords-st.pkl')
most_common_words = [(word, freq) for word, freq in most_common_words_complete if freq >= 150]
most_common_words_df = pd.DataFrame(most_common_words, columns = ['word' , 'count'])

fig_most_common = px.bar(most_common_words_df, x='word', y='count')
fig_most_common.update_layout(
    title="Most Common Words in All Transcripts",
    xaxis_title="Word",
    yaxis_title="Count",
)
st.plotly_chart(fig_most_common)

# EDA 2
# comment out first 
st.markdown('#### Word Cloud')
st.markdown('Transcripts are cleaned, no stopwords, no most common words and etc. Word cloud should be meaningful')

sparse_tf_stop = pd.read_pickle('st-files-dashboard/sparse_tf_stop.pkl')
tf_colnames = pd.read_pickle('st-files-dashboard/tf_colnames.pkl')
tf_matrix = pd.DataFrame(sparse_tf_stop.toarray(), columns=tf_colnames)
tf_matrix = tf_matrix.transpose()

transcript_num = st.slider('Select transcript to view word cloud : ', 0, 414, 60)

wc = WordCloud().generate_from_frequencies(tf_matrix[transcript_num])

fig_wc1, ax = plt.subplots(figsize = (12, 8))
ax.imshow(wc)
plt.axis("off")

st.write('You are viewing transcript ', transcript_num)
st.write(df.loc[df.index[transcript_num]])
st.pyplot(fig_wc1)



