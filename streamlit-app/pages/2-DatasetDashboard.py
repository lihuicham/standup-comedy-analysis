import streamlit as st
import pandas as pd 
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

'''
# Dataset Dashboard 
- Dataset description 
- Data Cleaning & Text Preprocessing
- Feature Selection 

## Data Collection 

We collected the data via web scraping. We scraped the a standup comedy transcript archive website called [Scraps From the Loft](https://scrapsfromtheloft.com/stand-up-comedy-scripts/).  

The initial columns that we obtained are `Comedian`, `Date`, `Title`, `Subtitle` and `Transcript`. All variables are of type `string`. 
'''

df = pd.read_csv('/Users/lihuicham/Desktop/Y2S2/BT4222/project/standup-comedy-analysis/transcripts.csv')
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

'''
#### Extra 1 : Stemming vs Lemmatization
Why do we choose lemmitization over stemming ?  
Let's explain with one simple sentence. 
'''

'''
#### Extra 2 : How we lemmatize based on parts-of-speech ?  
Reusing the simple sentence from Extra 1. 
'''

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
2. Word Cloud for 6 random transcripts (cleaned data)

'''
# EDA 1
st.markdown('#### Most Common Words')

most_common_words_complete = pd.read_pickle('/Users/lihuicham/Desktop/Y2S2/BT4222/project/standup-comedy-analysis/main/pickle/mostcommonwords-st.pkl')
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
st.markdown('#### Word Cloud')
st.markdown('Transcripts are cleaned, no stopwords, no most common words and etc. Word cloud should be meaningful')

tf_matrix = pd.read_pickle('/Users/lihuicham/Desktop/Y2S2/BT4222/project/standup-comedy-analysis/main/pickle/tfidf_stop.pkl')
tf_matrix = tf_matrix.transpose()

transcript_num = st.slider('Select transcript to view word cloud : ', 0, 414, 60)

wc = WordCloud().generate_from_frequencies(tf_matrix[transcript_num])

fig_wc1, ax = plt.subplots(figsize = (12, 8))
ax.imshow(wc)
plt.axis("off")
st.pyplot(fig_wc1)



