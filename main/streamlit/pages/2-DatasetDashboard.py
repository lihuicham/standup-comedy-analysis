import streamlit as st
import pandas as pd 

'''
# Dataset Dashboard 
- Dataset description 
- Data Cleaning & Text Preprocessing
- Feature Selection 

## Data Collection 

We collected the data via web scraping. We scraped the a standup comedy transcript archive website called [Scraps From the Loft](https://scrapsfromtheloft.com/stand-up-comedy-scripts/).  

The initial columns that we obtained are `Comedian`, `Date`, `Title`, `Subtitle` and `Transcript`. All variables are of type `string`. 
'''

df = pd.read_csv('/Users/lihuicham/Desktop/Y2S2/BT4222/project/standup-comedy-analysis/main/transcripts.csv')
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
After thorough data cleaning, our dats is organized in two ways : 
- Corpus : the "cleaned" `Transcript` column. 
- Document-Term Matrix : **Count Vectorizer** and **TF-IDF Vectorizer**

With more accurate predictions and more meaningful words, we chose the document-term matrix created with TF-IDF Vectorizer. 

## Exploratory Data Analysis (EDA) 


'''
