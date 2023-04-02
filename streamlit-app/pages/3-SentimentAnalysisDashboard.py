import streamlit as st
import pandas as pd 
import plotly.express as px
from textblob import TextBlob
from empath import Empath

'''
# Sentiment Analysis Dashboard

## TextBlob Demo 
Let's get familiar on how `TextBlob` module gives polarity and subjectivity scores to sentences of different sentiments. 
'''
pol_score_sample = lambda x : TextBlob(x).sentiment.polarity
sub_score_sample = lambda x : TextBlob(x).sentiment.subjectivity

option = st.selectbox(
    'Select sentence',
    ('I love to cheer.', 'I hate to cheer.', 'I am okay with cheering.'))

st.write('Polarity Score : ', pol_score_sample(option))
st.write('Subjectivity Score : ', sub_score_sample(option))

st.markdown('You can try out TextBlob module by inputting any sentence in the text field below !')

pol_score_user = lambda x : TextBlob(x).sentiment.polarity
sub_score_user = lambda x : TextBlob(x).sentiment.subjectivity

user_input = st.text_input('What sentence would you like to try ?')
st.write('Your sentence\'s polarity score : ', pol_score_user(user_input))
st.write('Your sentence\'s subjectivity score : ', sub_score_user(user_input))

'''
## Sentiment Analysis of Transcripts 
'''

df_textblob = pd.read_pickle('/Users/lihuicham/Desktop/Y2S2/BT4222/project/standup-comedy-analysis/main/pickle/sentiments_textblob.pkl')
st.dataframe(df_textblob.head())

# scatterplot of polarity and subjectivity

fig_scatter = px.scatter(df_textblob, x="Polarity_Score", y="Subjectivity_Score", 
                         custom_data=df_textblob[['Comedian', 'Title']],
                         )

fig_scatter.update_traces(
    hovertemplate="<br>".join([
        "Polarity Score : %{x}",
        "Subjectivity Score : %{y}",
        "Comedian : %{customdata[0]}",
        "Show Title : %{customdata[1]}",
    ])
)


fig_scatter.update_layout(
    title="Polarity and Subjectivity Scores of Transcripts",
    xaxis_title="<--- Negative -------- Positive --->",
    yaxis_title="<-- Facts -------- Opinions -->",
)
st.plotly_chart(fig_scatter)

'''
## Topic Modelling
We are using the [**Empath**](https://github.com/Ejhfast/empath-client) library for topic modelling. 

Let's try this out ! 
'''

sent = st.text_input('Here\'s a sample sentence.', 'He hit the other person.')

lexicon = Empath()

def demo(sent) : 
    lex_dict_test = lexicon.analyze(sent, normalize=True)
    topics = sorted(lex_dict_test, key=lex_dict_test.get, reverse=True)[:5]
    return ', '.join(topics)

st.markdown('**Top 5 Topics :**')
st.write(demo(sent))

