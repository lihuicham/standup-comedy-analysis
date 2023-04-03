import streamlit as st 
from empath import Empath

'''
# Demo 
'''

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