import streamlit as st
import pandas as pd 
import plotly.express as px

@st.cache
def load_df():
	  return pd.read_csv('/Users/lihuicham/Documents/GitHub/standup-comedy-analysis/main/transcripts.csv')

df = load_df()
df = df[df.columns[1:]]

# create a word count column
df['Comedian'] = df['Comedian'].astype('str')
df['Comedian'] = df['Comedian'].apply(lambda x: x.lower())
df['Word Count'] = df['Transcript'].str.split().str.len()



comedian_count_df = df['Comedian'].value_counts().rename_axis('Comedian').reset_index(name='counts')
comedian_count_df = comedian_count_df.iloc[1: , :]  # drop the nan row
fig_comedian = px.bar(comedian_count_df, x="Comedian", y="counts", color="counts", title="Most Common Comedians in Website")
st.plotly_chart(fig_comedian)

top_10_list = list(comedian_count_df.head(10)['Comedian'])

top_10_comedians = ", ".join(top_10_list)
st.write('The 10 most common comedians on the website are : ', top_10_comedians)

st.write('Let\'s look at their word count per standup-comedy show !')

selected_comedian = st.selectbox(
    'Comedians',
    top_10_list)

selected_df = df[(df.Comedian == selected_comedian)]

st.write(selected_comedian, 'has ', len(selected_df.index), ' shows.')


selected_df = selected_df.drop(['Date', 'Subtitle', 'Transcript'], axis=1)

st.dataframe(selected_df)
