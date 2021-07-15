import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.axes as axes
from PIL import Image

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer

data=pd.read_csv("train-balanced-sarcasm.csv",low_memory=False)
X=data
y=data.label
shape=X.shape
head=X.head(20)
n=len(np.unique(y))

data["cleaned"]=data["comment"].astype(str)
def format(text):
    text = re.sub(r'https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\<a href', ' ', text)
    text = re.sub(r'&amp;', '', text) 
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', text)
    text = re.sub(r'<br />', ' ', text)
    text = re.sub(r'\'', ' ', text)
    return(text)
data["cleaned"]=data["cleaned"].apply(format)

stop_words=set(stopwords.words('english'))
def remove(text):
    text=str(text)
    word=word_tokenize(text)
    return ([w for w in word if not w in stop_words])
data["cleaned"]=data["cleaned"].apply(remove)

def joins(a):
    text=" ".join(a)
    return text
data["cleaned"]=data["cleaned"].apply(joins)


temp1 = pd.DataFrame(data.subreddit.value_counts())
temp2 = pd.DataFrame(temp1.head())
temp2 = temp2.reset_index()


def app():
    writesl='<p style="font-family:black body ; color:#000033  ; text-align:center; font-size: 50px;">Reddit Sarcasm Dataset.</p>'
    st.markdown(writesl,unsafe_allow_html=True)
    sidebar=st.sidebar.selectbox("Select Graph",('Dataset','Comments','Subreddit', 'Wordcloud'))
    if sidebar=='Dataset':
        st.write('Let us understand the dataset.')
        st.write("Shape of the dataset",shape)
        st.write("Head of the dataset",head)
        st.write("Number of classes",n)

    elif sidebar=='Comments':
        writer='<p style="font-family:black body ; color:#000033  ; text-align:center; font-size: 25px;">Count of Sarcastic and Nonsarcastic Comments</p>'
        st.markdown(writer,unsafe_allow_html=True)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.figure(figsize=(4,2))
        
        a=sns.countplot(data=data,x="label" )
        a.plot()
        plt.xlabel("Comments")
        plt.ylabel("No. of Comments")
        plt.xlim(-1,3)
        axes.Axes.set_xticklabels(a,("Non-Sarcastic", "Sarcastic"),rotation=45)
        plt.show()
        st.pyplot()

    
    elif sidebar=='Subreddit':
        writes='<p style="font-family:black body ; color:#000033  ; text-align:center; font-size: 25px;">The proportion of comments for top 5 Subreddits in the dataset.</p>'

        # cap1= st.selectbox("Subreddit",("Top 5 Subreddits","Subreddits v\s Comments"))
        # if cap1=="Top 5 Subreddits":
        st.markdown(writes,unsafe_allow_html=True)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.xlabel("SubReddits")
        plt.ylabel("Total")
        sns.barplot(data=temp2, y='index', x='subreddit', orient='h')
        st.pyplot()
        

    elif sidebar=="Wordcloud":
        writesj='<p style="font-family:black body ; color:#000033  ; text-align:center; font-size: 25px;">Wordcloud for comments.</p>'
        st.markdown(writesj,unsafe_allow_html=True)
        cap= st.selectbox("Sarcastic or Non-Sarcastic",("Sarcastic","Non Sarcastic"))

        if cap=="Sarcastic":
            image = Image.open('sarcastic_cloud.png')
            st.image(image)

        elif cap=="Non Sarcastic":
            image = Image.open('nonsarcastic_cloud.png')
            st.image(image)



 
          

    
