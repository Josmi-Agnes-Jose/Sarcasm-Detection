import streamlit as st
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import seaborn as sns

data=pd.read_csv('streamlit_data1.csv')
head = pd.DataFrame(data.head(10))
head = head.drop(['n_gram_prediction','author', 'subreddit', 'score', 'ups','ratio_char',
'downs', 'date', 'created_utc', 'parent_comment', 'cleaned','punc(”)',"punc(')",'label'], axis=1)
head.columns = ['Comment', 'Sentiment score','Capital words', 'Total words', 
'.', ',', '!','?', '*', 'Characters repeated','Unique Characters', 'Total Characters',
'Subreddit Ratio','Sentence length', 'Syllables per word', 'Flesch Score','Swear Words']

head1 = head[['Comment','Capital words','Total Characters','Unique Characters','Characters repeated',
'Capital words','Sentence length','Syllables per word','Flesch Score','Swear Words',
'.', ',', '!','?', '*','Subreddit Ratio','Sentiment score']]


data['abs_cc_score']=abs(data['cc_score'])

shape = data.shape
n = 16

st.set_option('deprecation.showPyplotGlobalUse', False)
def app():
    writesa='<p style="font-family:black body ; color:#000033  ; text-align:center; font-size: 60px;">Extracted Features.</p>'

    st.markdown(writesa,unsafe_allow_html=True)
    graph_sel = st.sidebar.selectbox("Select Graph",('Dataset', 'Swear Words', 'Capital Words',
    'Complexity', 'Sentiment', 'Punctuation', 'Repeated Characters'))
    
    if graph_sel=='Dataset':
        st.write("Head of the dataset")
        st.dataframe(data=head1)
        st.write("Number of Features extracted",n)
    
    elif graph_sel=='Swear Words':
        writer='<p style="font-family:black body ; color:#000033  ; text-align:left; font-size: 25px;">Swear Words.</p>'
        st.markdown(writer,unsafe_allow_html=True)
        plt.figure(figsize=[10,6])
        sns.countplot(data=data, x="SwearWord", hue="label")
        plt.xlabel( "Swear Words Used", size = 16 )
        plt.ylabel( "Number of comments", size = 16 )
        plt.xlim(-1,3)
        plt.title( "Swear Words", size = 24 )
        plt.legend(["No", "Yes"], loc="upper right", title="Sarcastic Comment")
        plt.annotate("481268    484150", (-0.35,182653), fontsize=12)
        plt.annotate("24137     21218", (0.65,50721), fontsize=12)
        plt.show()
        st.pyplot()

    elif graph_sel=='Capital Words':
        writer='<p style="font-family:black body ; color:#000033  ; text-align:left; font-size: 25px;">Capital words.</p>'
        st.markdown(writer,unsafe_allow_html=True)
        slider_sel = st.sidebar.slider("Y limit", max_value=400000, min_value=10000, value=400000, step = 25000)
        plt.figure(figsize=[10,6])
        sns.histplot(data=data, x='capital_words', hue='label', binrange=(-0.5,10.5),binwidth=1)
        plt.xlabel( "Capital word" , size = 16 )
        plt.ylim(0,slider_sel)
        plt.ylabel( "Number of comments" , size = 16 )
        plt.title( "Complete capital word" , size = 24 )
        plt.legend(["Yes", "No"], loc="upper right", title="Sarcastic Comment")
        plt.show()
        st.pyplot()

    elif graph_sel=='Repeated Characters':
        writer='<p style="font-family:black body ; color:#000033  ; text-align:left; font-size: 25px;">Repeated characters.</p>'
        st.markdown(writer,unsafe_allow_html=True)
        plt.figure(figsize=[10,6])
        a = sns.countplot(data=data, x="char_repeated", hue="label")
        a.plot()
        plt.xlabel( "Characters repeated" , size = 16 )
        plt.ylabel( "Number of comments" , size = 16 )
        plt.title( "Repeated Chaaaaaaracters" , size = 24 )
        plt.xlim(-1,3)
        plt.legend(["No","Yes"], loc="upper right", title="Sarcastic Comment")
        plt.annotate("481268    484150", (-0.30,182653), fontsize=12)
        axes.Axes.set_xticklabels(a,['False','True'])
        plt.annotate("24137     21218", (0.7,50721), fontsize=12)
        plt.show()
        st.pyplot()

    elif graph_sel=='Sentiment':
        writer='<p style="font-family:black body ; color:#000033  ; text-align:left; font-size: 25px;">Sentiment in sarcasm.</p>'
        st.markdown(writer,unsafe_allow_html=True)
        switch = st.selectbox("Polarity or Intensity", ('Polar', 'Intense'))
        slider_sel = st.sidebar.slider("Y limit", max_value=225000, min_value=25000, value=225000, step = 25000)
        slider_sel1 = st.sidebar.slider("X Range", max_value=1.1, min_value=-1.1, value=[-1.1,1.1], step = 0.1)


        if switch=='Polar':
            plt.figure(figsize=[12,6])
            sns.histplot(data=data,x='cc_score', hue='label', binrange=(-1,1), binwidth=0.1)
            plt.xlabel( "Polarity" , size = 16 )
            plt.ylabel( "Number of comments" , size = 16 )
            plt.ylim((0,slider_sel))
            plt.xlim(slider_sel1)
            plt.title( "Sentiment Score" , size = 24 )
            plt.legend(["Yes", "No"], loc="upper right", title="Sarcastic Comment")
            plt.show()
            st.pyplot()
        
        elif switch=='Intense':
            plt.figure(figsize=[8,6])
            a = sns.histplot(data=data, x='cc_score', hue='label', binrange=(0,1),binwidth=0.33)
            a.plot()
            plt.xlabel( "Intensity" , size = 16 )
            plt.ylabel( "Number of comments" , size = 16 )
            plt.xticks(ticks=[0,0.165,0.33,0.5,0.66,0.83,0.99])
            labels = [0,'Nuetral',0.33, 'Moderate',0.66,'Strong',1]
            axes.Axes.set_xticklabels(a,labels=labels)
            plt.title( "Sentiment Score" , size = 24 )
            plt.legend(["Yes", "No"], loc="upper right", title="Sarcastic Comment")
            plt.show()
            st.pyplot()

    elif graph_sel=='Punctuation':
        writer='<p style="font-family:black body ; color:#000033  ; text-align:left; font-size: 25px;">Punctuation marks!!</p>'
        st.markdown(writer,unsafe_allow_html=True)
        mark = st.selectbox("Punctuation Mark",('.',',','!','?','*'))
        slider_sel = st.sidebar.slider("Y limit", max_value=200000, min_value=50, value=93450, step = 25)

        if mark == '.':
            plt.figure(figsize=[8,6])
            sns.histplot(data=data,x='punc(.)',
            hue='label' ,
            binrange=(-0.5,10.5),
            binwidth=1)
            plt.xlabel( "Total numer of '.'s" , size = 16 )
            plt.ylabel( "Number of comments" , size = 16 )
            plt.title( "Punc(.)" , size = 24 )
            plt.ylim(0,slider_sel)
            plt.legend(["Yes", "No"], loc="upper right", title="Sarcastic Comment")
            plt.show()
            st.pyplot()
        elif mark == ',':
            plt.figure(figsize=[8,6])
            sns.histplot(data=data,x='punc(,)',
            hue='label' ,
            binrange=(-0.5,10.5),
            binwidth=1)
            plt.xlabel( "Total numer of ','s" , size = 16 )
            plt.ylabel( "Number of comments" , size = 16 )
            plt.title( "Punc(,)" , size = 24 )
            plt.ylim(0,slider_sel)
            plt.legend(["Yes", "No"], loc="upper right", title="Sarcastic Comment")
            plt.show()
            st.pyplot()
        elif mark == '!':
            plt.figure(figsize=[8,6])
            sns.histplot(data=data,x='punc(!)',
            hue='label' ,
            binrange=(-0.5,10.5),
            binwidth=1)
            plt.xlabel( "Total numer of '!'s" , size = 16 )
            plt.ylabel( "Number of comments" , size = 16 )
            plt.title( "Punc(!)" , size = 24 )
            plt.ylim(0,slider_sel)
            plt.legend(["Yes", "No"], loc="upper right", title="Sarcastic Comment")
            plt.show()
            st.pyplot()
        elif mark == '?':
            plt.figure(figsize=[8,6])
            sns.histplot(data=data,x='punc(?)',
            hue='label' ,
            binrange=(-0.5,10.5),
            binwidth=1)
            plt.xlabel( "Total numer of '?'s" , size = 16 )
            plt.ylabel( "Number of comments" , size = 16 )
            plt.title( "Punc(?)" , size = 24 )
            plt.ylim(0,slider_sel)
            plt.legend(["Yes", "No"], loc="upper right", title="Sarcastic Comment")
            plt.show()
            st.pyplot()
        elif mark == '*':
            plt.figure(figsize=[8,6])
            sns.histplot(data=data,x='punc(*)',
            hue='label' ,
            binrange=(-0.5,10.5),
            binwidth=1)
            plt.xlabel( "Total numer of '*'s" , size = 16 )
            plt.ylabel( "Number of comments" , size = 16 )
            plt.title( "Punc(*)" , size = 24 )
            plt.ylim(0,slider_sel)
            plt.legend(["Yes", "No"], loc="upper right", title="Sarcastic Comment")
            plt.show()
            st.pyplot()
    
    elif graph_sel == 'Complexity':
        writer='<p style="font-family:black body ; color:#000033  ; text-align:left; font-size: 25px;">Complexity.</p>'
        st.markdown(writer,unsafe_allow_html=True)
        minigrap = st.selectbox("Measure of Complexity", ('Flesch Score', 'Sentence Length',
        'Comment Length', 'Syllables per word'))

        if minigrap == 'Flesch Score':
            plt.figure(figsize=[16,6])
            sns.histplot(data=data,x='Flesch_score',
            hue='label' ,
            binrange=(0,100),
            binwidth=1)
            plt.xlabel( "Flesch Score" , size = 16 )
            plt.ylabel( "Number of people" , size = 16 )
            plt.title( "Flesch Score" , size = 24 )
            plt.legend(["Yes", "No"], loc="upper right", title="Sarcastic Comment")
            plt.show()
            st.pyplot()
        
        elif minigrap == 'Sentence Length':
            plt.figure(figsize=[16,6])
            sns.histplot(data=data,x='Avg_sentence_length',
            hue='label' ,
            binrange=(.5,70.5),
            binwidth=1)
            plt.xlabel( "Number of words per sentence" , size = 16 )
            plt.ylabel( "Number of comment" , size = 16 )
            plt.title( "Sentence lengths" , size = 24 )
            plt.legend(["Yes", "No"], loc="upper right", title="Sarcastic Comment")
            plt.show()
            st.pyplot()
        
        elif minigrap == 'Comment Length':
            subminigrap = st.selectbox("Measuremnts based on", ('Words', 'Characters', 'Unique Characters'))

            if subminigrap == 'Words':
                plt.figure(figsize=[16,6])
                sns.histplot(data=data,x='total_words',
                hue='label' ,
                binrange=(-0.5,80.5),
                binwidth=1)
                plt.xlabel( "Total words" , size = 16 )
                plt.ylabel( "Number of comments" , size = 16 )
                plt.title( "Total numer of words" , size = 24 )
                plt.legend(["Yes", "No"], loc="upper right", title="Sarcastic Comment")
                plt.show()
                st.pyplot()
            elif subminigrap == 'Characters':
                plt.figure(figsize=[16,6])
                sns.histplot(data=data,x='tot_chars',
                hue='label' ,
                binrange=(-0.5,200.5),
                binwidth=1)
                plt.xlabel( "Characters" , size = 16 )
                plt.ylabel( "Number of comments" , size = 16 )
                plt.title( "Total numer of Characters" , size = 24 )
                plt.legend(["Yes", "No"], loc="upper right", title="Sarcastic Comment")
                plt.show()
                st.pyplot()
            elif subminigrap == 'Unique Characters':
                plt.figure(figsize=[16,6])
                sns.histplot(data=data,x='unique_char',
                hue='label' ,
                binrange=(-0.5,50.5),
                binwidth=1)
                plt.xlabel( "Unique Characters" , size = 16 )
                plt.ylabel( "Number of comments" , size = 16 )
                plt.title( "Total numer of unique characters" , size = 24 )
                plt.legend(["Yes", "No"], loc="upper right", title="Sarcastic Comment")
                plt.show()
                st.pyplot()
        
        elif minigrap == 'Syllables per word':
            Yslider = st.sidebar.slider("Y limit", max_value=200000, min_value=50, value=179610, step=10)
            plt.figure(figsize=[8,6])
            sns.histplot(data=data,x='Avg_syllables_per_word',
            hue='label' ,
            binrange=(0,6),
            binwidth=0.3)
            plt.xlabel( "Number syllables per word" , size = 16 )
            plt.ylabel( "Number of comments" , size = 16 )
            plt.title( "Syllables per Word" , size = 24 )
            plt.ylim(0,Yslider)
            plt.legend(["Yes", "No"], loc="upper right", title="Sarcastic Comment")
            plt.show()
            st.pyplot()

