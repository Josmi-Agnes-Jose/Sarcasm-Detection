import streamlit as st
import numpy as np
import pandas as pd
import nltk
import Model

#nltk.download('all')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textstat.textstat import textstatistics, legacy_round
analyzer = SentimentIntensityAnalyzer()


def predict(comment,ratio,score,p):
    df = pd.DataFrame()
    comment=str(comment)
    df["comment"]=[comment]
    df["score"]=score
    #df["cleaned"]=[df["comment"].astype(str)]
    df['cc_score'] = [analyzer.polarity_scores(df.comment) ]
    df['cc_score']=[df['cc_score'][0]["compound"]]
    
    def cap_count(text):
        cap=0
        for word in text.split():
            if word.isupper():
                cap +=1
        return cap
    df["capital_words"]=[cap_count(df["comment"][0])]

    def tot_words(text):
        return(len(text.split()))
    df["total_words"]=[tot_words(df["comment"][0])]

    pun=[".",",","!","?","’","*","”"]
    def punct(text,p):
        a=0
        for i in range(0,len(text)):
            if text[i]==p:
                a+=1
        return(a)
    
    for p in pun:
        df[p]=[punct(df["comment"][0],p)]
    df=df.rename(columns={".":"punc(.)",",":"punc(,)","!":"punc(!)",
                "?":"punc(?)","’":"punc(')","*":"punc(*)","”":"punc(”)"})

    def repeat(text):
        text=text.split()
        words=[]
        for word in text:
            chars=0
            for char in word:
                if word.count(char)>=5:
                    chars+=1
            words.append(chars)
        if max(words)>0:
            return(1)
        else:
            return(0)
    df["char_repeated"]=[repeat(df["comment"][0])]

    def unique_char(text):
        chars=[]
        for i in text:
            if i not in chars:
                chars.append(i)
        return (len(chars))
    df["unique_char"]=[unique_char(df["comment"][0])]
    df["ratio_char"]=[df["unique_char"][0]/len(df["comment"][0])]
    df["tot_chars"]=[len(df["comment"][0])]

    df["ratio"]=[ratio]

    def break_sentences(text):
        a_list =nltk.tokenize.sent_tokenize(text)
        return a_list 
    def word_count(text):
        string1=text.strip()
        count=1
        for i in string1:
            if i==" ":
                count+=1
        return count
    def sentence_count(text):
        sentences = break_sentences(text)
        return len(sentences)
    def avg_sentence_length(text):
        words = word_count(text)
        sentences = sentence_count(text)
        average_sentence_length = float(words / sentences)
        return average_sentence_length
    df["Avg_sentence_length"]=[avg_sentence_length(df["comment"][0])]
    
    def syllable_count(word):
        word = word.lower()
        count = 0
        vowels = "aeiouy"
        if word[0] in vowels:
            count += 1
        for index in range(1, len(word)):
            if word[index] in vowels and word[index - 1] not in vowels:
                count += 1
        if word.endswith("e"):
            count -= 1
        if count == 0:
            count += 1
        return count

    def avg_syllables_per_word(text):
        syllable = syllable_count(text)
        words = word_count(text)
        ASPW = float(syllable) / float(words)
        return legacy_round(ASPW, 1)
    df["Avg_syllables_per_word"]=[avg_syllables_per_word(df["comment"][0])]

    def flesch_reading_ease(text):
        FRE = 206.835 - float(1.015 * avg_sentence_length(text)) -\
          float(84.6 * avg_syllables_per_word(text))
        return legacy_round(FRE, 2)

    df["Flesch_score"]=[flesch_reading_ease(df["comment"][0])]
    def swearWord(text):
        feature3=False
        Swearwords =["shit","fuck","damn","bitch","crap","piss","dick","darn",
                 "cock","pussy","asshole","fag","bastard","slut","douche",
                 "bloody","cunt","bugger","bollocks","arsehole"]
        for item in Swearwords:
            if item in text:
                feature3=True
        return feature3
    df["SwearWord"]=[swearWord(df["comment"][0])]

    df['n_gram_prediction'] = Model.tfidf_logit_pipeline.predict(df['comment'])

    df1 = df[['score', 'cc_score', 'capital_words', 'total_words', 'punc(.)',
    'punc(,)', 'punc(!)', 'punc(?)', 'punc(*)',
    'char_repeated', 'unique_char', 'ratio_char', 'tot_chars', 'ratio',
    'Avg_sentence_length', 'Avg_syllables_per_word', 'Flesch_score',
    'SwearWord', 'n_gram_prediction']]
    

    df.columns= ['The comment', 'Comment score', 'Sentiment score','Capital words', 'Total words', 
    '.', ',', '!','?', '\'','*','"', 'Characters repeated','Unique Characters','rc' ,'Total Characters',
    'Subreddit Ratio','Sentence length', 'Syllables per word', 'Flesch Score','Swear Words', 'ngram']

    df = df[['The comment','Capital words','Total Characters','Unique Characters','Characters repeated',
    'Capital words','Sentence length','Syllables per word','Flesch Score','Swear Words',
    '.', ',', '!','?', '*','Subreddit Ratio','Sentiment score']]

    df = df.transpose()
    df.columns = [['Demo Example']]


    return df,df1
data=pd.read_csv("ratiodata.csv")
write1='<p style="font-family:black body ; color:#480000  ; text-align:left; font-size: 60px;">Let us do some Predictions.</p>'
write2='<p style="font-family:black body ; color:#000033  ; text-align:center; font-size: 25px;">Here you can check whether a comment is sarcastic or not.</p>'

def app():
    st.markdown(write1,unsafe_allow_html=True)
    st.markdown(write2,unsafe_allow_html=True)
    df = pd.DataFrame()
    user_input = st.text_area("Type the comment here : ","This is a genuine compliment!!!")
    cola, colb = st.beta_columns(2)
    sratio=cola.selectbox("Select Subreddit.",data["index"])
    ratio=data.loc[data["index"] ==sratio , "ratio"]
    score=colb.number_input("Comment Score: ",max_value=10000,min_value=-10000)
    st.write("")

    write='<p style="font-family:black body ; color:#000033 ; text-align:center; font-size: 25px;">Predictions</p>'
    writea='<p style="font-family:black body ; color:#000033 ; text-align:center; font-size: 25px;">Features</p>'


    if st.button('Predict'):
        output,sample = predict(user_input,ratio,score,p="p2")
        st.markdown(write,unsafe_allow_html=True)
        col2, col3 = st.beta_columns(2)
        col2.subheader("Logistic Regression:")
        if Model.clf1.predict(sample)==0:
            col2.write("That right there, that was SARCASM")
        else:
            col2.write("This isn't sarcasm")
        output1,sample1 = predict(user_input,ratio,score,p="p1")
        col3.subheader("Random Forest:")
        if Model.clf2.predict(sample1)==0:
            col3.write("That right there, that was SARCASM")
        else:
            col3.write("This isn't sarcasm")
        st.markdown(writea,unsafe_allow_html=True)
        st.dataframe(output)

