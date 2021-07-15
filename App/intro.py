import streamlit as st
import numpy as np
import pandas as pd

original_title = '<p style="font-family:black body; color:#400000 ;  text-align:center; font-size: 50px;">Are you being sarcastic ???</p>'
write2='<p style="font-family:black body ; color:Black ; text-align:left; font-size: 20px;">Sarcasm refers to the use of words that mean the opposite of what you really want to say, especially in order to insult someone, or to show irritation, or just to be funny.</p>'
write3='<p style="font-family:black body ; color:Black ; text-align:left; font-size: 20px;">Aim : We have the Reddit Sarcasm Detection Dataset and our goal is to predict whether a given comment is sarcastic or not.</p>'
ex='<p style="font-family:black body ; color:yellow ; text-align:right; font-size: 18px;">Bites Makes Right</p>'

def app():
    st.markdown(original_title, unsafe_allow_html=True)
    st.markdown(ex,unsafe_allow_html=True)
    my_expander = st.beta_expander("Team members :")
    my_expander.write('Josmi Agnes Jose, 20BDA27| Megha Roy, 20BDA41| Ajay Bhargav, 20BDA64| Abraham G.K, 20BDA20')
    st.markdown(write2,unsafe_allow_html=True)
    st.markdown(write3,unsafe_allow_html=True)
    st.write("")
    col1, col2, col3 = st.beta_columns(3)
    with col1:
        st.write("")
    with col2:
        st.image(
            "https://tenor.com/view/selena-gomez-its-called-sarcasm-look-it-up-sarcasm-sarcastic-gif-5134882.gif",
            width=500, 
        )
    with col3:
        st.write("")
    
