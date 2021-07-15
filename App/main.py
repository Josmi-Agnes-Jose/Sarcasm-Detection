import intro
import Dataset
import Features
import Model
import Predict
import streamlit as st
import numpy as np
import pandas as pd

# st.set_page_config(layout="wide")

PAGES = {
    "Introduction":intro,
    "Dataset": Dataset,
    "Features": Features,
    "Model":Model,
    "Predict":Predict
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()

