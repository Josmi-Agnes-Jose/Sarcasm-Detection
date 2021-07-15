import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score,classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer


dataset=pd.read_csv("streamlit_data1.csv")
datas1=dataset.drop(['label', 'comment', 'author', 'subreddit', 'ups',
                    'downs', 'date', 'created_utc', 'parent_comment',
                    'cleaned',"punc(')", 'punc(”)'], axis=1)

#min_max_scaler = preprocessing.MinMaxScaler()
#X_train = min_max_scaler.fit_transform(X_train)
#X_test = min_max_scaler.fit_transform(X_test)

train_texts, valid_texts, y_train, y_valid = \
        train_test_split(dataset['comment'], 
                        dataset['label'], 
                        random_state=17)

# build bigrams, put a limit on maximal number of features
# and minimal word frequency
tf_idf = TfidfVectorizer(ngram_range=(1, 3), max_features=50000, min_df=2)
# multinomial logistic regression a.k.a softmax classifier
logit = LogisticRegression(C=1, n_jobs=4, solver='lbfgs', 
                        random_state=17, verbose=1)
# sklearn's pipeline
tfidf_logit_pipeline = Pipeline([('tf_idf', tf_idf), 
                                ('logit', logit)])
tfidf_logit_pipeline.fit(train_texts, y_train)

X_train, X_test, y_train, y_test = train_test_split(datas1, dataset["label"], test_size=0.2, random_state=1234456)


clf1=LogisticRegression(max_iter=500,random_state=1234456)
clf2= RandomForestClassifier(n_estimators=50,random_state=1234456)


clf1.fit(X_train,y_train)
y_pred1=clf1.predict(X_test)
clf2.fit(X_train,y_train)
y_pred2=clf2.predict(X_test)

report1 = classification_report(y_test, y_pred1, output_dict=True)
df1 = pd.DataFrame(report1).transpose()
report2 = classification_report(y_test, y_pred2, output_dict=True)
df2 = pd.DataFrame(report2).transpose()

a1=accuracy_score(y_train,clf1.predict(X_train))
a2=accuracy_score(y_train,clf2.predict(X_train))

cm1=confusion_matrix(y_test,y_pred1)
cm2=confusion_matrix(y_test,y_pred2)


def app():
    writelr='<p style="font-family:black body ; color:#000033  ; text-align:left; font-size: 25px;">Logistic Regression</p>'
    writerf='<p style="font-family:black body ; color:#000033  ; text-align:left; font-size: 25px;">Random Forest Regression</p>'

    writesm='<p style="font-family:black body ; color:#000033  ; text-align:center; font-size: 60px;">Machine Learning Model.</p>'
    st.markdown(writesm,unsafe_allow_html=True)
    classifier_name=st.sidebar.selectbox("Select Classifier",("Logistic Regression","Random Forest Classifier"))

    if classifier_name=="Logistic Regression":
        st.markdown(writelr,unsafe_allow_html=True)
        report=df1
        acc=a1
        cm=cm1
    elif classifier_name=="Random Forest Classifier": 
        st.markdown(writerf,unsafe_allow_html=True)           
        report=df2
        acc=a2
        cm=cm2
    st.write('')
    st.write(report)
    st.markdown('##')
    st.write("Train score : ",acc)
    st.markdown('##')
    st.write('Confusion matrix: ', cm)
        

    
    