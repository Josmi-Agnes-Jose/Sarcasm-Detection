# Sarcasm-Detection
The aim is to predict whether a given Reddit comment is sarcastic or not based on features extracted from the comments using Machine Learning algorithms. The python library, Streamlit has been used to make the data app for presenting the results.

## Dataset
The classification of comments into sarcastic and non-sarcastic is a special branch of sentiment analysis. The dataset used is the *Sarcasm on Reddit* dataset available in [Kaggle](https://www.kaggle.com/danofer/sarcasm).The dataset contains 1.1 million reddit comments along with their features which includes : score, number of up-votes and down-votes, subreddit to which the comment belongs etc..

## Features Extracted
The purpose of this project was to extract features from the orginal reddit comments and build a machine learning model that could predict whether the given comment is sarcastic or not.We extracted 19 features which are listed below :

* `cc_score` : This assigns a value between -1 and +1 to each comment indicating how negative and positive the comments are represented.
* `capital_words` : Gives the number of whole capital words present in each comment.
* `total_words` : The total number of words in the comment.
* `punc(_)` : The number of times a punctuation repeats in the comment. This is calculated for 7 different punctuations.
* `char_repeated` : This is a boolean features taking value 0 if there are no unusual repetitions of characters in the comment and takes the value 1 otherwise.
* `unique_char` : The number of distinct characters in the comment.
* `ratio_char` : The ratio of distinct characters to total number of characters in the comment.
* `tot_chars` : The total number of characters in the comment.
* `ratio` : The ratio of the number of non sarcastic comments to the number of sarcastic comments in the subreddit to which the comment belongs.
* `Avg_sentence_length`, `Avg_syllables_per_word`, `Flesch_score` : Indicating the readability of the comment.
* `SwearWord` : This is a boolean feature, which takes value 1 if swear words are present in the comment and 0 otherwise.
* `n-grams` : sequence off n items (range(1,3)).


These extracted features are concatenated into a dataset which is named as `streamlit_data1.csv` which is used in further analysis and modelling.

## Machine Learning models
*Logistic Regression* and *Random forest* were used to build the classification model. Both of them yielded an accuracy of 74% but the Random Forest model was overfitting.

## Streamlit web app
The python library streamlit is used to build the data app that displays the results of the project. The app is divided into 6 `.py` files.
* `main.py` : This contains the main code that integrates all other `.py` files in the application.
* `intro.py` : A brief introduction to the problem statement and the team behind the project.
* `Dataset.py` : Analysis of the features available in the orginal dataset. For successfully running this file, you may need the *train-balanced-sarcasm.csv* dataset which is the orginal dataset available in kaggle (link is provided in dataset section).
* `Features.py` : Analysis of features extracted. This file requires *streamlit_data1.csv.
* `Model.py` : This file contains the training of the model and also displays the results of the 2 classification models.
* `Predict.py` : This tab can be used to make predictions based on the models built. Here you can type in the comment, select the subreddit and score. Pressing the `Predict` tab will display the features extracted for the comment and the prediction by the 2 models.

The images used in the code are also provided in the `App` folder in the repository.
