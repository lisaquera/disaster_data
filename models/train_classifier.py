import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet', 'stopwords'])

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib


def load_data(database_filepath):
    '''Access the dataset stored in SQL database. Note that the train_test_split is performed in main func.
    Input: database_filepath as string
    Output: X as a dataframe of messages
        Y as a dataframe of one-hot-encoded ground_truth categories
        category_names as a list of strings for the labels
    '''
    engine = create_engine('sqlite:///'+database_filepath)
    conn = engine.connect()
    df = pd.read_sql('dmessages',
                     con=engine,
                     coerce_float=False,    # keep category binaries as ints
                    )

    X = df['message']
    Y = df.drop(['id','message','original','genre'], axis=1)
    #check for null values before training
    if Y.isnull().sum().sum() > 0:
        Y = Y.fillna(value=0)
    category_names = Y.columns
    conn.close()
    return X, Y, category_names


def tokenize(text):
    '''Convert raw text string to usable tokens for ML processing.
    Inputs: text as raw string
    Outputs: list of normalized, root-word tokens as strings
    '''
    # remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    # break into word tokens
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for w in words:
        # remove filler words
        if w not in stopwords.words('english'):
            # get roots
            lem = lemmatizer.lemmatize(w)
            lem = lemmatizer.lemmatize(w, pos='v')
            #normalize
            clean_tok = lem.lower().strip()
            clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    '''Create a machine learning classification model using pipeline and grid search.
    Inputs: none
    Outputs:
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize, lowercase=False)),
        ('tfidf', TfidfTransformer(smooth_idf=False)),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator': [RandomForestClassifier(max_depth=3, n_estimators=200), KNeighborsClassifier(), AdaBoostClassifier()]
    }
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=1)
    # NOTE model.fit() performed in main() func
    return model




def evaluate_model(model, X_test, Y_test, category_names):
    '''Get model performance metrics using sklearn classification report on each category. Metrics are precision, recall, and f1-score. Accuracy is usually not a good metric for imbalanced datasets. The macro-average is recommended for prioritizing minority classes.
    (See https://datascience.stackexchange.com/questions/36862/macro-or-micro-average-for-imbalanced-class-problems)
    Input: model as created in build_model() function
        X_test as test dataset
        Y_test as test labels
        category_names as list of category strings
    Outcome: print the macro average for each metric for each category.

    Note to reviewer: in sklearn 0.20 an output_dict was added to classification_report() and my dev environment function included that.
    However, this Workspace has sklearn 0.19 so I'm just doing a print.
    Dev environment code:
    Y_pred = model.predict(X_test)
    score_dict = {}
    targets = Y_test.values
    for idx, cat in enumerate(category_names):
        cr = classification_report(targets[idx], Y_pred[idx], output_dict=True)
        cat_ma = cat+'_macro_avg'
        score_dict[cat_ma] = cr['macro avg']
    score_df = pd.DataFrame.from_dict(score_dict, orient='columns').transpose()
    score_df = score_df.drop(['support'], axis=1)
    return score_df
    '''
    Y_pred = model.predict(X_test)
    total_accuracy = (Y_pred == Y_test).mean().mean()
    print("Total accuracy across categories for the model is "+str(total_accuracy))
    targets = Y_test.values
    for idx, cat in enumerate(category_names):
        cr = classification_report(targets[idx], Y_pred[idx])
        print(cat, cr)


def save_model(model, model_filepath):
    '''Store the trained model for use in the web application.
    Input: model as trained model parameters
        model_filepath as string of path and filename for storage
    Outcome: model.pkl stored at filepath location
    '''
    joblib.dump(model, model_filepath)


def main():
    '''NOTE: unchanged from provided file
    Create a trained model using data from the sql database and store it for use in the web application.
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
