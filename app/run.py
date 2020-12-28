import json
import plotly
import pandas as pd

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine

app = Flask(__name__)

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


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('dmessages', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    # get the category data
    cat_names = df.columns[4:]
    df_cats = df[cat_names]
    # What are the top 15 categories?
    cat_sums = df_cats.sum(axis=0).sort_values(ascending=False)
    top_15_cat_amts = cat_sums[:15]
    top_15_cat_names = list(top_15_cat_amts.index)
    # What categories most associated with requests?
    req = df_cats.loc[df_cats['request'] == 1]
    req_sums = req.sum(axis=0).sort_values(ascending=False)
    # intentionally exclude 'related' as uninformational and 'request' as duplicative
    top_15_req = req_sums[2:17]
    top_15_req_names = list(top_15_req.index)
    # How many messages have more than one category assigned?
    cat_counts = df_cats.apply(lambda row: sum(row[0:35]==1) ,axis=1)
    cat_vc = cat_counts.value_counts()
    # get the number of messages that have more than one category assigned (>0,1)
    cat_multi = cat_vc.iloc[2:]
    cat_multi_buckets = list(cat_multi.index)

    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=top_15_cat_names,
                    y=top_15_cat_amts
                )
            ],
            'layout': {
                'title': 'The Top 15 Categories',
                'yaxis': {
                    'title': "Messages"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top_15_req_names,
                    y=top_15_req
                )
            ],
            'layout': {
                'title': 'The Top 15 Requested Needs',
                'yaxis': {
                    'title': "Number of Requests"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=cat_multi_buckets,
                    y=cat_multi
                )
            ],
            'layout': {
                'title': 'Messages with Multiple Categories',
                'yaxis': {
                    'title': "Number of Messages"
                },
                'xaxis': {
                    'title': "Number of Categories"
                }
            }
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
