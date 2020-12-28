# README

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Associated webapp

TODO: link

## Motivation

TODO: Discussion of project goals

## TODO: Results
* Visualization One
* Visualization Two
* Visualization Three
*
* Good Performers Table Screenshot


## TODO: WebAppImages
![alt text](https://github.com/lisaquera/disaster_data/blob/[branch]/image.jpg?raw=true)
or
![GitHub Logo](/images/logo.png)

## Libraries and files
Libraries required: sys, re, pandas, NLTK, Scikit-learn, joblib, plotly, JSON,
flask, and sqlalchemy

Files:
* disaster_messages.csv and disaster_categories.csv provide the dataset of text messages labeled by FigureEight with 36 potential categories.
* DisasterResponse.db table dmessages
* Model file is drcat_model.pkl
* process.py
* training.py
* run.py
* webapp defined with


## Final Model: drcat_model.pkl
RandomForestClassifier with parameters:

## Dataset Class Compositions
The provided dataset is imbalanced, with the largest category, 'related', having 24% of total positive values, and the smallest category, 'shops', having 0.14% of total positive values. (Note that the 'child_alone' category has zero positive values.)  The top 5 categories comprise 57% of the positive values, leaving most of the others between 0.3% and 3%.
With such imbalanced language classification datasets, you are faced with the decision to either 1) alter the data with techniques like upsampling, 2) add to the process with intent recognition, or 3) try different algorithms on the existing data to create the best model. Upsampling risks changing the dataset so much that it no longer accurately reflects the true Bayesian priors in the environment being modeled. Intent recognition is borrowed from chatbot design and adds domain expertise to the process by including specific definitions to match on before attempting to predict with the model. While adding intents would normally be my choice for this problem, on the assumption that rescue workers have that domain knowledge to create good intent recognition, executing on that would be outside the scope of this project and potentially violating the grading rubric(?). Thus, my project choice was to leave the dataset as is, on the assumption that it best reflects the probabilities for messages that will be received in the future, and use GridSearching to find the best model to make predictions for that reality.


## Acknowledgements
Thank you to FigureEight for providing the dataset
