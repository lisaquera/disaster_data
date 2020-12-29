# README

## Associated webapp to view output of these files:

Current: https://view6914b2f4-3001.udacity-student-workspaces.com/


### Instructions for running these files in local environment:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves it for use
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        (Please note that train_classifier takes an average of 110min to run on a single processor.)

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Motivation

This project is motivated by the need for disaster relief organizations to assign messages from the public to the correct need category.  The dataset used to train a natural language classifier is comprised of two csv files that have sample messages and their associated category assignments.  The end product is a web application that 1) displays graphics to help aid workers prioritize categories and 2) allows the public to enter their message to see if it will be correctly assigned their priority categories. If not, they could refine the text until it does.

## Results
### Data analysis
* Visualization One: What are the top 15 categories?
* Visualization Two: Which categories are most associated with requests for help?
* Visualization Three: How many messages have more than one category assigned?
### Model performance
* The currently saved overall model accuracy is 94%.


## TODO: WebAppImages

![Web app dashboard screenshot top categories](https://github.com/lisaquera/disaster_data/blob/main/images/DR_dashboard_top_cats.png?raw=true)
![Web app dashboard screenshot most requested categories](https://github.com/lisaquera/disaster_data/blob/main/images/DR_dashboard_most_needed.png?raw=true)
![Web app dashboard screenshot multiple category messages](https://github.com/lisaquera/disaster_data/blob/main/images/DR_dashboard_multi_cats.png?raw=true)
![Web app dashboard screenshot category predictions from query text](https://github.com/lisaquera/disaster_data/blob/main/images/DR_category_predictions.png?raw=true)

## Libraries and files
Libraries required: sys, re, pandas, NLTK, Scikit-learn, joblib, plotly, JSON,
flask, and sqlalchemy

Files:
* disaster_messages.csv and disaster_categories.csv provide the dataset of text messages labeled by FigureEight with 36 potential categories.
* DisasterResponse.db table dmessages
* process_data.py
* training_classifier.py
* run.py
* webappUI is defined with master.html and go.html in templates
* requirements.txt is a pip freeze from the Workspace
* Model file classifier.pkl is larger than github limit of 25mb so not included


## Dataset Class Compositions
The provided dataset is imbalanced, with the largest category, 'related', having 24% of total positive values, and the smallest category, 'shops', having 0.14% of total positive values. (Note that the 'child_alone' category has zero positive values.)  The top 5 categories comprise 57% of the positive values, leaving most of the others between 0.3% and 3%.

With such imbalanced language classification datasets, you are faced with the decision to either 1) alter the data with techniques like upsampling, 2) add to the process with intent recognition, or 3) try different algorithms on the existing data to create the best model. Upsampling risks changing the dataset so much that it no longer accurately reflects the true Bayesian priors in the environment being modeled. Intent recognition is borrowed from chatbot design and adds domain expertise to the process by including specific definitions to match on before attempting to predict with the model. While adding intents would normally be my choice for this problem, on the assumption that rescue workers have that domain knowledge to create good intent recognition, executing on that would be outside the scope of this project and potentially violating the grading rubric(?). Thus, my project choice was to leave the dataset as is, on the assumption that it best reflects the probabilities for messages that will be received in the future, and use GridSearching to find the best model to make predictions for that reality. In this case, GridSearch chose the AdaBoostClassifier over RandomForestClassifier and KNeighborsClassifier.

Metrics used are precision, recall, and f1-score as accuracy is usually not a good metric for imbalanced datasets. I selected the macro-average from the classification report as recommended for prioritizing minority classes.
(See https://datascience.stackexchange.com/questions/36862/macro-or-micro-average-for-imbalanced-class-problems)


## Acknowledgements
Thank you to FigureEight for providing the dataset
