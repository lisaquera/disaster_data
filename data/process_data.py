import sys
import pandas as pd
import re
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''Create a dataframe from csv files.
    Input: messages_filepath as string pointing to csv file
        categories_filepath as string pointing to csv file
    Output: df as pd.DataFrame containing all data from csv files
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how="outer", on='id')
    return df

def clean_data(df):
    '''Convert original data to machine-learning compatible data.
    Input: df as pd.DataFrame containing all data from csv files
    Output: df as pd.DataFrame containing all data from csv files
    '''
    cats = df['categories'].astype(str).str.split(';', expand=True)
    # rename the category columns
    row = cats.head(1)
    category_colnames = row.apply(lambda x: x[0][:-2])
    cats.columns = category_colnames
    # convert category values to numbers 0,1
    for column in cats:
        # set each value to be the last character of the string
        cats[column] = cats[column].apply(lambda x: x[-1:])
        # convert column from string to numeric
        cats[column] = pd.to_numeric(cats[column])
    # drop the original categories column from `df` as now unnecessary
    df = df.drop(['categories'], axis=1)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, cats], axis=1, ignore_index=False)
    # remove any duplicate values from the training data column
    df = df.drop_duplicates(subset=['message'])
    return df

def save_data(df, database_filename):
    '''Store the cleaned data in a sql database table.
    Input: df as pd.DataFrame containing cleaned data
        database_filename as string with
    Outcome: df data is written to database table
    '''
    db_path = 'sqlite:///'+database_filename
    engine = create_engine(db_path)
    df.to_sql('dmessages', engine, if_exists='replace', index=False)



def main():
    # Unchanged from provided file
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    # Unchanged from provided file
    main()
