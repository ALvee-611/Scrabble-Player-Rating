# Loading Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict

import os

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

import joblib

# get_data by specifying folder_name and file_name
def get_data(folder_name, file_name):
    """
        loads the 'file_name' csv file from the 'folder_name' folder
        get_data(folder_name, file_name) : Str Str -> Pandas DataFrame 
    """
    source_path = os.path.join('..\data', folder_name)
    if file_name == 'sample_submission.csv':
        path_data = os.path.join(source_path, file_name)
        data_df = pd.read_csv(path_data)
    elif file_name == 'train.csv':
        path_data = os.path.join(source_path, file_name)
        data_df = pd.read_csv(path_data)
    elif file_name == 'games.csv':
        path_data = os.path.join(source_path, file_name)
        data_df = pd.read_csv(path_data, parse_dates = ['created_at'])
    elif file_name == 'turns.csv':
        path_data = os.path.join(source_path, file_name)
        data_df = pd.read_csv(path_data)
    else:
        path_data = os.path.join(source_path, file_name)
        data_df = pd.read_csv(path_data)
    return data_df


# Custom Transformer to add bot features

class AddBotFeatures (BaseEstimator, TransformerMixin):
    def __init__(self):
        return None
    def fit(self, X):
        return self
    def transform (self, X):
        bot_names = ['BetterBot', 'STEEBot', 'HastyBot']
        bot_data = X.loc[X['nickname'].isin(bot_names)].copy()
        bot_data.rename(columns={'nickname':'bot_nickname', 'score': 'bot_score', 'rating': 'bot_rating'}, inplace= True )
        human_data = X.loc[~X['nickname'].isin(bot_names)].copy()
        
        # Join the two dataframe
        new_df = human_data.join(bot_data.set_index('game_id'), on='game_id')

        # Move the rating column to the end
        column_to_move = new_df.pop("rating")

        new_df.insert(6, "rating", column_to_move)

        return new_df


def split_data(df):
    """Splits df into training, testing and validation sets
        split_data: Pandas DataFrame -> Pandas DataFrame, Pandas DataFrame, Pandas DataFrame, Pandas DataFrame
    """
    X_data = df.drop(columns = "rating")
    train_y = df["rating"].copy()
    X_train, X_test, y_train, y_test = train_test_split(X_data, train_y, test_size=0.3, random_state=123)
    return X_train, X_test, y_train, y_test


# PREPROCESSING PIPELINE

# Standardize the score, bot_score and bot_rating
Preprocess = ColumnTransformer([
    ('StandardScaler', StandardScaler(), [2,4,5]),
    ('oneHotEncoding', OneHotEncoder(), [3])
])


def evaluate_model(test_data,true_labels, model):
    """
        evaluate model and return the RMSE of the model
    """
    test_X = Preprocess.transform(test_data)
    predictions = model.predict(test_X)
    error = abs(predictions - true_labels)
    rmse = np.sqrt(np.mean(error))
    return rmse


def save_model_or_pipeline(model, model_name, folder):
    """
        save_model saves the model in the model folder
        save_model(model,model_name): ML model Str -> None
    """
    model_n = model_name + '.pkl'
    model_path = os.path.join('..', folder, model_n)
    joblib.dump(model, model_path)


def load_model_or_pipeline(model_name, folder_name):
    """
        load_model loads the model named 'model_name'
    """
    model_n = model_name + '.pkl'
    model_path = os.path.join('..',  folder_name, model_n)
    load_rf = joblib.load(model_path)
    return load_rf



def save_submission(submission_name, model, pipeline_name):
    """
        save_submission requires 3 inputs: submission_name which is the name of the submission file,
        model which is the name of the model to be loaded from the models folder to make the predictions and
        pipeline_name which is the name of the preprocessing pipeline to be load from the pipeline folder

        save_submission: Str Str Str -> Pandas DataFrame
    """

    sample_submission = get_data('main_data', 'sample_submission.csv')
    
    test_data = get_data('main_data', 'test.csv')

    AddBot = load_model_or_pipeline('Add_bot_features', 'pipeline')
    testing_data = AddBot.transform(test_data)
    testing_data.drop(columns = 'rating', inplace = True)
    
    # Data Pipeline
    Preprocess = load_model_or_pipeline(pipeline_name, 'pipeline')
    
    scaled_testing = Preprocess.transform(testing_data)

    rating = model.predict(scaled_testing)
    sample_submission['rating'] = rating

    file_name = submission_name + '.csv'
    prediction_path = os.path.join('../predictions', file_name)
    sample_submission.to_csv(prediction_path, index = False)

    return sample_submission