# Loading Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict

import os

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import train_test_split

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