"""This module prepares the data"""

# Import necessary modules
import pandas as pd
import streamlit as st

@st.cache()
def load_data():
    """This function returns pre-processed data"""
    # Load the dataset.
    df = pd.read_csv("./data/iris-species.csv")

    # Add a column in the Iris DataFrame to resemble the non-numeric 'Species' column as numeric using the 'map()' function.
    # Create the numeric target column 'Label' to 'iris_df' using the 'map()' function.
    df['Label'] = df['Species'].map({'Iris-setosa': 0, 'Iris-virginica': 1, 'Iris-versicolor':2})

    # Create a model for Support Vector classification to classify the flower types into labels '0', '1', and '2'.

    # Create features and target DataFrames.
    X = df[['SepalLengthCm','SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
    y = df['Label']

    return df, X, y