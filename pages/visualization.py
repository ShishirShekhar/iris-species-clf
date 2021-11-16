"""This modules contains data about home page"""

# Import necessary modules
import streamlit as st
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Import necessary functions
from prep_data import load_data


def app():
    """This funciton creates the home page"""
    # Add title to the page
    st.title("Graph for prediction model")

    # Load the data
    df, X, y = load_data()

    # Plot the graph
    plot_graph(X, y)


def plot_graph(X, y):
    """
    This reduce the dimension data
    and creates the scatter plot
    """
    pca = PCA(2)
    X_projected = pca.fit_transform(X)
    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    fig = plt.figure()
    plt.scatter(x1, x2, c=y, cmap='viridis')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.colorbar()
    st.pyplot(fig)