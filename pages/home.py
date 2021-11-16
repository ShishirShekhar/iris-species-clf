"""This modules contains data about home page"""

# Import necessary modules
import streamlit as st

# Import necessary functions
from prep_data import load_data


def app():
    """This funciton creates the home page"""
    # Add title to page
    st.title('Welcome to Iris Flower Species Prediction App')

    # Add Image to pages
    st.image('./images/welcome.jpg')

    # Add breif description about the app
    st.markdown("""This website predicts the species of iris flower with different Machine learning classificaion model.""")

    # Show the data which is used
    st.subheader("Data Used")

    # Load the data
    df, X, y = load_data()

    if st.checkbox("Show data used"):
	    st.dataframe(df)