"""This modules contains data about home page"""

# Import necessary modules
import streamlit as st


def app():
    """This funciton creates the home page"""
    # Add title to the page
    st.title("Working of This Website")

    # Add types of species section
    st.subheader("Types of Iris Flower Species")

    # Add some cols for space customization
    col1, col2, col3 = st.columns([1, 6, 1])
    
    with col2:
        st.image("./images/iris_types.jpg")

    # Add prediction info
    st.subheader("How Species are predicted")
    st.image("./images/iris_c.png")