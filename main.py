"""
This module display the web page
"""
# Import necessary modules
import streamlit as st

# Import pages
from pages import home, working, prediction, visualization, about

# Set app configuration
st.set_page_config(
	page_title="Iris Species Classifier",
	page_icon="random",
	layout="centered",
	initial_sidebar_state="auto"
)

# Create dict of pages
pages = {
	"Home": home,
	"Visulization": visualization,
	"Prediction": prediction,
	"Working": working,
	"About": about
}

# Add sidbar title
st.sidebar.title("Menu")

# Add page navigator
page = st.sidebar.radio('Navigator', pages.keys())

# Create the page selected by the user
pages[page].app()
