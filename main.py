"""
This module display the web page
"""

import streamlit as st
from prep_data import df, X_train, y_train, X, y
from display_func import pred_page, graph_p

# Add sidbar title
st.sidebar.title("Menu")

nav = st.sidebar.radio('Navigator', ('Home', 'Working', 'Prediction', 'Graph', 'Contact Us'))

if nav == 'Prediction':
	st.title("Iris Flower Species Prediction App") 
	pred_page(df, X_train, y_train)

elif nav == 'Home':
    st.image('welcome.jpg')
    st.title('Welcome to Iris Flower Species Prediction App')
    st.markdown('### This website predict the specie of iris flower with different Machine learning classificaion model on basis of given data')
    st.subheader('Data Used')
    d_check = st.checkbox('Show data used')
    if d_check:
	    st.dataframe(df, width=1000, height=300)

elif nav == 'Working':
	st.header('Working of This Website')
	st.subheader('Types of Iris Flower Species')
	st.image('iris_types.jpg', width = 800)
	st.subheader('How Species are predicted')
	st.image('iris_c.png')

elif nav == 'Graph':
	st.title('Graph for prediction model')
	graph_p(X, y)

else:
	st.balloons()
	st.header('Contact Us')
	st.markdown('''### Name:
	Shishir Shekhar''')
	st.markdown('''### Email:
	sspdav02@gmail.com''')
	st.markdown('''### GitHub: [ShishirShekhar](https://github.com/ShishirShekhar/)''')
