import streamlit as st
from prep_data import X_train, X_test, y_train, y_test
from functions import prediction, m_score

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model, score = m_score(X_train, y_train)

# Add title widget
st.title("Iris Flower Species Prediction App")  

# Add 4 sliders and store the value returned by them in 4 separate variables.
s_length = st.slider("Sepal Length", 0.0, 10.0)
s_width = st.slider("Sepal Width", 0.0, 10.0)
p_length = st.slider("Petal Length", 0.0, 10.0)
p_width = st.slider("Petal Width", 0.0, 10.0)

# When 'Predict' button is pushed, the 'prediction()' function must be called 
# and the value returned by it must be stored in a variable, say 'species_type'. 
# Print the value of 'species_type' and 'score' variable using the 'st.write()' function.
if st.button("Predict"):
	species_type = prediction(svc_model, s_length, s_width, p_length, p_width)
	st.write("Species predicted:", species_type)
	st.write("Accuracy score of this model is:", score)
