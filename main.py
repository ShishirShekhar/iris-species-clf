import streamlit as st
from prep_data import iris_df, X_train, X_test, y_train, y_test
from functions import prediction, m_score, clf_selector

# Creating the SVC model and storing the accuracy score in a variable 'score'.
svc_model, score = m_score(X_train, y_train)

# Add title widget
st.title("Iris Flower Species Prediction App")  

# Add sidbar title
st.sidebar.title("Iris Flower Species Prediction App")

# Add 4 sliders and store the value returned by them in 4 separate variables.
s_len = st.sidebar.slider("Sepal Length", float(iris_df["SepalLengthCm"].min()), float(iris_df["SepalLengthCm"].max()))
s_wid = st.sidebar.slider("Sepal Width", float(iris_df["SepalWidthCm"].min()), float(iris_df["SepalWidthCm"].max()))
p_len = st.sidebar.slider("Petal Length", float(iris_df["PetalLengthCm"].min()), float(iris_df["PetalLengthCm"].max()))
p_wid = st.sidebar.slider("Petal Width", float(iris_df["PetalWidthCm"].min()), float(iris_df["PetalWidthCm"].max()))

# Add Classifier selector
clf = st.sidebar.selectbox('Classifier',('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))
model = clf_selector(clf)

# When 'Predict' button is pushed, the 'prediction()' function must be called 
# and the value returned by it must be stored in a variable, say 'species_type'. 
# Print the value of 'species_type' and 'score' variable using the 'st.write()' function.
if st.button("Predict"):
	species_type = prediction(model, s_len, s_wid, p_len, p_wid)
	st.write("Species predicted:", species_type)
	st.write('Classifier used:', model)
	st.write("Accuracy score of this model is:", score)
