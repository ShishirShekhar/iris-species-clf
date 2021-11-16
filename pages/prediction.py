"""This modules contains data about home page"""

# Import necessary modules
import streamlit as st
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import time

# Import necessary functions.
from prep_data import load_data


def app():
    """This funciton creates the home page"""
    # Add title to the page
    st.title("Iris Flower Species Prediction") 

    # Load the data
    df, X, y = load_data()

    # Get feature input from the user
    feat_list = feat_input(df)

    # Train model select by the user
    model, score = model_input(X, y)
    
    # Predict and show values
    if st.button("Predict"):
        # Get the type of species
        species = prediction(model, feat_list)
        
        # Add progress bar
        progress = st.progress(0)
        loading = st.empty()

        # Increase the value progress bar
        for i in range(100):
            time.sleep(0.005)
            progress.progress(i+1)
            loading.write(f"Progess: {i+1}%")
        
        # Show the predicted value
        st.success("Predicted successfully!")
        st.success(f"Species predicted: {species}")

        # Add image of predicted value
        species_image = "./images/" + species + ".jpg"
        st.image(species_image, width=600)

        # Write about the model used
        st.write("Classifier used:", model)
        st.write("Accuracy score of this model is:", score)
        st.warning("Also try different models!")


def feat_input(df):
    """
    This function takes the input from the user
    and return list of those values
    """
    # Add 4 sliders and store the value returned by them in 4 separate variables.
    SepalLength = st.slider("Sepal Length", float(df["SepalLengthCm"].min()), float(df["SepalLengthCm"].max()))
    SepalWidth = st.slider("Sepal Width", float(df["SepalWidthCm"].min()), float(df["SepalWidthCm"].max()))
    PetalLength = st.slider("Petal Length", float(df["PetalLengthCm"].min()), float(df["PetalLengthCm"].max()))
    PetalWidth = st.slider("Petal Width", float(df["PetalWidthCm"].min()), float(df["PetalWidthCm"].max()))

    # Create list of features
    feat_list = [SepalLength, SepalWidth, PetalLength, PetalWidth]

    return feat_list

def model_input(X, y):
    """
    This funciton take input from the user for the model and returns that trained model
    """
    # Add Classifier selector
    clf = st.selectbox("Classifier", ("Logistic Regression", "Support Vector Machine", "Random Forest Classifier"))

    if (clf == "Logistic Regression"):
        return train_lr(X, y)
    elif (clf == "Support Vector Machine"):
        return train_svc(X, y)
    elif (clf == "Random Forest Classifier"):
        return train_rf_clf(X, y)


@st.cache()
def train_lr(X, y):
    """This function process rf_clf model"""
    log_reg = LogisticRegression(
                C=100, class_weight=None, dual=False, fit_intercept=True,
                intercept_scaling=1, l1_ratio=None, max_iter=100,
                multi_class='auto', n_jobs=None, penalty='l2',
                random_state=None, solver='newton-cg', tol=0.0001, verbose=0,
                warm_start=False
    )
    log_reg.fit(X, y)
    score = log_reg.score(X, y)
    return log_reg, score


@st.cache()
def train_svc(X, y):
    """This function process svc model"""
    svc_model = SVC(
                C=0.1, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
                decision_function_shape='ovr', degree=3, gamma=0.1, kernel='poly',
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False
    )
    svc_model.fit(X, y)
    score = svc_model.score(X, y)
    return svc_model, score


def train_rf_clf(X, y):
    """This function process rf_clf model"""
    rf_clf = RandomForestClassifier(
            bootstrap=True, ccp_alpha=0.0, class_weight=None,
            criterion='gini', max_depth=110, max_features=2,
            max_leaf_nodes=None, max_samples=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=5, min_samples_split=8,
            min_weight_fraction_leaf=0.0, n_estimators=100,
            n_jobs=None, oob_score=False, random_state=None,
            verbose=0, warm_start=False
    )
    rf_clf.fit(X, y)
    score = rf_clf.score(X, y)
    return rf_clf, score


def prediction(model, feat_list):
    """This function returns the preddicted value."""
    species = model.predict([feat_list])
    species = species[0]
    if species == 0:
        return "Iris-setosa"
    elif species == 1:
        return "Iris-virginica"
    else:
        return "Iris-versicolor"