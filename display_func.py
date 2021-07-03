import streamlit as st
from functions import prediction, clf_s, svc_score, lr_score, rf_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time


def pred_page(df, X_train, y_train):
    # Get data
    s_len, s_wid, p_len, p_wid, clf = get_s_data(df)

    # More customization
    check_c = st.checkbox('More Customization')
    if check_c:
        model, score = more_cust(clf, X_train, y_train)
    else:
        model, score = clf_s(clf, X_train, y_train)
    
    p_b = st.button("Predict")
    # Predict and show values
    if p_b:
        species_type = prediction(model, s_len, s_wid, p_len, p_wid)
        progress = st.progress(0)
        loading = st.empty()
        for i in range(100):
            time.sleep(0.05)
            progress.progress(i+1)
            loading.write(f'Progess: {i+1}%')
        st.success('Predicted successfully!')
        st.success(f"Species predicted: {species_type}")
        st.image(f'{species_type + ".jpg"}', width=600)
        st.write('Classifier used:', model)
        st.write("Accuracy score of this model is:", score)
        st.warning('Also try different models!')

def more_cust(clf, X_train, y_train):
    if clf == 'Support Vector Machine':
        op = st.selectbox('Mode', ('Manual', 'Auto'))
        if op == 'Manual':
            kernel, C, gamma, degree = manual_cut()
            model, score = svc_score(X_train, y_train, kernel, C, gamma, degree)
            return model, score
        
        else:
            parameters = {'kernel': ('linear', 'rbf', 'poly'), 'C': [0.01, 1, 10, 100], 'gamma': [0.01, 0.1, 1]}
            gdc = GridSearchCV(SVR(), parameters)
            gdc.fit(X_train, y_train)
            
            model = SVR(gdc.best_estimator_)
            model.fit(X_train, y_train)
            score = model.score(X_train, y_train)
            return model, score

    elif clf == 'Logistic Regression':
        C = st.slider('C', 0.01, 100.0)
        model, score = lr_score(X_train, y_train, C)
        return model, score
    else:
        n_estimator = st.slider('N Estimator', 1, 100)
        max_depth = st.slider('Max Depth', 2, 15)
        model, score = rf_score(X_train, y_train, n_estimator, max_depth)
        return model, score

def manual_cut():
    kernel = st.selectbox('Kernel', ('linear', 'rbf', 'poly'))
    if kernel == 'linear':
        C = st.slider('C', 0.01, 100.0)
        gamma = 'scale'
        degree = 3
    elif kernel == 'rbf':
        C = st.slider('C', 0.01, 100.0)
        gamma = st.slider('gamma', 0.01, 1.0)
        degree = 3
    else:
        C = st.slider('C', 0.01, 100.0)
        gamma = st.slider('gamma', 0.01, 1.0)
        degree = st.slider('degree', 1, 6)
    return kernel, C, gamma, degree

def get_s_data(df):
    # Add 4 sliders and store the value returned by them in 4 separate variables.
    s_len = st.slider("Sepal Length", float(df["SepalLengthCm"].min()), float(df["SepalLengthCm"].max()))
    s_wid = st.slider("Sepal Width", float(df["SepalWidthCm"].min()), float(df["SepalWidthCm"].max()))
    p_len = st.slider("Petal Length", float(df["PetalLengthCm"].min()), float(df["PetalLengthCm"].max()))
    p_wid = st.slider("Petal Width", float(df["PetalWidthCm"].min()), float(df["PetalWidthCm"].max()))

    # Add Classifier selector
    clf = st.selectbox('Classifier',('Support Vector Machine', 'Logistic Regression', 'Random Forest Classifier'))
    return s_len, s_wid, p_len, p_wid, clf

def graph_p(X, y):
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
