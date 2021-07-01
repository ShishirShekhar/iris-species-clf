"""This module contains the functions"""

from sklearn.svm import SVC

def prediction(svc_model, SepalLength, SepalWidth, PetalLength, PetalWidth):
    """This function returns the preddicted value."""
    species = svc_model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
    species = species[0]
    if species == 0:
        return "Iris-setosa"
    elif species == 1:
        return "Iris-virginica"
    else:
        return "Iris-versicolor"

def m_score(X_train, y_train):
    """This function creates the model the returns the model score."""
    svc_model = SVC(kernel = 'linear')
    svc_model.fit(X_train, y_train)
    score = svc_model.score(X_train, y_train)
    return svc_model, score
