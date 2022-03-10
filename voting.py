import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

class TemplateClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, weights=None, voting="soft"):
        self.estimators = estimators
        self.voting = voting
        self.classes_ = self.estimators[0].classes_
        if weights is None:
            self.weights = [1]*len(estimators)
        else:
            self.weights = weights
    
    def fit(self, X, y):
        return self

    def predict(self, X):
        y = np.zeros((X.shape[0], len(self.classes_)))
        if self.voting == "soft":
            for i, est in enumerate(self.estimators):
                y += est.predict_proba(X)*self.weights[i]
        return y