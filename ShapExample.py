import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import shap


boston = load_boston()
regr = pd.DataFrame(boston.data)
regr.columns = boston.feature_names
regr['MEDV'] = boston.target

X = regr.drop('MEDV', axis = 1)
Y = regr['MEDV']

fit = LinearRegression().fit(X, Y)

explainer = shap.LinearExplainer(fit, X, feature_dependence = 'independent')

shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values, X, plot_size=[15, 10])
