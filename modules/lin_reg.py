import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.stats.outliers_influence \
import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm

#plots
def abline(ax:plt.Axes, m, b, *args, **kwargs):
    xlim = ax.get_xlim()
    ylim = [m*xlim[0]+b, m*xlim[1] + b]
    ax.plot(xlim, ylim, *args, **kwargs)

def abfitline(ax:plt.Axes, model:RegressionResults, *args, **kwargs):
    abline(ax, model.params.iloc[1], model.params.iloc[0], *args, **kwargs)

def abresidues(ax:plt.Axes, model:RegressionResults, *args, **kwargs):
    ax.scatter(model.fittedvalues, model.resid)
    ax.set_xlabel('Fitted value')
    ax.set_ylabel('Residual')
    ax.axhline(0,*args, **kwargs)

#fit and predict
def fitOne(df:pd.DataFrame, covariate:str, response:str)->RegressionResults:
    X = pd.DataFrame({'intercept':1}, index=range(df.shape[0]))
    X = pd.concat([X, df[covariate].copy()], axis=1)
    Y = df[response].copy()
    return sm.OLS(Y, X).fit()

def fit(X, Y)->RegressionResults:
    return sm.OLS(Y, X).fit()

def predict(model:RegressionResults, X):
    return model.get_prediction(X).predicted_mean

def predict_intv(model:RegressionResults, X, alpha=0.05, obs=True):
    return model.get_prediction(X).conf_int(alpha=alpha, obs=obs)

#model evaluation

def leverage(model:RegressionResults):
    pass

def vif (X:pd.DataFrame) -> pd.DataFrame:
    vals = [VIF(X, i) for i in range(0, X.shape)]
    return pd.DataFrame({'vif':vals}, index=X.columns)
