import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List

def msX(df:pd.DataFrame, response:str)->pd.DataFrame:
    X = pd.DataFrame({'intercept':1}, index=range(df.shape[0]))
    X = pd.concat([X, df.drop([response], axis=1)], axis=1)
    return X

def msXcols(df:pd.DataFrame, covariates:List[str])->pd.DataFrame:
    X = pd.DataFrame({'intercept':1}, index=range(df.shape[0]))
    X = pd.concat([X, df[covariates].copy()], axis=1)
    return X

def msY(df:pd.DataFrame, response:str)->pd.Series:
    Y = df[response].copy()
    return Y