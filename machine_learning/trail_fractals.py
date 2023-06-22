"""This is the script that will be run once a day to retrain the model using a grid search on the latest data. once a
week it will also run a sequential floating backwards feature selection to update the feature list, and once a month it
will do a full search of williams fractal params as well"""

import pandas as pd
import numpy as np
from pathlib import Path
import indicators as ind
import features
import binance_funcs as funcs
import entry_modelling as em
import plotly.express as px
import plotly.graph_objects as go
from pprint import pprint
import time
from itertools import product
from xgboost import XGBClassifier

from sklearnex import get_patch_names, patch_sklearn, unpatch_sklearn
patch_sklearn()

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, make_scorer
from mlxtend.feature_selection import SequentialFeatureSelector as SFS



scorer = make_scorer(fbeta_score, beta=0.333, zero_division=0)
selector = SFS(estimator=model, k_features=15, forward=True, floating=True, verbose=2, scoring=scorer, n_jobs=-1)