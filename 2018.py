"""
@author: Mohsen
ML+APSIM for Corn Yield Prediction
"""

import numpy as np
import pandas as pd
import random
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler
from sklearn import linear_model
from sklearn.linear_model import Lasso, ElasticNet, Ridge, LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score, cross_val_predict, cross_validate, KFold
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
from sklearn.feature_selection import RFE
import warnings
from scipy.io import loadmat
from sklearn.model_selection import LeavePGroupsOut, GridSearchCV, GroupKFold
from hyperopt import STATUS_OK
from hyperopt import hp
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import fmin
import os
from pathlib import Path


warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', 500)
np.random.seed(1369)
population = loadmat('INFO_POPULATION.mat')['INFO_POPULATION']
progress = loadmat('INFO_PROGRESS.mat')['INFO_PROGRESS']
soil = loadmat('INFO_SOIL.mat')['INFO_SOIL']
Yield = pd.DataFrame(loadmat('INFO_Yield.mat')['INFO_Yield'], columns =['year', 'state', 'county', 'yield'])

weather = pd.read_parquet('main_weather_final.parquet')
weather = weather[(weather.year >= 1984)&(weather.year <= 2018)]
weather.state = weather.state.astype('int')
weather.county = weather.county.astype('int')
weather.year = weather.year.astype('int')

# Constructing quarterly and cumulative weather features
weather['prcp_Q2'] = weather.loc[:,'prcp_14':'prcp_26'].sum(axis=1)
weather['prcp_Q3'] = weather.loc[:,'prcp_27':'prcp_39'].sum(axis=1)
weather['prcp_Q4'] = weather.loc[:,'prcp_40':'prcp_52'].sum(axis=1)
weather['prcp_Q1:Q2'] = weather.loc[:,'prcp_1':'prcp_26'].sum(axis=1)
weather['prcp_Q1:Q3'] = weather.loc[:,'prcp_1':'prcp_39'].sum(axis=1)
weather['prcp_Q1:Q4'] = weather.loc[:,'prcp_1':'prcp_52'].sum(axis=1)

weather['tmax_Q2'] = weather.loc[:,'tmax_14':'tmax_26'].mean(axis=1)
weather['tmax_Q3'] = weather.loc[:,'tmax_27':'tmax_39'].mean(axis=1)
weather['tmax_Q4'] = weather.loc[:,'tmax_40':'tmax_52'].mean(axis=1)
weather['tmax_Q1:Q2'] = weather.loc[:,'tmax_1':'tmax_26'].mean(axis=1)
weather['tmax_Q1:Q3'] = weather.loc[:,'tmax_1':'tmax_39'].mean(axis=1)
weather['tmax_Q1:Q4'] = weather.loc[:,'tmax_1':'tmax_52'].mean(axis=1)

weather['tmin_Q2'] = weather.loc[:,'tmin_14':'tmin_26'].mean(axis=1)
weather['tmin_Q3'] = weather.loc[:,'tmin_27':'tmin_39'].mean(axis=1)
weather['tmin_Q4'] = weather.loc[:,'tmin_40':'tmin_52'].mean(axis=1)
weather['tmin_Q1:Q2'] = weather.loc[:,'tmin_1':'tmin_26'].mean(axis=1)
weather['tmin_Q1:Q3'] = weather.loc[:,'tmin_1':'tmin_39'].mean(axis=1)
weather['tmin_Q1:Q4'] = weather.loc[:,'tmin_1':'tmin_52'].mean(axis=1)

weather['gddf_Q2'] = weather.loc[:,'gddf_14':'gddf_26'].sum(axis=1)
weather['gddf_Q3'] = weather.loc[:,'gddf_27':'gddf_39'].sum(axis=1)
weather['gddf_Q4'] = weather.loc[:,'gddf_40':'gddf_52'].sum(axis=1)
weather['gddf_Q1:Q2'] = weather.loc[:,'gddf_1':'gddf_26'].sum(axis=1)
weather['gddf_Q1:Q3'] = weather.loc[:,'gddf_1':'gddf_39'].sum(axis=1)
weather['gddf_Q1:Q4'] = weather.loc[:,'gddf_1':'gddf_52'].sum(axis=1)

weather['srad_Q2'] = weather.loc[:,'srad_14':'srad_26'].sum(axis=1)
weather['srad_Q3'] = weather.loc[:,'srad_27':'srad_39'].sum(axis=1)
weather['srad_Q4'] = weather.loc[:,'srad_40':'srad_52'].sum(axis=1)
weather['srad_Q1:Q2'] = weather.loc[:,'srad_1':'srad_26'].sum(axis=1)
weather['srad_Q1:Q3'] = weather.loc[:,'srad_1':'srad_39'].sum(axis=1)
weather['srad_Q1:Q4'] = weather.loc[:,'srad_1':'srad_52'].sum(axis=1)


# Removing weather data after harvesting and before next planting date
idx = list(['state', 'county', 'year']) + \
      list(weather.loc[:,'prcp_16':'prcp_43'].columns) + \
      list(weather.loc[:,'tmax_16':'tmax_43'].columns) + \
      list(weather.loc[:,'tmin_16':'tmin_43'].columns) + \
      list(weather.loc[:,'gddf_16':'gddf_43'].columns) + \
      list(weather.loc[:,'srad_16':'srad_43'].columns) + \
      list(weather.loc[:, 'prcp_Q2':])
weather = weather[idx]


cv = 10

# Importing APSIM variables
data_d = pd.read_csv('data_all_apsim.csv', index_col=0)



##  -----------------  data preprocessing ----------------- ##


# Feature construction (trend)
data_d['yield_trend'] = 0
for s in data_d.State.unique():
    for c in data_d[data_d.State==s].County.unique():
        y1 = pd.DataFrame(data_d.Yield[(data_d.Year<2018) & ((data_d.State).astype('int') == s) & ((data_d.County).astype('int') == c)])
        x1 = pd.DataFrame(data_d.Year[(data_d.Year<2018) & ((data_d.State).astype('int') == s) & ((data_d.County).astype('int') == c)])
        regressor = LinearRegression()
        regressor.fit(x1, y1)
        data_d.loc[(data_d.Year<2018)&(data_d.State==s)&(data_d.County==c),'yield_trend'] = regressor.predict(x1)
        if len(data_d.Year[(data_d.Year==2018)&(data_d.State==s)&(data_d.County==c)].unique()) != 0:
            data_d.loc[(data_d.Year==2018)&(data_d.State==s)&(data_d.County==c),'yield_trend'] = regressor.predict(pd.DataFrame([2018]))

# Joining the APSIM, soil and progress variables together
data = pd.concat([data_d,pd.DataFrame(progress[:,12:25])], axis=1)
data = pd.concat([data,pd.DataFrame(soil)],axis=1)

# dropping rows with na values (years before 1984)
data = data.dropna()
data = data.reset_index(drop=True)

# renaming columns
progress_names = ['Progress_' + str(i) for i in range(1,14)]
soil_names = ['Soil_' + str(i) for i in range(1,181)]
names = [progress_names, soil_names]
names = [item for sublist in names for item in sublist]
col_names = data.columns.values
col_names[1:4] = ['year', 'state', 'county']
col_names[28:] = names
data.columns = col_names

# Joining weather variables
data = pd.merge(data, weather , on=['year','state','county'])

# Scaling the variables
data = data.rename(columns = {'year':'Year'})
columns_to_scale = data.drop(columns=['Yield','Year','state','county']).columns.values
scaler = MinMaxScaler()
scaled_columns = scaler.fit_transform(data[columns_to_scale])
scaled_columns = pd.DataFrame(scaled_columns, columns=columns_to_scale)

data2 = pd.DataFrame(data.Yield)
data = pd.concat([data2, data.Year, scaled_columns], axis=1)

# Splitting the data set to test and train
test = data[data.Year==2018]
train = data[data.Year!=2018]

x_test = test.drop(columns=['Yield'])
y_test = test.Yield

X = train.drop(columns=['Yield'])
X = X.reset_index(drop=True)
Y = train.Yield
Y.reset_index(inplace=True, drop=True)


# feature selection with random forest
rf = RandomForestRegressor(n_estimators=100)
rf.fit(X, Y)

from eli5.sklearn import PermutationImportance

perm = PermutationImportance(rf, cv=cv, n_iter=10).fit(X, Y)
feature_importances = [(feature, importance) for feature, importance in zip(list(X.columns), list(np.abs(perm.feature_importances_)))]
feature_importances = pd.DataFrame(sorted(feature_importances, key = lambda x: x[1], reverse = True))
selected_features = feature_importances.iloc[0:80,:][0]
if np.isin('Year', selected_features)==False:
    selected_features = selected_features.append(pd.Series('Year'))
X = X.loc[:,selected_features]
x_test = x_test.loc[:,selected_features]
selected_features.to_csv('RF_features_2018.csv')


# CV
kf = KFold(cv)



 ## ---------------- Bayesian Search ---------------- ##


max_evals = 20

def objective_LASSO(params):
    LASSO_df_B = pd.DataFrame()
    L1_B = Lasso()
    for train_index, test_index in kf.split(X):
        LASSO_B = L1_B.fit(np.array(X.drop(columns='Year'))[train_index], np.array(Y)[train_index])
        LASSO_df_B = pd.concat([LASSO_df_B, pd.DataFrame(LASSO_B.predict(np.array(X.drop(columns='Year'))[test_index]))])
    loss_LASSO = mse(data_d.Yield[(data_d.Year < 2018)], LASSO_df_B)
    return {'loss': loss_LASSO, 'params': params, 'status': STATUS_OK}

space_LASSO = {'alpha': hp.uniform('alpha', 10**-5, 1)}
tpe_algorithm = tpe.suggest
trials_LASSO = Trials()
best_LASSO = fmin(fn=objective_LASSO, space=space_LASSO, algo=tpe.suggest,
                  max_evals=max_evals, trials=trials_LASSO, rstate=np.random.RandomState(1369))
LASSO_param_B = pd.DataFrame({'alpha': []})
for i in range(max_evals):
    LASSO_param_B.alpha[i] = trials_LASSO.results[i]['params']['alpha']
LASSO_param_B = pd.DataFrame(LASSO_param_B.alpha)



def objective_XGB(params):
    XGB_df_B = pd.DataFrame()
    X1_B = XGBRegressor(objective='reg:squarederror', **params)
    for train_index, test_index in kf.split(X):
        XGB_B = X1_B.fit(np.array(X.drop(columns='Year'))[train_index], np.array(Y)[train_index])
        XGB_df_B = pd.concat([XGB_df_B, pd.DataFrame(X1_B.predict(np.array(X.drop(columns='Year'))[test_index]))])
    loss_XGB = mse(data_d.Yield[(data_d.Year < 2018)], XGB_df_B)
    return {'loss': loss_XGB, 'params': params, 'status': STATUS_OK}

space_XGB = {'gamma': hp.uniform('gamma', 0, 1),
             'learning_rate': hp.uniform('learning_rate', 0.001, 0.5),
             'n_estimators': hp.choice('n_estimators', [100, 300, 500, 1000]),
             'max_depth': hp.choice('max_depth', [int(x) for x in np.arange(3, 20, 1)])}
tpe_algorithm = tpe.suggest
trials_XGB = Trials()
best_XGB = fmin(fn=objective_XGB, space=space_XGB, algo=tpe.suggest,
                max_evals=max_evals, trials=trials_XGB, rstate=np.random.RandomState(1369))
XGB_param_B = pd.DataFrame({'gamma': [], 'learning_rate': [], 'n_estimators': [], 'max_depth': []})
for i in range(max_evals):
    XGB_param_B.gamma[i] = trials_XGB.results[i]['params']['gamma']
    XGB_param_B.learning_rate[i] = trials_XGB.results[i]['params']['learning_rate']
    XGB_param_B.n_estimators[i] = trials_XGB.results[i]['params']['n_estimators']
    XGB_param_B.max_depth[i] = trials_XGB.results[i]['params']['max_depth']
XGB_param_B = pd.DataFrame({'gamma': XGB_param_B.gamma,
                            'learning_rate': XGB_param_B.learning_rate,
                            'n_estimators': XGB_param_B.n_estimators,
                            'max_depth': XGB_param_B.max_depth})


def objective_LGB(params):
    LGB_df_B = pd.DataFrame()
    G1_B = LGBMRegressor(objective='regression', **params)
    for train_index, test_index in kf.split(X):
        LGB_B = G1_B.fit(np.array(X.drop(columns='Year'))[train_index], np.array(Y)[train_index])
        LGB_df_B = pd.concat([LGB_df_B, pd.DataFrame(G1_B.predict(np.array(X.drop(columns='Year'))[test_index]))])
    loss_LGB = mse(data_d.Yield[(data_d.Year < 2018)], LGB_df_B)
    return {'loss': loss_LGB, 'params': params, 'status': STATUS_OK}

space_LGB = {'num_leaves': hp.choice('num_leaves', [int(x) for x in np.arange(5, 40, 2)]),
             'learning_rate': hp.uniform('learning_rate', 0.1, 0.5),
             'n_estimators': hp.choice('n_estimators', [500, 1000, 1500, 2000])}
tpe_algorithm = tpe.suggest
trials_LGB = Trials()
best_LGB = fmin(fn=objective_LGB, space=space_LGB, algo=tpe.suggest,
                max_evals=max_evals, trials=trials_LGB, rstate=np.random.RandomState(1369))
LGB_param_B = pd.DataFrame({'num_leaves': [], 'learning_rate': [], 'n_estimators': []})
for i in range(max_evals):
    LGB_param_B.num_leaves[i] = trials_LGB.results[i]['params']['num_leaves']
    LGB_param_B.learning_rate[i] = trials_LGB.results[i]['params']['learning_rate']
    LGB_param_B.n_estimators[i] = trials_LGB.results[i]['params']['n_estimators']
LGB_param_B = pd.DataFrame({'num_leaves': LGB_param_B.num_leaves,
                            'learning_rate': LGB_param_B.learning_rate,
                            'n_estimators': LGB_param_B.n_estimators})


def objective_RF(params):
    RF_df_B = pd.DataFrame()
    R1_B = RandomForestRegressor(**params)
    for train_index, test_index in kf.split(X):
        RF_B = R1_B.fit(np.array(X.drop(columns='Year'))[train_index], np.array(Y)[train_index])
        RF_df_B = pd.concat([RF_df_B, pd.DataFrame(R1_B.predict(np.array(X.drop(columns='Year'))[test_index]))])
    loss_RF = mse(data_d.Yield[(data_d.Year < 2018)], RF_df_B)
    return {'loss': loss_RF, 'params': params, 'status': STATUS_OK}

space_RF = {'n_estimators': hp.choice('n_estimators', [100, 200, 300, 500]),
            'max_depth': hp.choice('max_depth', [int(x) for x in np.arange(5, 41, 5)])}
tpe_algorithm = tpe.suggest
trials_RF = Trials()
best_RF = fmin(fn=objective_RF, space=space_RF, algo=tpe.suggest,
               max_evals=max_evals, trials=trials_RF, rstate=np.random.RandomState(1369))
RF_param_B = pd.DataFrame({'n_estimators': [], 'max_depth': []})
for i in range(max_evals):
    RF_param_B.n_estimators[i] = trials_RF.results[i]['params']['n_estimators']
    RF_param_B.max_depth[i] = trials_RF.results[i]['params']['max_depth']
RF_param_B = pd.DataFrame({'n_estimators': RF_param_B.n_estimators,
                           'max_depth': RF_param_B.max_depth})


## ---------------- Permutation feature importance ---------------- ##


def perm_fi(model, cv, n_iter):
    perm = PermutationImportance(model, cv=cv, n_iter=n_iter).fit(X.drop(columns='Year'), Y)
    feature_importances = [(feature, importance) for feature, importance in zip(list(X.columns), list(np.abs(perm.feature_importances_)))]
    feature_importances = pd.DataFrame(sorted(feature_importances, key = lambda x: x[1], reverse = True))
    return feature_importances


## ---------------- Building models ---------------- ##
LASSO_df2 = pd.DataFrame()
L2 = Lasso(alpha=trials_LASSO.best_trial['result']['params']['alpha'], random_state=1369)
for train_index, test_index in kf.split(X):
    L2.fit(np.array(X.drop(columns='Year'))[train_index], np.array(Y)[train_index])
    LASSO_df2 = pd.concat([LASSO_df2, pd.DataFrame(L2.predict(np.array(X.drop(columns='Year'))[test_index]))])
LASSO_df2 = LASSO_df2.reset_index(drop=True)
LASSO_mse2 = mse(data_d.Yield[(data_d.Year<2018)], LASSO_df2)
LASSO = L2.fit(X.drop(columns='Year'), Y)
LASSO_preds_test2 = LASSO.predict(x_test.drop(columns='Year'))
pd.DataFrame(LASSO_preds_test2).to_csv('LASSO_preds_test_2018.csv')
LASSO_mse_test2 = mse(data_d.Yield[data_d.Year==2018], LASSO_preds_test2)
LASSO_rmse_test2 = np.sqrt(LASSO_mse_test2)
LASSO_preds_train = LASSO.predict(X.drop(columns='Year'))
pd.DataFrame(LASSO_preds_train).to_csv('LASSO_preds_train_2018.csv')
LASSO_rmse_train = np.sqrt(mse(data_d.Yield[data_d.Year<2018], LASSO_preds_train))
feature_importances_lasso = perm_fi(L2, cv, 10)
feature_importances_lasso.to_csv('feature_importances_lasso_2018.csv')



### ---------- XGB ------------ ###
XGB_df2 = pd.DataFrame()
X2 = XGBRegressor(objective='reg:squarederror',
                  gamma=trials_XGB.best_trial['result']['params']['gamma'],
                  learning_rate=trials_XGB.best_trial['result']['params']['learning_rate'],
                  n_estimators=int(trials_XGB.best_trial['result']['params']['n_estimators']),
                  max_depth=int(trials_XGB.best_trial['result']['params']['max_depth']), random_state=1369)
for train_index, test_index in kf.split(X):
    X2.fit(np.array(X.drop(columns='Year'))[train_index], np.array(Y)[train_index])
    XGB_df2 = pd.concat([XGB_df2, pd.DataFrame(X2.predict(np.array(X.drop(columns='Year'))[test_index]))])
XGB_df2 = XGB_df2.reset_index(drop=True)
XGB_mse2 = mse(data_d.Yield[(data_d.Year<2018)], XGB_df2)
XGB = X2.fit(X.drop(columns='Year'), Y)
XGB_preds_test2 = XGB.predict(x_test.drop(columns='Year'))
pd.DataFrame(XGB_preds_test2).to_csv('XGB_preds_test_2018.csv')
XGB_mse_test2 = mse(data_d.Yield[data_d.Year==2018], XGB_preds_test2)
XGB_rmse_test2 = np.sqrt(XGB_mse_test2)
XGB_preds_train = XGB.predict(X.drop(columns='Year'))
pd.DataFrame(XGB_preds_train).to_csv('XGB_preds_train_2018.csv')
XGB_rmse_train = np.sqrt(mse(data_d.Yield[data_d.Year<2018], XGB_preds_train))
perm_xgb = PermutationImportance(X2, cv=cv, n_iter=10).fit(X.as_matrix(), Y.as_matrix())
feature_importances_xgb = [(feature, importance) for feature, importance in
                           zip(list(X.columns), list(np.abs(perm_xgb.feature_importances_)))]
feature_importances_xgb = pd.DataFrame(sorted(feature_importances_xgb, key=lambda x: x[1], reverse=True))
feature_importances_xgb.to_csv('feature_importances_xgb_2018.csv')


### ---------- LGB ------------ ###
LGB_df2 = pd.DataFrame()
G2 = LGBMRegressor(objective='regression', random_state=1369,
                   num_leaves=int(trials_LGB.best_trial['result']['params']['num_leaves']),
                   learning_rate=trials_LGB.best_trial['result']['params']['learning_rate'],
                   n_estimators=int(trials_LGB.best_trial['result']['params']['n_estimators']))
for train_index, test_index in kf.split(X):
    G2.fit(np.array(X.drop(columns='Year'))[train_index], np.array(Y)[train_index])
    LGB_df2 = pd.concat([LGB_df2, pd.DataFrame(G2.predict(np.array(X.drop(columns='Year'))[test_index]))])
LGB_df2 = LGB_df2.reset_index(drop=True)
LGB_mse2 = mse(data_d.Yield[(data_d.Year<2018)], LGB_df2)
LGB = G2.fit(X.drop(columns='Year'), Y)
LGB_preds_test2 = LGB.predict(x_test.drop(columns='Year'))
pd.DataFrame(LGB_preds_test2).to_csv('LGB_preds_test_2018.csv')
LGB_mse_test2 = mse(data_d.Yield[data_d.Year==2018], LGB_preds_test2)
LGB_rmse_test2 = np.sqrt(LGB_mse_test2)
LGB_preds_train = LGB.predict(X.drop(columns='Year'))
pd.DataFrame(LGB_preds_train).to_csv('LGB_preds_train_2018.csv')
LGB_rmse_train = np.sqrt(mse(data_d.Yield[data_d.Year<2018], LGB_preds_train))
feature_importances_lgb = perm_fi(G2, cv, 10)
feature_importances_lgb.to_csv('feature_importances_lgb_2018.csv')


### ---------- RF ------------ ###
RF_df2 = pd.DataFrame()
R2 = RandomForestRegressor(max_depth=int(trials_RF.best_trial['result']['params']['max_depth']),
                           n_estimators=int(trials_RF.best_trial['result']['params']['n_estimators']), random_state=1369)
for train_index, test_index in kf.split(X):
    R2.fit(np.array(X.drop(columns='Year'))[train_index], np.array(Y)[train_index])
    RF_df2 = pd.concat([RF_df2, pd.DataFrame(R2.predict(np.array(X.drop(columns='Year'))[test_index]))])
RF_df2 = RF_df2.reset_index(drop=True)
RF_mse2 = mse(data_d.Yield[(data_d.Year<2018)], RF_df2)
RF = R2.fit(X.drop(columns='Year'), Y)
RF_preds_test2 = RF.predict(x_test.drop(columns='Year'))
pd.DataFrame(RF_preds_test2).to_csv('RF_preds_test_2018.csv')
RF_mse_test2 = mse(data_d.Yield[data_d.Year==2018], RF_preds_test2)
RF_rmse_test2 = np.sqrt(RF_mse_test2)
RF_preds_train = RF.predict(X.drop(columns='Year'))
pd.DataFrame(RF_preds_train).to_csv('RF_preds_train_2018.csv')
RF_rmse_train = np.sqrt(mse(data_d.Yield[data_d.Year<2018], RF_preds_train))
feature_importances_rf = perm_fi(R2, cv, 10)
feature_importances_rf.to_csv('feature_importances_rf_2018.csv')


### ---------- LR ------------ ###
LR_df2 = pd.DataFrame()
lm2 = LinearRegression()
lm2.fit(X.drop(columns='Year'),Y)
for train_index, test_index in kf.split(X):
    lm2.fit(np.array(X.drop(columns='Year'))[train_index], np.array(Y)[train_index])
    LR_df2 = pd.concat([LR_df2, pd.DataFrame(lm2.predict(np.array(X.drop(columns='Year'))[test_index]))])
LR_df2 = LR_df2.reset_index(drop=True)
LR_mse2 = mse(data_d.Yield[(data_d.Year<2018)], LR_df2)
LR = lm2.fit(X.drop(columns='Year'), Y)
LR_preds_test2 = LR.predict(x_test.drop(columns='Year'))
pd.DataFrame(LR_preds_test2).to_csv('LR_preds_test2_2018.csv')
LR_mse_test2 = mse(data_d.Yield[data_d.Year==2018], LR_preds_test2)
LR_rmse_test2 = np.sqrt(LR_mse_test2)
LR_preds_train = LR.predict(X.drop(columns='Year'))
pd.DataFrame(LR_preds_train).to_csv('LR_preds_train_2018.csv')
LR_rmse_train = np.sqrt(mse(data_d.Yield[data_d.Year<2018], LR_preds_train))
feature_importances_lr = perm_fi(lm2, cv, 10)
feature_importances_lr.to_csv('feature_importances_lr_2018.csv')



## ---------------- Optimizing Ensembles ---------------- ##

def objective2(y):
    return mse(data_d.Yield[(data_d.Year<2018)],
               (y[0]*LASSO_df2 + y[1]*XGB_df2 + y[2]*LGB_df2 + y[3]*RF_df2 + y[4]*LR_df2))

def constraint12(y):
    return y[0] + y[1] + y[2] + y[3] + y[4] - 1.0
def constraint22(y):
    return LASSO_mse2 - objective2(y)
def constraint32(y):
    return XGB_mse2 - objective2(y)
def constraint42(y):
    return LGB_mse2 - objective2(y)
def constraint52(y):
    return RF_mse2 - objective2(y)
def constraint62(y):
    return LR_mse2 - objective2(y)


y0 = np.zeros(5)
y0[0] = 1 / 5
y0[1] = 1 / 5
y0[2] = 1 / 5
y0[3] = 1 / 5
y0[4] = 1 / 5

b = (0, 1.0)
bnds2 = (b, b, b, b, b)
con12 = {'type': 'eq', 'fun': constraint12}
con22 = {'type': 'ineq', 'fun': constraint22}
con32 = {'type': 'ineq', 'fun': constraint32}
con42 = {'type': 'ineq', 'fun': constraint42}
con52 = {'type': 'ineq', 'fun': constraint52}
con62 = {'type': 'ineq', 'fun': constraint62}

cons2 = [con12, con22, con32, con42, con52, con62]

solution2 = minimize(objective2, y0, method='SLSQP',
                    options={'disp': True, 'maxiter': 3000, 'eps': 1e-3}, bounds=bnds2,
                    constraints=cons2)
y = solution2.x

cowe_preds_test = y[0]*LASSO_preds_test2 + y[1]*XGB_preds_test2 + y[2]*LGB_preds_test2 + y[3]*RF_preds_test2 + y[4]*LR_preds_test2
cowe_mse_test = mse(data_d.Yield[data_d.Year==2018], cowe_preds_test)
cowe_rmse_test = np.sqrt(cowe_mse_test)
pd.DataFrame(cowe_preds_test).to_csv('cowe_preds_test_2018.csv')
cowe_preds_train = y[0]*LASSO_preds_train + y[1]*XGB_preds_train + y[2]*LGB_preds_train + y[3]*RF_preds_train + y[4]*LR_preds_train
pd.DataFrame(cowe_preds_train).to_csv('cowe_preds_train_2018.csv')
cowe_rmse_train = np.sqrt(mse(data_d.Yield[data_d.Year<2018], cowe_preds_train))


cowe_preds_CV = y[0]*LASSO_df2 + y[1]*XGB_df2 + y[2]*LGB_df2 + y[3]*RF_df2 + y[4]*LR_df2
cowe_mse_CV = mse(data_d.Yield[(data_d.Year<2018)], cowe_preds_CV)
cowe_rmse_CV = np.sqrt(cowe_mse_CV)


cls_preds_test = y0[0]*LASSO_preds_test2 + y0[1]*XGB_preds_test2 + y0[2]*LGB_preds_test2 + y0[3]*RF_preds_test2 + y0[4]*LR_preds_test2
cls_mse_test = mse(data_d.Yield[data_d.Year==2018], cls_preds_test)
cls_rmse_test = np.sqrt(cls_mse_test)
pd.DataFrame(cls_preds_test).to_csv('cls_preds_test_2018.csv')
cls_preds_train = y0[0]*LASSO_preds_train + y0[1]*XGB_preds_train + y0[2]*LGB_preds_train + y0[3]*RF_preds_train + y0[4]*LR_preds_train
pd.DataFrame(cls_preds_train).to_csv('cls_preds_train_2018.csv')
cls_rmse_train = np.sqrt(mse(data_d.Yield[data_d.Year<2018], cls_preds_train))


cls_preds_CV = y0[0]*LASSO_df2 + y0[1]*XGB_df2 + y0[2]*LGB_df2 + y0[3]*RF_df2 + y0[4]*LR_df2
cls_mse_CV = mse(data_d.Yield[(data_d.Year<2018)], cls_preds_CV)
cls_rmse_CV = np.sqrt(cls_mse_CV)



## -------------------------------- STACKING -------------------------------- ##

predsDF2 = pd.DataFrame()
predsDF2['LASSO'] = LASSO_df2[0]
predsDF2['XGB']= XGB_df2[0]
predsDF2['LGB'] = LGB_df2[0]
predsDF2['RF'] = RF_df2[0]
predsDF2['LR'] = LR_df2[0]
predsDF2['Y'] = data_d.Yield[(data_d.Year < 2018)].reset_index(drop=True)
x_stacked2 = predsDF2.drop(columns='Y', axis=1)
y_stacked2 = predsDF2['Y']
testPreds2 = pd.DataFrame([LASSO_preds_test2, XGB_preds_test2, LGB_preds_test2, RF_preds_test2, LR_preds_test2]).T
testPreds2.columns = ['LASSO', 'XGB', 'LGB', 'RF', 'LR']


stck_reg2 = LinearRegression()
stck_reg2.fit(x_stacked2, y_stacked2)
stck_reg_preds_test2 = stck_reg2.predict(testPreds2)
stck_reg_mse_test2 = mse(data_d.Yield[data_d.Year == 2018], stck_reg_preds_test2)
stck_reg_rmse_test2 = np.sqrt(stck_reg_mse_test2)
pd.DataFrame(stck_reg_preds_test2).to_csv('stck_reg_preds_test_2018.csv')
stck_reg_preds_train = stck_reg2.predict(x_stacked2)
pd.DataFrame(stck_reg_preds_train).to_csv('stck_reg_preds_train_2018.csv')
stck_reg_rmse_train = np.sqrt(mse(data_d.Yield[data_d.Year < 2018], stck_reg_preds_train))

stck_lasso2 = Lasso()
stck_lasso2.fit(x_stacked2, y_stacked2)
stck_lasso_preds_test2 = stck_lasso2.predict(testPreds2)
stck_lasso_mse_test2 = mse(data_d.Yield[data_d.Year == 2018], stck_lasso_preds_test2)
stck_lasso_rmse_test2 = np.sqrt(stck_lasso_mse_test2)
pd.DataFrame(stck_lasso_preds_test2).to_csv('stck_lasso_preds_test_2018.csv')
stck_lasso_preds_train = stck_lasso2.predict(x_stacked2)
pd.DataFrame(stck_lasso_preds_train).to_csv('stck_lasso_preds_train_2018.csv')
stck_lasso_rmse_train = np.sqrt(mse(data_d.Yield[data_d.Year < 2018], stck_lasso_preds_train))

stck_rf2 = RandomForestRegressor()
stck_rf2.fit(x_stacked2, y_stacked2)
stck_rf_preds_test2 = stck_rf2.predict(testPreds2)
stck_rf_mse_test2 = mse(data_d.Yield[data_d.Year == 2018], stck_rf_preds_test2)
stck_rf_rmse_test2 = np.sqrt(stck_rf_mse_test2)
pd.DataFrame(stck_rf_preds_test2).to_csv('stck_rf_preds_test_2018.csv')
stck_rf_preds_train = stck_rf2.predict(x_stacked2)
pd.DataFrame(stck_rf_preds_train).to_csv('stck_rf_preds_train_2018.csv')
stck_rf_rmse_train = np.sqrt(mse(data_d.Yield[data_d.Year < 2018], stck_rf_preds_train))

stck_lgb2 = LGBMRegressor()
stck_lgb2.fit(x_stacked2, y_stacked2)
stck_lgb_preds_test2 = stck_lgb2.predict(testPreds2)
stck_lgb_mse_test2 = mse(data_d.Yield[data_d.Year == 2018], stck_lgb_preds_test2)
stck_lgb_rmse_test2 = np.sqrt(stck_lgb_mse_test2)
pd.DataFrame(stck_lgb_preds_test2).to_csv('stck_lgb_preds_test_2018.csv')
stck_lgb_preds_train = stck_lgb2.predict(x_stacked2)
pd.DataFrame(stck_lgb_preds_train).to_csv('stck_lgb_preds_train_2018.csv')
stck_lgb_rmse_train = np.sqrt(mse(data_d.Yield[data_d.Year < 2018], stck_lgb_preds_train))



## -------------------------- RESULTS -------------------------- ##


test_results = pd.DataFrame(data={'model':['RMSE'],'LASSO':[LASSO_rmse_test2], 'XGB':[XGB_rmse_test2], 'LGB':[LGB_rmse_test2],
                                  'RF': [RF_rmse_test2], 'LR': [LR_rmse_test2],
                                  'COWE': [cowe_rmse_test], 'Classical': [cls_rmse_test],
                                  'stck_reg': [stck_reg_rmse_test2], 'stck_lasso': [stck_lasso_rmse_test2],
                                  'stck_rf': [stck_rf_rmse_test2], 'stck_lgb': [stck_lgb_rmse_test2]})

train_results = pd.DataFrame(data={'model':['RMSE'],'LASSO':[LASSO_rmse_train], 'XGB':[XGB_rmse_train], 'LGB':[LGB_rmse_train],
                                  'RF': [RF_rmse_train], 'LR': [LR_rmse_train],
                                  'COWE': [cowe_rmse_train], 'Classical': [cls_rmse_train],
                                  'stck_reg': [stck_reg_rmse_train], 'stck_lasso': [stck_lasso_rmse_train],
                                  'stck_rf': [stck_rf_rmse_train], 'stck_lgb': [stck_lgb_rmse_train]})

CV_results = pd.DataFrame(data={'model':['RMSE'], 'LASSO':[np.sqrt(LASSO_mse2)], 'XGB':[np.sqrt(XGB_mse2)],
                                'LGB':[np.sqrt(LGB_mse2)], 'RF': [np.sqrt(RF_mse2)], 'LR': [np.sqrt(LR_mse2)],
                                'COWE': [cowe_rmse_CV],
                                'Classical':[cls_rmse_CV]})

test_results.to_csv('2018_test.csv')
train_results.to_csv('2018_train.csv')
CV_results.to_csv('2018_CV.csv')
