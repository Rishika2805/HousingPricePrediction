import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

# STEPS 

# 1. Read data 
housing = pd.read_csv("housing.csv")

# 2. Creat a Stratified test and train set based on income category

housing['income_cat'] = pd.cut(
    housing['median_income'],
    bins = [0.0,1.5,3.0,4.5,6.0,np.inf],
    labels=[1,2,3,4,5]
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2,random_state=42)

for train_index,test_index in split.split(housing,housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop('income_cat',axis = 1)
    strat_test_set = housing.loc[test_index].drop('income_cat', axis = 1)

# working on a copy of train set
housing = strat_train_set.copy()

# 3. Separate predictors and labels

housing_labels = housing['median_house_value'].copy()
housing = housing.drop('median_house_value', axis = 1)

# 4. Separate numerical and categorical columns
num_attri = housing.drop('ocean_proximity', axis = 1).columns.tolist()
cat_attri = ['ocean_proximity']

print(housing)
# 5. Pipline

# Numerical Pipline

num_pipline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)

# Categorical Pipline 

cat_pipline = Pipeline(
    [
        ("encoder", OneHotEncoder(handle_unknown='ignore'))
    ]
)

# Full Pipline

full_pipline = ColumnTransformer(
    [
        ('num', num_pipline, num_attri),
        ('cat', cat_pipline, cat_attri)
    ]
)

# 6. Transform the data

housing_prepared = full_pipline.fit_transform(housing)

print(housing_prepared.shape)

# 7. Selecting model

# Linear Regression 

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared,housing_labels)
lin_pred = lin_reg.predict(housing_prepared)
lin_rmse = root_mean_squared_error(housing_labels,lin_pred)
print(f"The root mean sqaure error for linear regression {lin_rmse}")

# Decision Tree 

dec_reg = DecisionTreeRegressor()
dec_reg.fit(housing_prepared,housing_labels)
dec_pred = dec_reg.predict(housing_prepared)
dec_rmse = root_mean_squared_error(housing_labels,dec_pred)
print(f"The root mean sqaure error for linear regression {dec_rmse}")

# Random Forest

rd_reg = RandomForestRegressor()
rd_reg.fit(housing_prepared,housing_labels)
rd_pred = rd_reg.predict(housing_prepared)
rd_rmse = root_mean_squared_error(housing_labels, rd_pred)
print(f"The root mean sqaure error for linear regression {rd_rmse}")


# Cross validation for all ML Algorithm

# WARNING: Scikit-Learn’s scoring uses utility functions (higher is better), so RMSE is returned as negative.
# We use minus (-) to convert it back to positive RMSE.


lin_rmses = -cross_val_score(
    lin_reg,
    housing_prepared,
    housing_labels,
    scoring='neg_root_mean_squared_error', 
    cv = 10
)

print(pd.Series(lin_rmses).describe())

dec_rmses = -cross_val_score(
    dec_reg,
    housing_prepared,
    housing_labels,
    scoring='neg_root_mean_squared_error', 
    cv = 10
)

print(pd.Series(dec_rmses).describe())

rd_rmses = -cross_val_score(
    rd_reg,
    housing_prepared,
    housing_labels,
    scoring='neg_root_mean_squared_error', 
    cv = 10
)

print(pd.Series(rd_rmses).describe())
