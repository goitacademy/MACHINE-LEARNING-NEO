import joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import TargetEncoder
from sklearn.utils import estimator_html_repr
from mod_07_topic_13_transformer import VisRatioEstimator

# %%

data = pd.read_csv('../datasets/mod_07_topic_13_bigmart_data.csv')
data.sample(10, random_state=42)

# %%

data.info()

# %%

data['Outlet_Establishment_Year'] = 2013 - data['Outlet_Establishment_Year']

# %%

data['Item_Visibility'] = (data['Item_Visibility']
                           .mask(data['Item_Visibility'].eq(0), np.nan))

data['Item_Visibility_Avg'] = (data
                               .groupby(['Item_Type',
                                         'Outlet_Type'])['Item_Visibility']
                               .transform('mean'))

data['Item_Visibility'] = (
    data['Item_Visibility'].fillna(data['Item_Visibility_Avg']))

data['Item_Visibility_Ratio'] = (
    data['Item_Visibility'] / data['Item_Visibility_Avg'])

data[['Item_Visibility', 'Item_Visibility_Ratio']].describe()

# %%

data['Item_Fat_Content'].unique()

# %%

data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({
    'low fat': 'Low Fat',
    'LF': 'Low Fat',
    'reg': 'Regular'})

# %%

data['Item_Identifier_Type'] = data['Item_Identifier'].str[:2]

data[['Item_Identifier', 'Item_Identifier_Type', 'Item_Type']].head()

# %%

data_num = data.select_dtypes(include=np.number)
data_cat = data.select_dtypes(include='object')

# %%

X_train_num, X_test_num, X_train_cat,  X_test_cat, y_train, y_test = (
    train_test_split(
        data_num.drop(['Item_Outlet_Sales',
                       'Item_Visibility_Avg'], axis=1),
        data_cat.drop('Item_Identifier', axis=1),
        data['Item_Outlet_Sales'],
        test_size=0.2,
        random_state=42))

# %%

num_imputer = SimpleImputer().set_output(transform='pandas')

X_train_num = num_imputer.fit_transform(X_train_num)
X_test_num = num_imputer.transform(X_test_num)

# %%

cat_imputer = SimpleImputer(
    strategy='most_frequent').set_output(transform='pandas')

X_train_cat = cat_imputer.fit_transform(X_train_cat)
X_test_cat = cat_imputer.transform(X_test_cat)

# %%

enc_auto = TargetEncoder(random_state=42).set_output(transform='pandas')

X_train_cat = enc_auto.fit_transform(X_train_cat, y_train)
X_test_cat = enc_auto.transform(X_test_cat)

# %%

X_train_concat = pd.concat([X_train_num, X_train_cat], axis=1)
X_test_concat = pd.concat([X_test_num, X_test_cat], axis=1)

X_train_concat.head()

# %%

clf = (RandomForestRegressor(
    n_jobs=-1,
    random_state=42)
    .fit(X_train_concat,
         y_train))

pred_clf = clf.predict(X_test_concat)

rmse_clf = root_mean_squared_error(y_test, pred_clf)

print(f"Prototype's RMSE on test: {rmse_clf:.1f}")

# %%

sns.set_theme()

(pd.Series(
    data=clf.feature_importances_,
    index=X_train_concat.columns)
    .sort_values(ascending=True)
    .plot
    .barh())

plt.show()

# %%

vis_est = VisRatioEstimator()

data = (data.rename(columns={
    'Item_Visibility_Ratio': 'Item_Visibility_Ratio_prev'}))

data = vis_est.fit_transform(data)

(data[['Item_Visibility_Ratio_prev', 'Item_Visibility_Ratio']]
 .sample(10, random_state=42))

# %%

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', TargetEncoder(random_state=42))])

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer())])

preprocessor = (ColumnTransformer(
    transformers=[
        ('cat',
         cat_transformer,
         make_column_selector(dtype_include=object)),
        ('num',
         num_transformer,
         make_column_selector(dtype_include=np.number))],
    n_jobs=-1,
    verbose_feature_names_out=False)
    .set_output(transform='pandas'))

model_pipeline = Pipeline(steps=[
    ('vis_estimator', VisRatioEstimator()),
    ('pre_processor', preprocessor),
    ('reg_estimator', RandomForestRegressor(
        n_jobs=-1,
        random_state=42))])

# %%

with open('../derived/mod_07_topic_13_mlpipe.html', 'w', encoding='utf-8') as fl:
    fl.write(estimator_html_repr(model_pipeline))

with open('../models/mod_07_topic_13_mlpipe.joblib', 'wb') as fl:
    joblib.dump(model_pipeline, fl)

# %%

data.drop([
    'Item_Visibility_Avg',
    'Item_Visibility_Ratio_prev',
    'Item_Visibility_Ratio',
    'Item_Identifier_Type'],
    axis=1,
    inplace=True)

data.sample(10, random_state=42)

# %%

data.to_pickle('../derived/mod_07_topic_13_bigmart_data_upd.pkl.gz')

# %%

X_train, X_test, y_train, y_test = (
    train_test_split(
        data.drop(['Item_Identifier',
                   'Item_Outlet_Sales'],
                  axis=1),
        data['Item_Outlet_Sales'],
        test_size=0.2,
        random_state=42))

X_train.head(10)

# %%

model = model_pipeline.fit(X_train, y_train)

pred_pipe = model.predict(X_test)
rmse_pipe = root_mean_squared_error(y_test, pred_pipe)

print(f"Pipe's RMSE on test: {rmse_pipe:.1f}")

# %%

cv_results = cross_val_score(
    estimator=model_pipeline,
    X=X_train,
    y=y_train,
    scoring='neg_root_mean_squared_error',
    cv=5,
    verbose=1)

rmse_cv = np.abs(cv_results).mean()

print(f"Pipe's RMSE on CV: {rmse_cv:.1f}")
