import warnings
from tempfile import mkdtemp
import mlflow
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline

# %%

mlflow.set_tracking_uri(uri='http://127.0.0.1:8080')
mlflow.set_experiment('MLflow Tracking')

# %%

data = pd.read_pickle('../derived/mod_07_topic_13_bigmart_data_upd.pkl.gz')

X, y = (data.drop(['Item_Identifier',
                   'Item_Outlet_Sales'],
                  axis=1),
        data['Item_Outlet_Sales'])

# %%

with open('../models/mod_07_topic_13_mlpipe.joblib', 'rb') as fl:
    pipe_base = joblib.load(fl)

# %%

cv_results = cross_val_score(
    estimator=pipe_base,
    X=X,
    y=y,
    scoring='neg_root_mean_squared_error',
    cv=5)

rmse_cv = np.abs(cv_results).mean()

# %%

model_base = pipe_base.fit(X, y)

# %%

params_base = pipe_base.named_steps['reg_estimator'].get_params()

# %%

# Start an MLflow run
with mlflow.start_run(run_name='rfr'):
    # Log the hyperparameters
    mlflow.log_params(params_base)
    # Log the loss metric
    mlflow.log_metric('cv_rmse_score', rmse_cv)
    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag('Model', 'RandomForest for BigMart')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # Infer the model signature
        signature = mlflow.models.infer_signature(
            X.head(),
            model_base.predict(X.head()))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=model_base,
        artifact_path='model_base',
        signature=signature,
        input_example=X.head(),
        registered_model_name='model_base_tracking')

# %%

pipe_upd = Pipeline(
    steps=pipe_base.steps[:-1] +
    [('reg_model',
      GradientBoostingRegressor(random_state=42))],
    memory=mkdtemp())

# %%

parameters = {
    'reg_model__learning_rate': (0.1, 0.3),
    'reg_model__subsample': (0.75, 0.85),
    'reg_model__max_features': ('sqrt', 'log2')}

search = (GridSearchCV(
    estimator=pipe_upd,
    param_grid=parameters,
    scoring='neg_root_mean_squared_error',
    cv=5,
    refit=False)
    .fit(X, y))

# %%

parameters_best = search.best_params_
pipe_upd = pipe_upd.set_params(**parameters_best)

model_upd = pipe_upd.fit(X, y)

# %%

cv_results_upd = cross_val_score(
    estimator=pipe_upd,
    X=X,
    y=y,
    scoring='neg_root_mean_squared_error',
    cv=5)

rmse_cv_upd = np.abs(cv_results_upd).mean()

# %%

with mlflow.start_run(run_name='gbr'):
    mlflow.log_params(pipe_upd.named_steps['reg_model'].get_params())
    mlflow.log_metric('cv_rmse_score', rmse_cv_upd)
    mlflow.set_tag('Model', 'GradientBoosting model for BigMart')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        signature = mlflow.models.infer_signature(
            X.head(),
            model_upd.predict(X.head()))

    model_info = mlflow.sklearn.log_model(
        sk_model=model_upd,
        artifact_path='model_upd',
        signature=signature,
        input_example=X.head(),
        registered_model_name='model_upd_tracking')

# %%

# best_run = (mlflow
#             .search_runs(
#                 experiment_names=['MLflow Tracking'],
#                 order_by=['metrics.cv_rmse_score'],
#                 max_results=1))

# best_run[['tags.Model', 'metrics.cv_rmse_score']]
