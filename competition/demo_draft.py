import warnings
import pandas as pd
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import (
    PowerTransformer,
    KBinsDiscretizer,
    TargetEncoder)
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

# %%

train = pd.read_csv('demo_train.csv')
valid = pd.read_csv('demo_valid.csv')

# train = pd.read_csv('/kaggle/input/customer-churn-prediction-2020/train.csv')
# valid = pd.read_csv('/kaggle/input/customer-churn-prediction-2020/test.csv')

# %%

target = train.pop('churn')

# %%

model = make_pipeline(
    make_column_transformer(
        (TargetEncoder(random_state=42),
         make_column_selector(dtype_include=object)),
        remainder='passthrough',
        n_jobs=-1),
    SelectKBest(),
    PowerTransformer(),
    SMOTE(random_state=42),
    KBinsDiscretizer(
        encode='onehot-dense',
        strategy='uniform',
        subsample=None,
        random_state=42),
    GradientBoostingClassifier(
        random_state=42))

# %%

# model.get_params().keys()

params = {
    'selectkbest__k': [10, 15],
    'smote__k_neighbors': [7, 9],
    'gradientboostingclassifier__subsample': [0.65, 0.85],
    'gradientboostingclassifier__max_depth': [5, 7]
}

rs = RandomizedSearchCV(
    model,
    params,
    n_jobs=-1,
    refit=False,
    random_state=42,
    verbose=1)

search = rs.fit(train, target)
search.best_params_

# %%

model.set_params(**search.best_params_)

# %%

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    cv_results = cross_val_score(
        estimator=model,
        X=train,
        y=target,
        scoring='balanced_accuracy',
        cv=10,
        n_jobs=-1)

cv_results.mean()

# %%

model.fit(train, target)

# %%

output = pd.DataFrame({'id': valid['id'],
                       'churn': model.predict(valid)})

# output.to_csv('submission.csv', index=False)
