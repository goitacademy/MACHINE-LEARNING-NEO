import pandas as pd
import numpy as np
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import FunctionTransformer
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.pipeline import make_pipeline
from imblearn.combine import SMOTEENN

# %%

data = pd.read_csv('fp_train.csv')
test = pd.read_csv('fp_valid.csv')

# %%

# data = pd.read_csv('/kaggle/input/{competition_url}/final_proj_data.csv')
# test = pd.read_csv('/kaggle/input/{competition_url}/final_proj_test.csv')

# %%

target = data.pop('y')

# %%

to_drop = data.columns[data.isna().mean().ge(0.3)]

# %%

cat_transformer = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    TargetEncoder(random_state=42))

pre_processor = make_column_transformer(
    (cat_transformer,
     make_column_selector(dtype_include=object)),
    (SimpleImputer(strategy='median'),
     make_column_selector(dtype_include=np.number)),
    n_jobs=-1)

model = make_pipeline(
    FunctionTransformer(lambda x: x.drop(to_drop, axis=1)),
    pre_processor,
    SMOTEENN(random_state=42),
    HistGradientBoostingClassifier(random_state=42))

# %%

scores = cross_val_score(model,
                         data,
                         target,
                         scoring='balanced_accuracy',
                         cv=5,
                         n_jobs=-1)

print(f'Balanced accuracy: {np.mean(scores):.3f}')

# %%

model.fit(data, target)

# %%

predictions = model.predict(test)

submission = pd.Series(predictions, name='y').reset_index()
submission.to_csv('fp_bench.csv', index=False)

# %%

# from sklearn.metrics import balanced_accuracy_score, confusion_matrix

# truth = pd.read_csv('fp_truth.csv')
# public = truth[truth['Usage'] == 'Public'].index
# privat = truth[truth['Usage'] == 'Private'].index


# # public score
# balanced_accuracy_score(truth.loc[public, 'y'], submission.loc[public, 'y'])
# # 0.8952162655241382


# # private score
# balanced_accuracy_score(truth.loc[privat, 'y'], submission.loc[privat, 'y'])
# # 0.9184689844136011


# # all zeros
# balanced_accuracy_score(truth['y'], np.zeros_like(truth['y']))
# # 0.5


# # all ones
# balanced_accuracy_score(truth['y'], np.ones_like(truth['y']))
# # 0.5


# # random 50/50
# rnd_1 = np.random.choice([0, 1], size=len(truth), p=[0.5, 0.5])
# balanced_accuracy_score(truth['y'], rnd_1)
# # 0.498577625340462


# # random 85/15
# rnd_2 = np.random.choice(
#     [0, 1],
#     size=len(truth),
#     p=target.value_counts(normalize=True).values)
# balanced_accuracy_score(truth['y'], rnd_2)
# # 0.49126735935976323


# # auto ml
# from pycaret.classification import ClassificationExperiment
# se = ClassificationExperiment()
# se.setup(data, target=target, session_id=42)
# best = se.compare_models()
# pred = se.predict_model(best, data=test)
# balanced_accuracy_score(truth['y'], pred['prediction_label'])
# # 0.5017283701536702

