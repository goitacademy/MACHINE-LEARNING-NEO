import pickle
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import TargetEncoder
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# %%

with open('../datasets/mod_05_topic_10_various_data.pkl', 'rb') as fl:
    datasets = pickle.load(fl)

autos = datasets['autos']

# %%

X = autos.copy()
y = X.pop('price')

cat_features = X.select_dtypes('object').columns

for colname in cat_features:
    X[colname], _ = X[colname].factorize()

# %%

cat_features = cat_features.to_list() + ['num_of_doors',
                                         'num_of_cylinders']

# %%

mi_scores = mutual_info_regression(
    X, y,
    discrete_features=X.columns.isin(
        cat_features),
    random_state=42)

mi_scores = (pd.Series(
    mi_scores,
    name='MI Scores',
    index=X.columns)
    .sort_values())

# %%

data = autos.copy()
enc = TargetEncoder(target_type='continuous',
                    random_state=42).set_output(transform='pandas')

data[cat_features] = enc.fit_transform(data[cat_features], y)

# %%

regr = GradientBoostingRegressor(random_state=42)
regr.fit(data.drop('price', axis=1), y)

# %%

rf_scores = pd.Series(
    regr.feature_importances_,
    index=X.columns,
    name='GB Scores')

# %%

rf_scores = rf_scores.rank(pct=True)
mi_scores = mi_scores.rank(pct=True)

# %%

scores = pd.concat([rf_scores, mi_scores], axis=1)
scores = (scores
          .melt(ignore_index=False)
          .reset_index()
          .sort_values(['variable', 'value'],
                       ascending=[False, False]))

# %%

sns.catplot(
    data=scores,
    orient='h',
    x='value',
    y='index',
    hue='variable',
    kind='bar')

plt.show()
