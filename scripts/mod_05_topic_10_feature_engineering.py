import pickle
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns

# %%

with open('../datasets/mod_05_topic_10_various_data.pkl', 'rb') as fl:
    datasets = pickle.load(fl)

# %%

autos = datasets['autos']

autos['stroke_ratio'] = autos['stroke'] / autos['bore']

autos[['stroke', 'bore', 'stroke_ratio']].head()

# %%

accidents = datasets['accidents']

accidents['LogWindSpeed'] = accidents['WindSpeed'].apply(np.log1p)

sns.set_theme()

fig, axs = plt.subplots(1, 2, figsize=(8, 4))
sns.kdeplot(accidents.WindSpeed, fill=True, ax=axs[0])
sns.kdeplot(accidents.LogWindSpeed, fill=True, ax=axs[1])

plt.show()

# %%

roadway_features = ['Amenity',
                    'Bump',
                    'Crossing',
                    'GiveWay',
                    'Junction',
                    'NoExit',
                    'Railway',
                    'Roundabout',
                    'Station',
                    'Stop',
                    'TrafficCalming',
                    'TrafficSignal']

accidents['RoadwayFeatures'] = accidents[roadway_features].sum(axis=1)

accidents[roadway_features + ['RoadwayFeatures']].head(10)

# %%

concrete = datasets['concrete']

components = ['Cement',
              'BlastFurnaceSlag',
              'FlyAsh',
              'Water',
              'Superplasticizer',
              'CoarseAggregate',
              'FineAggregate']

concrete['Components'] = concrete[components].gt(0).sum(axis=1)

concrete[components + ['Components']].head(10)

# %%

customer = datasets['customer']

customer[['Type', 'Level']] = (
    customer['Policy']
    .str
    .split(' ', expand=True))

customer[['Policy', 'Type', 'Level']].head(10)

# %%

autos['make_and_style'] = autos['make'] + '_' + autos['body_style']
autos[['make', 'body_style', 'make_and_style']].head()

# %%

customer['AverageIncome'] = (customer
                             .groupby('State')['Income']
                             .transform('mean'))

customer[['State', 'Income', 'AverageIncome']].head(10)

# %%

customer = (customer
            .assign(StateFreq=lambda x:
                    x.groupby('State')['State']
                    .transform('count') /
                    x['State'].count()))

customer[['State', 'StateFreq']].head(10)

# %%

c_train = customer.sample(frac=0.75)
c_test = customer.drop(c_train.index)

c_train['AverageClaim'] = (c_train
                           .groupby('Coverage')['ClaimAmount']
                           .transform('mean'))

c_test = c_test.merge(
    c_train[['Coverage', 'AverageClaim']].drop_duplicates(),
    on='Coverage',
    how='left')

c_test[['Coverage', 'AverageClaim']].head(10)

# %%

x = np.linspace(0, 2, 50)
y = np.sin(2 * np.pi * 0.25 * x)

sns.regplot(x=x, y=y)

# %%

mis = mutual_info_regression(x.reshape(-1, 1), y)[0]
cor = np.corrcoef(x, y)[0, 1]

print(f'MI score: {mis:.2f} | Cor index: {cor:.2f}')

# %%

X = autos.copy()
y = X.pop('price')

cat_features = X.select_dtypes('object').columns

for colname in cat_features:
    X[colname], _ = X[colname].factorize()

# %%

mi_scores = mutual_info_regression(
    X, y,
    discrete_features=X.columns.isin(
        cat_features.to_list() +
        ['num_of_doors',
         'num_of_cylinders']),
    random_state=42)

mi_scores = (pd.Series(
    mi_scores,
    name='MI Scores',
    index=X.columns)
    .sort_values())

mi_scores.sample(5)

# %%

plt.figure(figsize=(6, 8))
plt.barh(np.arange(len(mi_scores)), mi_scores)
plt.yticks(np.arange(len(mi_scores)), mi_scores.index)
plt.title('Mutual Information Scores')

plt.show()

# %%

sns.regplot(data=autos, x='curb_weight', y='price', order=2)

plt.show()

# %%

sns.lmplot(data=autos,
           x='horsepower',
           y='price',
           hue='fuel_type',
           facet_kws={'legend_out': False})

plt.show()
