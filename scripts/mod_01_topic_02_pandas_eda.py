import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from ydata_profiling import ProfileReport

# %%

planets = sns.load_dataset('planets')
planets.shape

# %%

planets.head()

# %%

rng = np.random.RandomState(42)
ser = pd.Series(rng.rand(5))
ser

# %%

ser.sum()

# %%

ser.mean()

# %%

df = pd.DataFrame({'A': rng.rand(5),
                   'B': rng.rand(5)})
df

# %%

df.mean()

# %%

df.mean(axis='columns')

# %%

planets.dropna().describe()

# %%

df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data': range(6)}, columns=['key', 'data'])
df

# %%

df.groupby('key')

# %%

df.groupby('key').sum()

# %%

planets.groupby('method')['orbital_period'].median()

# %%

planets.groupby('method')['year'].describe()

# %%

rng = np.random.RandomState(0)
df = pd.DataFrame({'key': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'data1': range(6),
                   'data2': rng.randint(0, 10, 6)},
                  columns=['key', 'data1', 'data2'])
df

# %%

df.groupby('key').agg(['min', 'median', 'max'])

# %%

df.groupby('key').agg({'data1': 'min',
                       'data2': 'max'})

# %%


def center(x):
    return x - x.mean()


df.groupby('key').transform(center)

# %%


def norm_by_data2(x):
    # x is a DataFrame of group values
    x['data1'] /= x['data2'].sum()
    return x


df.groupby('key').apply(norm_by_data2, include_groups=False)

# %%

decade = 10 * (planets['year'] // 10)
decade = decade.astype(str) + 's'
decade.name = 'decade'

planets.groupby(['method', decade])['number'].sum().unstack().fillna(0)

# %%

titanic = sns.load_dataset('titanic')
titanic.head()

# %%

titanic.groupby('sex')[['survived']].mean()

# %%

(titanic
 .groupby(['sex', 'class'],
          observed=True)['survived']
 .mean()
 .unstack())

# %%

titanic.pivot_table('survived', index='sex', columns='class', observed=True)

# %%

age = pd.cut(titanic['age'], [0, 18, 80])

titanic.pivot_table('survived', ['sex', age], 'class', observed=True)

# %%

fare = pd.qcut(titanic['fare'], 2)

titanic.pivot_table('survived', ['sex', age], [fare, 'class'],  observed=True)

# %%

with warnings.catch_warnings():
    warnings.simplefilter('ignore')

    report = ProfileReport(
        titanic,
        title='Titanic')

    report.to_file('../derived/mod_01_topic_02_titanic_report.html')
