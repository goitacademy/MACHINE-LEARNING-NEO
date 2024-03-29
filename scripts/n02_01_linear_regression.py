import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# %%

california_housing = fetch_california_housing(as_frame=True)
california_housing.data.head()

# %%

california_housing.target.head()

# %%

data = california_housing.frame
data.isna().sum()

# %%

melted = california_housing.frame.melt()

sns.set_theme()

g = sns.FacetGrid(melted,
                  col='variable',
                  col_wrap=3,
                  sharex=False,
                  sharey=False)

g.map(sns.histplot, 'value')

g.set_titles(col_template='{col_name}')

g.tight_layout()

plt.show()

# %%

features_of_interest = ['AveRooms', 'AveBedrms', 'AveOccup', 'Population']
california_housing.frame[features_of_interest].describe()

# %%

fig, ax = plt.subplots(figsize=(6, 5))

sns.scatterplot(
    data=california_housing.frame,
    x='Longitude',
    y='Latitude',
    size='MedHouseVal',
    hue='MedHouseVal',
    palette='viridis',
    alpha=0.5,
    ax=ax)

plt.legend(
    title='MedHouseVal',
    bbox_to_anchor=(1.05, 0.95),
    loc='upper left')
plt.title('Median house value depending of\n their spatial location')

plt.show()

# %%

columns_drop = ['Longitude', 'Latitude']
subset = california_housing.frame.drop(columns=columns_drop)

corr_mtx = subset.corr()

mask_mtx = np.zeros_like(corr_mtx)
np.fill_diagonal(mask_mtx, 1)

fig, ax = plt.subplots(figsize=(7, 6))

sns.heatmap(subset.corr(),
            cmap='coolwarm',
            center=0,
            annot=True,
            fmt='.2f',
            linewidth=0.5,
            square=True,
            mask=mask_mtx,
            ax=ax)

plt.show()

# %%

X_train, X_test, y_train, y_test = train_test_split(
    california_housing.data,
    california_housing.target,
    test_size=0.2,
    random_state=42)

# %%

scaler = StandardScaler().set_output(transform='pandas').fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%

X_train_scaled.describe()

# %%

model = LinearRegression().fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

y_pred

# %%

r_sq = model.score(X_train_scaled, y_train)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'R2: {r_sq:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}')

# %%

pct_error = (y_pred / y_test - 1).clip(-1, 1)

sns.scatterplot(
    x=y_test,
    y=pct_error,
    hue=pct_error.gt(0),
    alpha=0.5,
    s=10,
    legend=False)

plt.show()
