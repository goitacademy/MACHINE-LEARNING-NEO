import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.ensemble import GradientBoostingClassifier

# %%

data, target = load_breast_cancer(return_X_y=True, as_frame=True)
data.head()

# %%

data.info()

# %%

target.value_counts()

# %%

out = (data
       .apply(lambda x:
              np.abs(zscore(x))
              .ge(3))
       .astype(int)
       .mean(1))

out_ind = np.where(out > 0.2)[0]

data.drop(out_ind, inplace=True)
target.drop(out_ind, inplace=True)

# %%

data.shape

# %%

X_train, X_test, y_train, y_test = (
    train_test_split(
        data,
        target,
        test_size=0.2,
        random_state=42))

# %%

scaler = StandardScaler().set_output(transform='pandas')

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %%

pca = PCA().set_output(transform='pandas').fit(X_train)

# %%

sns.set_theme()

explained_variance = np.cumsum(pca.explained_variance_ratio_)

ax = sns.lineplot(explained_variance)
ax.set(xlabel='number of components',
       ylabel='cumulative explained variance')

n_components = np.searchsorted(explained_variance, 0.85)

ax.axvline(x=n_components,
           c='black',
           linestyle='--',
           linewidth=0.75)

ax.axhline(y=explained_variance[n_components],
           c='black',
           linestyle='--',
           linewidth=0.75)

plt.show()

# %%

X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# %%

X_train_pca.iloc[:, :n_components].head()

# %%

plt.figure(figsize=(8, 8))

ax = plt.subplot(projection='3d')

ax.scatter3D(
    X_train_pca.iloc[:, 0],
    X_train_pca.iloc[:, 1],
    X_train_pca.iloc[:, 2],
    c=y_train,
    s=20,
    cmap='autumn',
    ec='black',
    lw=0.75)

ax.view_init(elev=30, azim=30)

plt.show()

# %%

clf_full = GradientBoostingClassifier()

clf_full.fit(X_train, y_train)

pred_full = clf_full.predict(X_test)

score_full = accuracy_score(y_test, pred_full)

print(f'Model accuracy: {score_full:.1%}')

# %%

clf_pca = GradientBoostingClassifier()

clf_pca.fit(X_train_pca.iloc[:, :n_components], y_train)

pred_pca = clf_pca.predict(X_test_pca.iloc[:, :n_components])

score_pca = accuracy_score(y_test, pred_pca)

print(f'Model accuracy (PCA): {score_pca:.1%}')

# %%

plt.figure(figsize=(3, 8))

(pd.Series(
    data=clf_full.feature_importances_,
    index=X_train.columns)
    .sort_values(ascending=True)
    .plot
    .barh())

plt.show()
