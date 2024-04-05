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
import pickle
import pickle
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
# from kneed import KneeLocator
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# %%

with open('../datasets/mod_05_topic_10_various_data.pkl', 'rb') as fl:
    datasets = pickle.load(fl)

# with open('../datasets/mod_05_topic_10_various_data.pkl', 'wb') as fl:
#     pickle.dump(datasets, fl)


# datasets = {}

# autos = pd.read_csv(r"C:\Users\holomb\Desktop\archive\autos.csv")
# autos.drop('symboling', axis=1, inplace=True)
# datasets['autos'] = autos

# datasets['concrete'] = data

# customer = pd.read_csv(r"C:\Users\holomb\Desktop\archive\customer.csv")
# customer.drop('Unnamed: 0', axis=1, inplace=True)
# datasets['customer'] = customer


# accidents = pd.read_csv(r"C:\Users\holomb\Desktop\archive\accidents.csv")
# customer.drop('Unnamed: 0', axis=1, inplace=True)
# datasets['accidents'] = accidents

data = datasets['concrete']


# %%

X = StandardScaler().fit_transform(data)


model = KMeans(random_state=42)

visualizer = KElbowVisualizer(
    model,
    k=(5, 15),
    timings=False)

visualizer.fit(X)
visualizer.show()

# %%

k_best = visualizer.elbow_value_

model_kmn = KMeans(n_clusters=k_best, random_state=42).fit(X)
labels_kmn = pd.Series(model_kmn.labels_, name='k-means')

ss = data.groupby(labels_kmn).mean()


pca = PCA(n_components=5, random_state=42).fit(X)
viz = pca.transform(X)

pve = pca.explained_variance_ratio_

# %%

sns.scatterplot(x=X[:, 0],
                y=X[:, 2],
                hue=labels_kmn,
                # edgecolor='black',
                # linewidth=0.5,
                # s=60,
                palette='tab20',
                legend=False,
                # ax=ax
                )

# ax.set(title=s.name)
