from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans

# %%

with open('../datasets/mod_05_topic_10_various_data.pkl', 'rb') as fl:
    datasets = pickle.load(fl)

data = datasets['concrete']

# %%

components = ['Cement',
              'BlastFurnaceSlag',
              'FlyAsh',
              'Water',
              'Superplasticizer',
              'CoarseAggregate',
              'FineAggregate']

data['Components'] = data[components].gt(0).sum(axis=1)

# %%

X = StandardScaler().fit_transform(data)

# %%

visualizer = KElbowVisualizer(
    KMeans(random_state=42),
    k=(5, 15),
    timings=False)

visualizer.fit(X)
visualizer.show()

# %%

k_best = visualizer.elbow_value_

model = KMeans(n_clusters=k_best, random_state=42).fit(X)

# %%

data['Count'] = data.groupby(model.labels_).transform('size')

# %%

report = (data
          .groupby(model.labels_)
          .median()
          .sort_values(['Components', 'Count']))
