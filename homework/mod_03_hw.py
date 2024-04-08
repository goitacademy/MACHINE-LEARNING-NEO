import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# %%

data = pd.read_csv('../datasets/mod_03_topic_05_weather_data.csv.gz')

# %%

data = data[data.columns[data.isna().mean().lt(0.35)]]
data = data.dropna(subset='RainTomorrow')

# %%

data_num = data.select_dtypes(include=np.number)
data_cat = data.select_dtypes(include='object')

# %%

data_cat['Date'] = pd.to_datetime(data['Date'])

data_cat[['Year', 'Month']] = (data_cat['Date']
                               .apply(lambda x:
                                      pd.Series([x.year, x.month])))

data_cat.drop('Date', axis=1, inplace=True)

data_cat['Month'] = data_cat['Month'].astype(str)

# %%

data_num['Year'] = data_cat.pop('Year')

# %%

train_mask = data_num['Year'] == data_num['Year'].max()

# %%

X_train_num, X_test_num, X_train_cat, X_test_cat, y_train, y_test = (
    data_num[~train_mask],
    data_num[train_mask],
    data_cat.drop('RainTomorrow', axis=1)[~train_mask],
    data_cat.drop('RainTomorrow', axis=1)[train_mask],
    data.loc[~train_mask, 'RainTomorrow'],
    data.loc[train_mask, 'RainTomorrow']
)

# %%

num_imputer = SimpleImputer().set_output(transform='pandas')
X_train_num = num_imputer.fit_transform(X_train_num)
X_test_num = num_imputer.transform(X_test_num)

# %%

cat_imputer = SimpleImputer(
    strategy='most_frequent').set_output(transform='pandas')
X_train_cat = cat_imputer.fit_transform(X_train_cat)
X_test_cat = cat_imputer.transform(X_test_cat)

# %%

scaler = StandardScaler().set_output(transform='pandas')

X_train_num = scaler.fit_transform(X_train_num)
X_test_num = scaler.transform(X_test_num)

# %%

encoder = (OneHotEncoder(drop='if_binary',
                         sparse_output=False)
           .set_output(transform='pandas'))

X_train_cat = encoder.fit_transform(X_train_cat)
X_test_cat = encoder.transform(X_test_cat)

# %%

X_train = pd.concat([X_train_num, X_train_cat], axis=1)
X_test = pd.concat([X_test_num, X_test_cat], axis=1)

# %%

clf = (LogisticRegression(
    solver='newton-cholesky',
    class_weight='balanced',
    random_state=42)
    .fit(X_train, y_train))

# %%

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
