import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore

# %%

california_housing = fetch_california_housing(as_frame=True)
data = california_housing['frame']

# %%

features_of_interest = ['AveRooms',
                        'AveBedrms',
                        'AveOccup',
                        'Population']

out_mask = (data[features_of_interest]
            .apply(lambda x:
                   np.abs(zscore(x)).ge(3))
            .any(axis=1))

data = data[~out_mask]

# %%

data = data.drop(['AveBedrms'], axis=1)

# %%

X_train, X_test, y_train, y_test = train_test_split(
    data.drop('MedHouseVal', axis=1),
    data['MedHouseVal'],
    test_size=0.2,
    random_state=42)

# %%

scaler = StandardScaler().set_output(transform='pandas').fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%

model = LinearRegression().fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# %%

r_sq = model.score(X_train_scaled, y_train)
mae = mean_absolute_error(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

print(f'R2: {r_sq:.2f} | MAE: {mae:.2f} | MAPE: {mape:.2f}')
