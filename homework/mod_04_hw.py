import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import TargetEncoder, PowerTransformer

# %%

data = pd.read_csv('../datasets/mod_04_hw_train_data.csv')
test = pd.read_csv('../datasets/mod_04_hw_test_data.csv')

# %%

data = data.dropna()

# %%

cols_to_drop = ['Name', 'Phone_Number', 'Date_Of_Birth']

# %%

data = data.drop(['Name', 'Phone_Number', 'Date_Of_Birth'], axis=1)

# %%

cats = data.select_dtypes(include='object').columns

enc = (TargetEncoder(
    target_type='continuous',
    random_state=42)
    .set_output(transform='pandas'))

data[cats] = enc.fit_transform(data[cats], data['Salary'])

# %%

trn = PowerTransformer().set_output(transform='pandas')

data.iloc[:, :-1] = trn.fit_transform(data.iloc[:, :-1])

# %%

model = KNeighborsRegressor()

model.fit(data.iloc[:, :-1], data.iloc[:, -1])

# %%

y_true = test.pop('Salary')

# %%

test = test.drop(cols_to_drop, axis=1)
test[cats] = enc.transform(test[cats])
test = trn.transform(test)

# %%

y_pred = model.predict(test)

mean_absolute_percentage_error(y_true, y_pred)
