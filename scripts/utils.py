from sklearn.base import BaseEstimator, TransformerMixin


# %%

class VisRatioEstimator(BaseEstimator, TransformerMixin):

    def __init__(self):
        self.vis_avg = None

    def fit(self, X, y=None):
        vis_avg = (X
                   .groupby(['Item_Type',
                             'Outlet_Type'])['Item_Visibility']
                   .mean())

        self.vis_avg = vis_avg

        return self

    def transform(self, X, y=None):

        X = X.copy()

        X['Item_Visibility_Ratio'] = (
            X
            .groupby(['Item_Type',
                      'Outlet_Type'])['Item_Visibility']
            .transform(lambda x:
                       x / self.vis_avg[x.name]))

        return X
