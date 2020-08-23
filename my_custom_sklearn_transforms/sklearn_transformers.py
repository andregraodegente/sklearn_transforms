from sklearn.base import BaseEstimator, TransformerMixin
from autogluon import TabularPrediction

# All sklearn Transforms must have the `transform` and `fit` methods
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class StringColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        for col in self.columns:
            data[col] = data[col].apply(lambda x: str(x) if x == x else "")
        return data

class MeanColumns(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Primeiro realizamos a cópia do dataframe 'X' de entrada
        data = X.copy()
        data.fillna(data.mean(), inplace=True)
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data



class CustomCatBoostClassifier(CatBoostClassifier):

    def fit(self, X, y):
        return super().fit(
            X,
            y=y,
            cat_features=list(range(0, X.shape[1])) ,
            verbose=True
        ) 