import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_scalar
from itertools import combinations
from scipy.stats import binom


class PolynomialBasisTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, degree=5, bias=False, na_value=0., interactions=False):
        self.degree = degree
        self.bias = bias
        self.na_value = na_value
        self.interactions = interactions

    def fit(self, X, y=None):
        self.degree = check_scalar(self.degree, 'degree', int, min_val=0)
        self.bias = check_scalar(self.bias, 'bias', bool)
        self.na_value = check_scalar(self.na_value, 'na_value', float)
        self.interactions = check_scalar(self.interactions, 'interactions', bool)
        self.is_fitted_ = True
        return self

    def transform(self, X, y=None):
        check_is_fitted(self)

        X = check_array(X, estimator=self, input_name='X')

        # Get the number of columns in the input array
        n_rows, n_features = X.shape

        # Compute the specific polynomial basis for each column
        basis_features = [
            self.feature_matrix(X[:, i])
            for i in range(n_features)
        ]

        # create interaction features - basis tensor products
        if self.interactions:
            interaction_features = [
                (u[:, None, :] * v[:, :, None]).reshape(n_rows, -1)
                for u, v in combinations(basis_features, 2)
            ]
            result_basis = interaction_features
        else:
            result_basis = basis_features

        # remove the first basis function, if no bias is required
        if not self.bias:
            result_basis = [basis[:, 1:] for basis in result_basis]

        return np.hstack(result_basis)

    def feature_matrix(self, column):
        vander = self.vandermonde_matrix(column)
        return np.nan_to_num(vander, self.na_value)

    def vandermonde_matrix(self, column):
        raise NotImplementedError("Subclasses must implement this method.")


class BernsteinFeatures(PolynomialBasisTransformer):
    def vandermonde_matrix(self, column):
        basis_idx = np.arange(1 + self.degree)
        basis = binom.pmf(basis_idx, self.degree, column[:, None])
        return basis
