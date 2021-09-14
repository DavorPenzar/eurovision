# -*- coding: utf-8 -*-

"""
Implementation of weighted PCA inspired by [scikit-learn's](http://scikit-learn.org/) [`PCA`](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

This script is a part of Davor Penzar's *[ESC](http://eurovision.tv/) Score
Predictor* project.

Author: [Davor Penzar `<davor.penzar@gmail.com>`](mailto:davor.penzar@gmail.com)
Date: 2021-09-14
Version: 1.0

"""

import copy as _copy
import math as _math

import numpy as _np

class WPCA (object):
    """
    Weighted principal component analysis.

    The algorithm is implemented using [NumPy's](http://numpy.org/) methods.

    Parameters
    ----------
    n_components : None or int, optional
    Number of principal components to use.  If not set, all components are
    used.

    weights : None or (m,) array, optional
    Non-negative weights of samples.  At least one weight must be strictly
    greater than 0.  If not set, all samples are weighted equally (`m ** -1`).

    dtype : dtype, optional
    Data type to use.

    See Also
    sklearn.decomposition.PCA

    """

    # Define which algorithms to use for specific functionalities.

    _sum = _math.fsum
    #_sum = _np.sum

    #_matmul = _np.matmul
    _matmul = _np.dot

    #_eig = _np.linalg.eig
    _eig = _np.linalg.eigh

    # Define class methods.

    @classmethod
    def weighted_cov (
        cls,
        X,
        weights,
        ddof = 1,
        dtype = _np.float32,
        return_mean = False
    ):
        """
        Compute weighted covariance matrix of a real-valued sample.

        Parameters
        ----------
        X : (m, n) array
        Matrix sample of real-valued vectors.  Each row represents a single
        observation, expressed through features represented by columns.

        weights : (m,) array
        Non-negative weights of samples in X.  At least one weight must be
        strictly greater than 0.

        ddof : int, optional
        Degrees of freedom.  The denominator for computing covariances is
        `m - ddof`.

        dtype : dtype, optional
        Data type of the output array(s).

        return_mean : boolean
        If true, the weighted mean of samples in `X` is returned as well.

        Returns
        -------
        mean : (1, n) array
        The weighted mean of samples in `X`.

        *Returned only if `return_mean` is true.*

        cov : (n, n) array
        The weighted covariance matrix.

        """

        # Prepare parameters.

        X = _np.asarray(X)

        weights = _np.asarray(weights).ravel()

        assert _np.all(weights >= 0)
        assert _np.any(weights != 0)

        weights /= cls._sum(weights)

        n_samples = int(X.shape[0])
        n_features = int(X.shape[1])

        n_free_samples = n_samples - int(ddof)
        full_weights = n_samples * weights

        # Compute the mean and the covariance matrix.

        mean = _np.sum(
            _np.expand_dims(weights, axis = 1) * X,
            axis = 0,
            dtype = dtype,
            keepdims = True
        )

        X_dev = X - mean

        cov = _np.zeros((n_features, n_features), dtype = dtype)

        for i in range(n_features):
            for j in range(i, n_features):
                c = _np.sum(
                    full_weights * (X_dev[:, i] * X_dev[:, j]),
                    axis = None,
                    dtype = dtype
                ) / n_free_samples

                cov[i, j] = c
                cov[j, i] = c

                del c

        del X_dev

        del n_free_samples
        del full_weights

        del n_samples
        del n_features

        # Return the computed values.

        ret = list()
        if return_mean:
            ret.append(mean)
        ret.append(cov)

        return ret[0] if len(ret) == 1 else tuple(ret)

    def __new__ (cls, *args, **kwargs):
        instance = super(WPCA, cls).__new__(cls)

        instance._n_features = None
        instance._n_samples = None
        instance._n_components = None

        instance._weights = None

        instance._mean = None

        instance._explained_variance = None
        instance._explained_variance_ratio = None

        #instance._singular_values = None
        instance._components = None

        return instance

    def __init__ (
        self,
        n_components = None,
        weights = None,
        dtype = _np.float32
    ):
        super(WPCA, self).__init__()

        self._dtype = _np.dtype(dtype)

        if weights is None:
            self._weights = None
        else:
            weights_arr = _np.asarray(weights)

            assert (
                weights.ndim == 1 and
                weights_arr.size and
                _np.all(weights_arr >= 0) and
                _np.any(weights_arr != 0)
            )

            self._weights = _np.array(
                weights_arr / WPCA._sum(weights_arr),
                dtype = self._dtype
            )

            del weights_arr

            assert _np.all(self._weights >= 0)

        if n_components is None:
            self._n_components = None
        else:
            n_components_int = int(n_components)

            assert n_components_int > 0

            self._n_components = n_components_int

    def fit (self, X, y = None):
        """
        Fit components.

        Parameters
        ----------
        X : (m, n) array
        Matrix sample of real-valued vectors.  Each row represents a single
        observation, expressed through features represented by columns.  The
        number of samples (`m`) must be the same as the length of `weights_`,
        the number of features (`n`) must be greater than or equal to
        `n_components_`.

        y : ignored
        Included only for compatibility with `sklearn` objects.

        Returns
        -------
        self : WPCA

        """

        assert self._components is None

        # Prepare parameters.

        X = _np.asarray(X)

        self._n_features = int(X.shape[1])
        self._n_samples = int(X.shape[0])

        if self._n_components is None:
            self._n_components = _copy.deepcopy(self._n_features)

        if self._weights is None:
            self._weights = _np.full(
                X.shape[0],
                fill_value = 1.0 / float(self._n_samples),
                dtype = self._dtype,
                order = 'C'
            )

        assert self._n_components <= self._n_features
        assert self._weights.size == self._n_samples

        n_free_samples = self._n_samples - 1
        full_weights = self._n_samples * self._weights

        # Compute the mean and the covariance matrix.

        self._mean = _np.sum(
            _np.expand_dims(self._weights, axis = 1) * X,
            axis = 0,
            dtype = self._dtype,
            keepdims = True
        )

        X_dev = X - self._mean

        cov = _np.zeros(
            (self._n_features, self._n_features),
            dtype = self._dtype
        )

        for i in range(self._n_features):
            for j in range(i, self._n_features):
                c = _np.sum(
                    full_weights * (X_dev[:, i] * X_dev[:, j]),
                    axis = None,
                    dtype = self._dtype
                ) / n_free_samples

                cov[i, j] = c
                cov[j, i] = c

                del c

        del X_dev

        # Compute eigendecomposition of `cov`.

        explained_variance, components = WPCA._eig(cov)
        total_variance = WPCA._sum(explained_variance)
        explained_variance_ratio = explained_variance / total_variance
        
        I = _np.flip(_np.argsort(explained_variance))

        explained_variance = explained_variance[I]
        explained_variance_ratio = explained_variance_ratio[I]
        components = components[:, I]

        del I

        self._explained_variance = _np.array(
            explained_variance[:self._n_components],
            dtype = self._dtype
        )
        self._explained_variance_ratio = _np.array(
            explained_variance_ratio[:self._n_components],
            dtype = self._dtype
        )
        self._components = _np.array(
            components[:, :self._n_components],
            dtype = self._dtype,
            order = 'C'
        )

        del explained_variance
        del explained_variance_ratio
        del total_variance
        del components

        del cov

        # Return `self`.
        return self

    def transform (self, X):
        """
        Transform a sample.

        Parameters
        ----------
        X : (k, n) array
        Matrix sample of real-valued vectors.  Each row represents a single
        observation, expressed through features represented by columns.  The
        number of features (`n`) must be the same as `n_features_`.

        Returns
        -------
        T : (k, n_components_) array
        Matrix sample of transformed real-valued vectors from `X`.

        """

        assert self._components is not None

        return WPCA._matmul(X, self._components)

    def fit_transform (self, X, y = None):
        """
        Fit components and transform a sample.

        Parameters
        ----------
        X : (m, n) array
        Matrix sample of real-valued vectors.  Each row represents a single
        observation, expressed through features represented by columns.  The
        number of samples (`m`) must be the same as the length of `weights_`,
        the number of features (`n`) must be greater than or equal to
        `n_components_`.

        y : ignored
        Included only for compatibility with `sklearn` objects.

        Returns
        -------
        T : (k, n_components_) array
        Matrix sample of transformed real-valued vectors from `X`.

        """

        X = _np.asarray(X)

        return self.fit(X, y).transform(X)

    @property
    def n_features_ (self):
        """Number of original features (components)."""
        return _copy.deepcopy(self._n_features)

    @property
    def n_samples_ (self):
        """Number of samples in the original sample."""
        return _copy.deepcopy(self._n_samples)

    @property
    def n_components_ (self):
        """Number of principal components used."""
        return _copy.deepcopy(self._n_components)

    @property
    def weights_ (self):
        """Original weights (normed to the sum of 1)."""
        return _copy.deepcopy(self._weights)

    @property
    def mean_ (self):
        """Weighted mean of the original sample."""
        return _copy.deepcopy(self._mean)

    @property
    def explained_variance_ (self):
        """Absolute variances per principal components."""
        return _copy.deepcopy(self._explained_variance)

    @property
    def explained_variance_ratio_ (self):
        """Relative variances per principal components."""
        return _copy.deepcopy(self._explained_variance_ratio)

#    @property
#    def singular_values_ (self):
#        return _copy.deepcopy(self._singular_values)

    @property
    def components_ (self):
        """Principal components arranged in rows."""
        return _copy.deepcopy(self._components)
