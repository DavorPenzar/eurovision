# -*- coding: utf-8 -*-

"""
Utilities for preparation of the dataset of songs.

"""

# Import SciPy packages.
import numpy as _np
import scipy.optimize as _opt

# Import TensorLy.
import tensorly as _tl
import tensorly.decomposition as _tld
import tensorly.tenalg as _tla

# Import librosa.
import librosa as _lr

# Define custom functions.

def compute_params (**params):
    """
    Compute parameters for song preprocessing.

    Parameters `hop_length`, `kernel_size`, `width` and `win_length` are
    computed from the parameter `sr` (sample rate).  See
    `librosa.effects.hpss`, `librosa.beat.beat_track`, `librosa.feature.mfcc`,
    `librosa.feature.delta`, `librosa.feature.chroma_cqt` and
    `librosa.feature.tempogram` for explanation of the parameters.

    Inspect code to see how the parameters are computed.

    Parameters
    ----------
    params
        If `sr` argument is in `params`, then its value is used; otherwise
        `22050` is used.  Other arguments are ignored and perhaps overwritten.

    Returns
    -------
    params : dict
        Input parameter `params` updated with the computed values.

    """

    # Get the sample rate.
    sr = int(params.get('sr', 22050))
    if sr <= 0:
        sr = 1

    # Compute other parameters.

    hop_length = int(round(sr / 50.0))
    if hop_length <= 0:
        hop_length = 1
    if hop_length & 0o77: # if hop_length % 64
        hop_length += 64 - (hop_length & 0o77)

    kernel_size = int(round(float(hop_length) / 16.0)) - 1
    while kernel_size < 3 or not (kernel_size & 1):
        # while kernel_size < 3 or not (kernel_size % 2)
        kernel_size += 1

    width = int(round(float(kernel_size + 1) / 4.0)) + 1
    while width < 3 or not (width & 1): # while width < 3 or not (width % 2)
        width += 1

    win_length = int(round(10.0 * sr / hop_length))
    if win_length <= 0:
        win_length = 1

    # Update `params`.
    params.update(
        {
            'sr': sr,
            'hop_length': hop_length,
            'kernel_size': kernel_size,
            'width': width,
            'win_length': win_length
        }
    )

    # Return the computed parameters.
    return params

def process_song (
    path,
    return_y = False,
    return_sr = False,
    comp = False,
    **params
):
    """
    Process a song.

    The song's chromagram, tempogram, MFCC and MFCC delta features are computed
    and returned.

    Parameters
    ----------
    path : string
        Path to the input song.  See `librosa.load` for explanation.

    return_y : boolean, optional
        If true, the raw audio time series and its separation to percussives
        and harmonics are returned as well.
    
    return_sr : boolean, optional
        If true, the sample rate is returned as well.

    comp : boolean, optional
        If true, parameters are computed using `compute_params` function.

    params
        Optional definitions of parameters `sr`, `kernel_size`, `hop_length`,
        `win_length`, `width`, `n_chroma`, `n_mfcc` and `bins_per_octave` for
        functions `librosa.effects.hpss`, `librosa.feature.chroma_cqt`,
        `librosa.feature.tempogram`, `librosa.feature.mfcc` and
        `librosa.feature.delta`.  If any of the parameters is undefined, its
        default value is used.  If `comp` is `True`, any initial values (except
        `sr` if it is not `None`) are overwritten.
        
        **Note.** Parameter `sr` may be `None` to use the original input file's
        sample rate.

    Returns
    -------
    y : (T,) numpy.ndarray
        Raw audio time series.  Dimension `T` is the length of the track in
        samples.

        *Returned only if `return_y` is true.*

    sr : int
        Sample rate.

        *Returned only if `return_sr` is true.*

    y_harmonic : (T,) numpy.ndarray
        Harmonic components of the input audio.

        *Returned only if `return_y` is true.*

    y_percussive : (T,) numpy.ndarray
        Percussive components of the input audio.

        *Returned only if `return_y` is true.*

    chromagram : (n_chroma, n) numpy.ndarray
        Chromagram matrix.

    tempogram : (win_length, n) numpy.ndarray
        Tempogram matrix.

    mfcc : (n_mfcc, n, 2) numpy.ndarray
        MFCC and MFCC delta features.  The first slice along the third axis is
        the MFCC matrix and the second slice is the MFCC delta matrix.

    """

    # Read the raw song.
    y, params['sr'] = _lr.load(
        path = path,
        sr = params.get('sr', 22050),
        mono = True,
        dtype = params.get('dtype', _np.float32)
    )
    params['sr'] = int(params['sr'])

    # If needed, compute parameters.
    if comp:
        params = compute_params(**params)

    # Separate harmonics and percussives.
    y_harmonic, y_percussive = _lr.effects.hpss(
        y = y,
        kernel_size = params.get('kernel_size', 31)
    )

    # Compute chroma features.
    chromagram = _lr.feature.chroma_cqt(
        y = y_harmonic,
        sr = params.get('sr', 22050),
        hop_length = params.get('hop_length', 512),
        norm = None,
        n_chroma = params.get('n_chroma', 12),
        bins_per_octave = params.get('bins_per_octave', 36)
    )

    # Compute tempo features.
    tempogram = _lr.feature.tempogram(
        y = y_percussive,
        sr = params.get('sr', 22050),
        hop_length = params.get('hop_length', 512),
        win_length = params.get('win_length', 384),
        norm = None
    )

    # Compute MFCC features and the first-order differences.
    mfcc = _lr.feature.mfcc(
        y = y,
        sr = params.get('sr', 22050),
        hop_length = params.get('hop_length', 512),
        n_mfcc = params.get('n_mfcc', 20)
    )
    mfcc_delta = _lr.feature.delta(data = mfcc, width = params.get('width', 9))
    mfcc = _np.concatenate(
        (
            _np.expand_dims(mfcc, mfcc.ndim),
            _np.expand_dims(mfcc_delta, mfcc_delta.ndim)
        ),
        axis = -1
    )

    # Return computed features.

    if return_y and return_sr:
        return (
            y,
            params['sr'],
            y_harmonic,
            y_percussive,
            chromagram,
            tempogram,
            mfcc
        )
    if return_y:
        return (y, y_harmonic, y_percussive, chromagram, tempogram, mfcc)
    if return_sr:
        return (params['sr'], chromagram, tempogram, mfcc)

    return (chromagram, tempogram, mfcc)

def as_audio (y):
    """
    Convert audio time series to audio data.

    Time series represented as real values from [-1, 1] is converted to
    integral values from [-32768, 32767].

    Parameters
    ----------
    y : array_like
        Input time series.  Array of real values from [-1, 1].

    Returns
    -------
    audio : numpy.ndarray
        Output audio data.  Array of integral values from [-32768, 32767]
        corresponding to input time series.

    """

    return (0o77777 * _np.asarray(y)).round().astype(_np.int16)
        # (32767 * _np.asarray(y)).round().astype(_np.int16)

def tensor_var (Y, axis = -1):
    """
    Compute variance of a sample arranged in a tensor.

    The variance of a tensor-shaped sample $ Y $ is defined as the square of
    the Frobenius norm of the tensor $ Y - \bar{Y} $ (the mean of all
    observations is subtracted from each observation, and then the transformed
    observations are joined in the transformed sample in the same way as the
    original observations were joined into $ Y $; the mean is computed
    pointwise, i. e. the "mean observation" is of the same shape as all
    original observations, but each of its elements is the mean of the elements
    at the corresponding positions in the original observations) divided by the
    appropriate denominator.  The denominator is $ n - df $, where $ n $ is the
    number of observations and $ df $ are the degrees of freedom.
    
    The variance should be equal to the total variance of the sample as
    computed via `tensor_var_svd` and `tensor_var_cov` functions; however, due
    to numeric computations some error is expected.  If only the overall
    variance is needed, this function should be preferred over the other two as
    the least amount of computation is needed.

    Parameters
    ----------
    Y : (m1, m2, ..., n, ..., mk) array
        Sample of observations in the shape of a tensor.  Let us denote the
        size of the tensor `Y` along the axis `axis` (see parameter below) as
        `n`.  Then each of the tensor `Y`'s `n` slices along the axis
        represents a single observation.

    axis : int, optional
        Axis along which the observations are joined into the tensor `Y`. If
        `axis` is negative, it counts from the last to the first axis, -1 being
        the last.

    Returns
    -------
    Var : float
        Square of the Frobenius norm of the tensor $ Y - \bar{Y} $.  This value
        can be considered as the total variance of the sample, multiplied by
        `n` minus the degrees of freedom.

    See also
    --------
    tensor_var_vec
    tensor_var_svd

    """

    return float(
        _np.sum(
            _np.square(Y - _np.mean(Y, axis = axis, keepdims = True)),
            axis = None
        )
    )

def tensor_var_vec (Y, axis = -1, **kwargs):
    """
    Compute variance of a sample arranged in a tensor.

    The covariance matrix of a sample $ Y $ is defined as the covariance matrix
    of the $ i $-mode unfolding of the tensor $ Y $ with observations being the
    rows, where $ i $ is the dimension (axis) along which the original
    observations were joined into the tensor $ Y $.  The overall (total)
    variance is then defined as the sum of variances of each variable, i. e. as
    the trace of the covariance matrix.
    
    The total variance should be equal to the total variance of the sample as
    computed via `tensor_var_svd` and `tensor_var` functions; however, due to
    numeric computations some error is expected.

    Parameters
    ----------
    Y : (m1, m2, ..., n, ..., mk) array
        Sample of observations in the shape of a tensor.  Let us denote the
        size of the tensor `Y` along the axis `axis` (see parameter below) as
        `n`.  Then each of the tensor `Y`'s `n` slices along the axis
        represents a single observation.

    axis : int, optional
        Axis along which the observations are joined into the tensor `Y`. If
        `axis` is negative, it counts from the last to the first axis, -1 being
        the last.

    kwargs
        Optional keyword arguments for `numpy.cov` function.

    Returns
    -------
    cov : (M, M) numpy.ndarray
        Covariance matrix of the `axis`-mode unfolding of the tensor `Y` with
        observations being the rows.  This matrix can be considered as the
        covariance matrix of the sample, and its trace as the total variance.

    See Also
    --------
    tensor_var
    tensor_var_svd

    """

    return _np.cov(
        _tl.unfold(Y, mode = axis),
        y = None,
        rowvar = False,
        **kwargs
    )

def tensor_var_svd (Y, axis = -1, return_d = False, **kwargs):
    """
    Compute variance of a sample arranged in a tensor.

    Inspired by the covariance matrix, variance of a tensor-shaped sample $ Y $
    is computed by conducting higher-order SVD (Tucker decomposition) on the
    tensor $ Y - \bar{Y} $ (the mean of all observation is subtracted from each
    observation, and then the transformed observations are joined in the
    transformed sample in the same way as the original observations were joined
    into $ Y $; the mean is computed pointwise, i. e. the "mean observation" is
    of the same shape as all original observations, but each of its elements is
    the mean of the elements at the corresponding positions in the original
    observations).  The variances along components (inspired by primary
    components) are then defined as squares of $ i $-mode singular values of
    the tensor $ Y - \bar{Y} $, where $ i $ is the dimension (axis) along
    which the original observations were joined into the tensor $ Y $, divided
    by the appropriate denominator.  The denominator is $ n - df $, where $ n $
    is the number of observations and $ df $ are the degrees of freedom.

    Parameters
    ----------
    Y : (m1, m2, ..., n, ..., mk) array
        Sample of observations in the shape of a tensor.  Let us denote the
        size of the tensor `Y` along the axis `axis` (see parameter below) as
        `n`.  Then each of the tensor `Y`'s `n` slices along the axis
        represents a single observation.

    axis : int, optional
        Axis along which the observations are joined into the tensor `Y`. If
        `axis` is negative, it counts from the last to the first axis, -1 being
        the last.

    return_d : boolean, optional
        If true, the decomposition of the tensor $ Y - \bar{Y} $ is returned as
        well.
  
    kwargs
        Optional keyword arguments for `tensorly.decomposition.tucker`
        function.

    Returns
    -------
    S : (m1, m2, ..., n, ..., mk) numpy.ndarray
        Core tensor from the decomposition of the tensor $ Y - \bar{Y} $.

        *Returned only if `return_d` is true.*

    U : tuple of (m1, m1), (m2, m2,), ..., (n, n), ..., (mk, mk) numpy.ndarrays
        Factor matrices from the decomposition of the tensor $ Y - \bar{Y} $.

        *Returned only if `return_d` is true.*

    sv2 : (n,) numpy.ndarray
        Squares of the $ i $-mode singular values of the tensor $ Y - \bar{Y} $
        in decreasing order.  These values can be considered as the variances
        of the sample along its primary components, and their sum as the total
        variance, multiplied by `n` minus the degrees of freedom.

    See Also
    --------
    tensor_var
    tensor_var_vec

    """

    # Compute the decomposition of `Y` - `mean(Y)`.
    S, U = _tld.tucker(Y - _np.mean(Y, axis = axis, keepdims = True), **kwargs)

    # Compute variances (squares of `axis`-mode singular values).
    sv2 = _np.array(
        list(
            _np.sum(_np.square(s), axis = None)
                for s in _np.moveaxis(S, axis, 0)
        )
    )

    # Return computed arrays.

    if return_d:
        return (S, U, sv2)

    return sv2

def diverse_sample (
    Y,
    size,
    axis = -1,
    n_iter = 100,
    early_stop = 16,
    random_state = None,
    return_ind = False,
    return_var = False
):
    """
    Generate a sample as diverse as possible.

    Diversity of a sample is measured by its variance (greater variance means
    greater diversity).  Variance of a tensor-shaped sample is computed using
    `tensor_var` function.

    The function produces many random samples and returns the sample that was
    the most diverse.  Because of this, the returned sample may be suboptimal,
    i. e. chances are a more diverse sample exists.  By setting a larger value
    for `n_iter` more samples are tried which may result in a more diverse
    sample, but the algorithm will probably take longer to finish.

    Parameters
    ----------
    Y : (m1, m2, ..., n, ..., mk) array
        Sample of observations in the shape of a tensor.  Let us denote the
        size of the tensor `Y` along the axis `axis` (see parameter below) as
        `n`.  Then each of the tensor `Y`'s `n` slices along the axis
        represents a single observation.

    size : int
        Size of the sample of `Y` to generate.  It is assumed that `size` > 1.

    axis : int, optional
        Axis along which the observations are joined into the tensor `Y`. If
        `axis` is negative, it counts from the last to the first axis, -1 being
        the last.

    n_iter : int, optional
        (Maximal) number of iterations of the algorithm.

    early_stop : None or int, optional
        If set (if not `None`) and if the variance does not improve in
        `early_stop` consequent iterations, the algorithm breaks early.
        Otherwise all `n_iter` iterations are run.

    random_state : None or numpy.random.RandomState, optional
        Random state of the algorithm (for reproducibility of results).

    return_ind : boolean, optional
        If true, indices of the subsample (in the original sample) are returned
        as well.

    return_var : boolean : optional
        If true, variance of the subsample is returned as well.

    Returns
    -------
    ind : (size,) numpy.ndarray
        Indices of the returned sample in the original sample.

        *Returned only if `return_ind` is true.*

    Z : (m1, m2, ..., size, ..., mk) numpy.ndarray
        The most diverse sample found.

    var : float
        Variance of the returned sample multiplied by `size` minus the degrees
        of freedom.

        *Returned only if `return_var` is true.*

    See also
    --------
    tensor_var
    diverse_sample_opt

    """

    # Prepare parameters.

    Y = _np.moveaxis(Y, axis, 0)

    if early_stop is None:
        early_stop = n_iter
    if random_state is None:
        random_state = _np.random

    # Initialise values.
    max_ind = None
    max_Z = None
    max_var = -float('inf')

    # Iteratively find the most diverse subsample.

    j = 0
    for i in range(n_iter):
        # Check if no improvement has happend in the last `early_stop`
        # iterations.
        if j >= early_stop:
            break

        # Extract and inspect a new subsample.
        ind = random_state.choice(Y.shape[0], size, replace = False)
        var = tensor_var(Y[ind], axis = 0)

        # Compare the new subsample.
        if var > max_var:
            max_ind = ind
            max_var = var

            j = 0
        else:
            j += 1
    max_ind.sort()
    max_Z = _np.moveaxis(Y[max_ind], 0, axis)

    # Return computed values.

    if return_ind and return_var:
        return (max_ind, max_Z, max_var)
    if return_ind:
        return (max_ind, max_Z)
    if return_var:
        return (max_Z, max_var)

    return max_Z

def diverse_sample_opt (
    Y,
    size,
    axis = -1,
    random_state = None,
    return_ind = False,
    return_var = False,
    **kwargs
):
    """
    Generate a sample as diverse as possible.

    Diversity of a sample is measured by its variance (greater variance means
    greater diversity).  Variance of a tensor-shaped sample is computed using
    `tensor_var` function.

    Unlike `diverse_sample` function, this function does not produce many
    random samples in order to find the optimal one; rather, it uses
    `scipy.optimize.minize` function for the optimisation.

    Parameters
    ----------
    Y : (m1, m2, ..., n, ..., mk) array
        Sample of observations in the shape of a tensor.  Let us denote the
        size of the tensor `Y` along the axis `axis` (see parameter below) as
        `n`.  Then each of the tensor `Y`'s `n` slices along the axis
        represents a single observation.

    size : int
        Size of the sample of `Y` to generate.  It is assumed that `size` > 1.

    axis : int, optional
        Axis along which the observations are joined into the tensor `Y`. If
        `axis` is negative, it counts from the last to the first axis, -1 being
        the last.

    random_state : None or numpy.random.RandomState, optional
        Random state of the algorithm (for reproducibility of results).  This
        is only used to produce the initial guess for `scipy.optimize.minimize`
        function.

    return_ind : boolean, optional
        If true, indices of the subsample (in the original sample) are returned
        as well.

    return_var : boolean : optional
        If true, variance of the subsample is returned as well.

    kwargs
        Optional keyword arguments for `scipy.optimize.minimize` function.

    Returns
    -------
    ind : (size,) numpy.ndarray
        Indices of the returned sample in the original sample.

        *Returned only if `return_ind` is true.*

    Z : (m1, m2, ..., size, ..., mk) numpy.ndarray
        The most diverse sample found.

    var : float
        Variance of the returned sample multiplied by `size` minus the degrees
        of freedom.

        *Returned only if `return_var` is true.*

    See also
    --------
    tensor_var
    diverse_sample

    """

    # Prepare parameters.

    Y = _np.moveaxis(Y, axis, 0)

    if random_state is None:
        random_state = _np.random

    # Find the most diverse subsample.
    max_ind = _np.sort(
        _opt.minimize(
            lambda ind: -tensor_var(Y[ind.round().astype(int)], axis = 0),
            random_state.choice(Y.shape[0], size, replace = False),
            bounds = _opt.Bounds(0, Y.shape[0] - 1),
            constraints = {
                'type': 'ineq',
                'fun': \
                    lambda ind: \
                        _np.diff(_np.sort(ind.round().astype(int))).min()
            },
            **kwargs
        ).x
    ).round().astype(int)
    max_Z = _np.moveaxis(Y[max_ind], 0, axis)
    max_var = None
    if return_var:
        max_var = tensor_var(max_Z, axis = axis)

    # Return computed values.

    if return_ind and return_var:
        return (max_ind, max_Z, max_var)
    if return_ind:
        return (max_ind, max_Z)
    if return_var:
        return (max_Z, max_var)

    return max_Z

def split_sample (
    Xs,
    size = [0.70, 0.15, 0.15],
    axis = -1,
    n_iter = 100,
    early_stop = 16,
    diff_weights = None,
    random_state = None,
    return_ind = False
):
    """
    Split a sample by preserving variance.

    Given an iterable `Xs` of samples (see parameter below) aranged in tensors
    all of which are of the same size along the axis `axis` and a 1-dimensional
    array `size` of subsamples' sizes, the samples are split into subsamples
    of given sizes along the provided axis by preserving the original variances
    as closely as possible.  If `size` is an array of weights, such as
    `[0.5, 0.5]`, actual sizes are first rounded up (if exactly between two
    integers) by resolving sizes from beginning to end.  For example, if the
    samples in `Xs` are of size 11, and `size` is `[0.5, 0.5]`, subsamples of
    sizes 6, 5 are generated.

    Actually, all given `sizes` are treated as positive weights, therefore
    given an array of sizes `[2, 3, 5]`, the resulting subsamples may not be of
    actual sizes 2, 3 and 5, but 0.2, 0.3 and 0.5 times the size of the
    original sample.

    The function produces many random subsamples and returns the sample that
    had the variance (see `tensor_var`) the closest to the original samples.
    Because of this, the returned subsample may be suboptimal, i. e. chances
    are better subsamples exists.  By setting a larger value for `n_iter` more
    samples are tried which may result in better subsamples, but the algorithm
    will probably take longer to finish.

    Parameters
    ----------
    Xs : iterable of array_like
        Original samples in the form of tensors, all of which are of the same
        size along the axis `axis` (see parameter below).  Let us denote the
        of the tensors along the axis `axis` as `n`.  Then each of the tensors'
        `n` slices along the axis represent a single observation.

        Passing more than one sample set is enabled to allow splitting a
        dataset according to both inputs and outputs (for example, a dataset
        of `n` inputs of shape `(256, 256)` as matrices and `n` 1-dimensional
        outputs as labels).  However, this is generalised to enable passing an
        arbitrary number of parts of observations in a sample.

    size : (m,) array, optional
        Weighted sizes of the resulting subsamples.  The resulting subsamples'
        sizes are computed by rounding the values of
        `size / numpy.sum(size) * n` so that the sum equals to `n`.

    axis : int, optional
        Axis along which the observations are joined into the tensors in `Xs`.
        If `axis` is negative, it counts from the last to the first axis, -1
        being the last.

    n_iter : int, optional
        (Maximal) number of iterations of the algorithm.

    early_stop : None or int, optional
        If set (if not `None`) and if the difference in variances does not
        improve in `early_stop` consequent iterations, the algorithm breaks
        early.  Otherwise all `n_iter` iterations are run.

    diff_weights : None or array_like, optional
        If provided, it must be broadcastable to an array of shape `(k, m)`,
        where `k` is the number of tensors in `Xs` and `m` is the number of
        sizes in `size`.  The element of the `(m, k)`-array at position
        `(i, j)` represents the normalised difference of the subsample's
        variance and the original sample's variance.  The normalised difference
        is computed as `(v - v0) / max(v0, 1)`, where `v` is the subsample's
        (subsample of the `j`-th size extracted from the `i`-th tensor)
        variance and `v0` is the original sample's (the `i`-th tensor)
        variance.  The mean of squared normalised differences is then computed
        and minimalised through the algorithm.  If `diff_weights` is provided
        (if it is not `None`), the weighted average is used where
        `diff_weights` are the weights.

        **Note.** True sample variances are used, meaning the result of the
        `tensor_var` function is divided by the number of observations
        decremented by 1.

    random_state : None or numpy.random.RandomState, optional
        Random state of the algorithm (for reproducibility of results).

    return_ind : boolean, optional
        If true, indices of the subsample (in the original sample) are returned
        as well.

    Raises
    ------
    ValueError
        If `Xs` is empty or the tensors are not of the same size along the axis
        `axis`.

    Returns
    -------
    subsamples : tuple of tuple of numpy.ndarray
        Tuple of tuples of the generated subsamples.  The `j`-th tensor in the
        `i`-th tuple is a subsample of the `j`-th size extracted from the
        `i`-th original tensor (sample part), represented as a `numpy.ndarray`.

    ind : tuple of (m,) numpy.ndarray
        Indices of the returned subsamples in the original samples.  The `j`-th
        array represents the indices of the subsamples of the `j`-th size.

        *Returned only if `return_ind` is true.*

    See Also
    --------
    tensor_var
    split_sample_opt

    """

    # Prepare parameters.

    Xs = tuple(_np.moveaxis(X, axis, 0) for X in Xs)
    if not len(Xs) or not all(X.shape[0] == Xs[0].shape[0] for X in Xs):
        raise ValueError('Either no tensor is provided or they are not of the same size along the provided axis.')

    if early_stop is None:
        early_stop = n_iter
    if diff_weights is None:
        diff_weights = 1
    if random_state is None:
        random_state = _np.random

    size = _np.asarray(size).ravel()
    size /= _np.flip(_np.flip(size).cumsum())

    m = int(Xs[0].shape[0])
    n = list()
    for u in size:
        n.append(int(round(u * m)))
        m -= n[-1]
    del size
    del m
    n = _np.array(n, dtype = int)
    r = _np.concatenate(([0], n.cumsum()))

    v = _np.expand_dims([tensor_var(X) / (X.shape[0] - 1) for X in Xs], 1)
    vd = _np.maximum(v, 1.0)

    # Initialise values.
    min_ind = None
    min_Xs = None
    min_d = float('inf')

    # Iteratively find the optimal split into subsamples.

    j = 0
    for i in range(n_iter):
        # Check if no improvement has happend in the last `early_stop`
        # iterations.
        if j >= early_stop:
            break

        # Extract and inspect new subsamples.
        ind = random_state.permutation(Xs[0].shape[0])
        d = _np.sum(
            diff_weights * _np.square(
                (
                    [
                        [
                            tensor_var(X[ind[r[a]:r[a + 1]]]) / (n[a] - 1)
                                for a in range(int(n.size))
                        ] for X in Xs
                    ] - v
                ) / vd
            ),
            axis = None
        )

        # Compare the new subsamples.
        if d < min_d:
            min_ind = ind
            min_d = d

            j = 0
        else:
            j += 1
    min_ind = tuple(
        _np.sort(min_ind[r[a]:r[a + 1]]) for a in range(int(n.size))
    )
    min_Xs = tuple(
        tuple(_np.moveaxis(X[I], 0, axis) for I in min_ind) for X in Xs
    )

    # Return computed values.

    if return_ind:
        return (min_ind, min_Xs)

    return min_Xs

def split_sample_opt (
    Xs,
    size = [0.70, 0.15, 0.15],
    axis = -1,
    n_iter = 100,
    early_stop = 16,
    diff_weights = None,
    random_state = None,
    return_ind = False,
    **kwargs
):
    """
    Split a sample by preserving variance.

    Given an iterable `Xs` of samples (see parameter below) aranged in tensors
    all of which are of the same size along the axis `axis` and a 1-dimensional
    array `size` of subsamples' sizes, the samples are split into subsamples
    of given sizes along the provided axis by preserving the original variances
    as closely as possible.  If `size` is an array of weights, such as
    `[0.5, 0.5]`, actual sizes are first rounded up (if exactly between two
    integers) by resolving sizes from beginning to end.  For example, if the
    samples in `Xs` are of size 11, and `size` is `[0.5, 0.5]`, subsamples of
    sizes 6, 5 are generated.

    Actually, all given `sizes` are treated as positive weights, therefore
    given an array of sizes `[2, 3, 5]`, the resulting subsamples may not be of
    actual sizes 2, 3 and 5, but 0.2, 0.3 and 0.5 times the size of the
    original sample.

    Unlike `diverse_sample` function, this function does not produce many
    random samples in order to find the optimal one; rather, it uses
    `scipy.optimize.minize` function for the optimisation.  However, this
    approach (as used in this function) may be worse since a permutation of the
    array `list(range(0, n))` instead of its (sub)sample.

    Parameters
    ----------
    Xs : iterable of array_like
        Original samples in the form of tensors, all of which are of the same
        size along the axis `axis` (see parameter below).  Let us denote the
        of the tensors along the axis `axis` as `n`.  Then each of the tensors'
        `n` slices along the axis represent a single observation.

        Passing more than one sample set is enabled to allow splitting a
        dataset according to both inputs and outputs (for example, a dataset
        of `n` inputs of shape `(256, 256)` as matrices and `n` 1-dimensional
        outputs as labels).  However, this is generalised to enable passing an
        arbitrary number of parts of observations in a sample.

    size : (m,) array, optional
        Weighted sizes of the resulting subsamples.  The resulting subsamples'
        sizes are computed by rounding the values of
        `size / numpy.sum(size) * n` so that the sum equals to `n`.

    axis : int, optional
        Axis along which the observations are joined into the tensors in `Xs`.
        If `axis` is negative, it counts from the last to the first axis, -1
        being the last.

    diff_weights : None or array_like, optional
        If provided, it must be broadcastable to an array of shape `(k, m)`,
        where `k` is the number of tensors in `Xs` and `m` is the number of
        sizes in `size`.  The element of the `(m, k)`-array at position
        `(i, j)` represents the normalised difference of the subsample's
        variance and the original sample's variance.  The normalised difference
        is computed as `(v - v0) / max(v0, 1)`, where `v` is the subsample's
        (subsample of the `j`-th size extracted from the `i`-th tensor)
        variance and `v0` is the original sample's (the `i`-th tensor)
        variance.  The mean of squared normalised differences is then computed
        and minimalised through the algorithm.  If `diff_weights` is provided
        (if it is not `None`), the weighted average is used where
        `diff_weights` are the weights.

        **Note.** True sample variances are used, meaning the result of the
        `tensor_var` function is divided by the number of observations
        decremented by 1.

    random_state : None or numpy.random.RandomState, optional
        Random state of the algorithm (for reproducibility of results).  This
        is only used to produce the initial guess for `scipy.optimize.minimize`
        function.

    return_ind : boolean, optional
        If true, indices of the subsample (in the original sample) are returned
        as well.

    kwargs
        Optional keyword arguments for `scipy.optimize.minimize` function.

    Raises
    ------
    ValueError
        If `Xs` is empty or the tensors are not of the same size along the axis
        `axis`.

    Returns
    -------
    subsamples : tuple of tuple of numpy.ndarray
        Tuple of tuples of the generated subsamples.  The `j`-th tensor in the
        `i`-th tuple is a subsample of the `j`-th size extracted from the
        `i`-th original tensor (sample part), represented as a `numpy.ndarray`.

    ind : tuple of (m,) numpy.ndarray
        Indices of the returned subsamples in the original samples.  The `j`-th
        array represents the indices of the subsamples of the `j`-th size.

        *Returned only if `return_ind` is true.*

    See Also
    --------
    tensor_var
    split_sample

    """

    # Prepare parameters.

    Xs = tuple(_np.moveaxis(X, axis, 0) for X in Xs)
    if not len(Xs) or not all(X.shape[0] == Xs[0].shape[0] for X in Xs):
        raise ValueError('Either no tensor is provided or they are not of the same size along the provided axis.')

    if diff_weights is None:
        diff_weights = 1
    if random_state is None:
        random_state = _np.random

    size = _np.asarray(size).ravel()
    size /= _np.flip(_np.flip(size).cumsum())

    m = int(Xs[0].shape[0])
    n = list()
    for u in size:
        n.append(int(round(u * m)))
        m -= n[-1]
    del size
    del m
    n = _np.array(n, dtype = int)
    r = _np.concatenate(([0], n.cumsum()))

    v = _np.expand_dims([tensor_var(X) / (X.shape[0] - 1) for X in Xs], 1)
    vd = _np.maximum(v, 1.0)

    def indices_objective_function (ind):
        ind = ind.round().astype(int)

        return _np.sum(
            diff_weights * _np.square(
                (
                    [
                        [
                            tensor_var(X[ind[r[a]:r[a + 1]]]) / (n[a] - 1)
                                for a in range(int(n.size))
                        ] for X in Xs
                    ] - v
                ) / vd
            ),
            axis = None
        )

    # Find the optimal split into subsamples.
    min_ind = _opt.minimize(
        indices_objective_function,
        random_state.permutation(Xs[0].shape[0]),
        bounds = _opt.Bounds(0, Xs[0].shape[0] - 1),
        constraints = {
                'type': 'ineq',
                'fun': \
                    lambda ind: \
                        _np.diff(_np.sort(ind.round().astype(int))).min()
            },
        **kwargs
    ).x.round().astype(int)
    min_ind = tuple(
        _np.sort(min_ind[r[a]:r[a + 1]]) for a in range(int(n.size))
    )
    min_Xs = tuple(
        tuple(_np.moveaxis(X[I], 0, axis) for I in min_ind) for X in Xs
    )

    # Return computed values.

    if return_ind:
        return (min_ind, min_Xs)

    return min_Xs
