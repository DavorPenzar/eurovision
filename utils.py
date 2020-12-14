# -*- coding: utf-8 -*-

"""
Utilities for preparation of the dataset of songs.

"""

# Import SciPy packages.
import numpy as _np
import scipy.optimize as _spo

# Import TensorLy.
import tensorly as _tl
import tensorly.decomposition as _tld
import tensorly.tenalg as _tla

# Import librosa.
import librosa as _lr

# Define custom functions.


##  AUDIO MANIPULATION

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
    dict
        Input parameter `params` updated with the computed values.

    See Also
    librosa.effects.hpss
    librosa.feature.chroma_cqt
    librosa.feature.tempogram
    librosa.feature.mfcc
    librosa.feature.delta

    """

    # Get the sample rate.
    sr = params.get('sr', 22050)
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

    win_length = int(round(10.0 * sr / hop_length))
    if win_length <= 0:
        win_length = 1

    width = int(round(float(kernel_size + 1) / 4.0)) + 1
    while width < 3 or not (width & 1): # while width < 3 or not (width % 2)
        width += 1

    # Update `params`.
    params.update(
        {
            'sr': sr,
            'hop_length': hop_length,
            'kernel_size': kernel_size,
            'win_length': win_length,
            'width': width
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

    See Also
    --------
    librosa.load
    librosa.effects.hpss
    librosa.feature.chroma_cqt
    librosa.feature.tempogram
    librosa.feature.mfcc
    librosa.feature.delta
    compute_params

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

    ret = list()
    if return_y:
        ret.append(y)
        if return_sr:
            ret.append(params['sr'])
        ret.append(y_harmonic)
        ret.append(y_percussive)
    elif return_sr:
        ret.append(params['sr'])
    ret.append(chromagram)
    ret.append(tempogram)
    ret.append(mfcc)

    return tuple(ret)

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
    numpy.ndarray
        Output audio data.  Array of integral values from [-32768, 32767]
        corresponding to input time series.

    """

    return (0o77777 * _np.asarray(y)).round().astype(_np.int16)
        # (32767 * _np.asarray(y)).round().astype(_np.int16)


##  TENSOR VARIANCE

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
    float
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
    (M, M) numpy.ndarray
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

def tensor_var_svd (Y, axis = -1, return_d = False, tucker_kwargs = dict()):
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
  
    kwargs : dict, optional
        Optional keyword arguments for `tensorly.decomposition.tucker`
        function (except parameter `tensor`).

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
    S, U = _tld.tucker(
        Y - _np.mean(Y, axis = axis, keepdims = True),
        **tucker_kwargs
    )

    # Compute variances (squares of `axis`-mode singular values).
    sv2 = _np.array(
        list(
            _np.sum(_np.square(s), axis = None)
                for s in _np.moveaxis(S, axis, 0)
        )
    )

    # Return computed arrays.

    ret = list()
    if return_d:
        ret.append(S)
        ret.append(U)
    ret.append(sv2)

    return ret[0] if len(ret) == 1 else tuple(ret)


##  TENSOR SAMPLES

def _truncate_indices (a, ind, axis = -1):
    """
    Truncate non-negative 1-dimensional indices to fit indexing of an array.

    The function assumes 1-dimensional indices and returns them truncated into
    range [0, `m`) for `m` being the length of a given array.  If any index is
    negative, it is truncated to 0.

    Parameters
    ----------
    a : (m, ...) array_like
        Input array.

    ind : (n,) array_like
        1-dimensional array of non-negative indices.

    Returns
    -------
    (n,) numpy.ndarray
        1-dimensional array of the same size as `ind` with values from `ind`
        truncated into range [0, `m`) (actually, range [0, `m` - 1]).

    Notes
    -----
    Reduced flexibility of the function (restrictions on the indexing axis, the
    sign and the dimensionality of the index) is intentional since the function
    is not intended to be used outsie of the library.  The function is, in
    fact, merely an auxiliary internal function.

    See Also
    --------
    numpy.round

    """

    # Prepare parameters.
    a = _np.asarray(a)
    ind = _np.asarray(ind).ravel()

    # Truncate and return indices.
    return _np.where(
        ind >= 0,
        _np.where(
            ind < a.shape[0],
            ind,
            a.shape[0] - 1
        ),
        0
    )

def _round_indices (ind):
    """
    Round real-valued 1-dimensional indices to nearest integers.

    The function assumes 1-dimensional indices and returns them rounded to
    nearest integral indices.  The rounding is computed using `numpy.round`
    function.

    Parameters
    ----------
    ind : (n,) array_like
        1-dimensional array of real-valued indices.

    Returns
    -------
    (n,) numpy.ndarray
        1-dimensional array of the same size as `ind` with values from `ind`
        rounded to nearest integers.  The data type (`dtype`) of the array is
        an integral data type.

    Notes
    -----
    Reduced flexibility of the function (restriction on the the dimensionality
    of the index) is intentional since the function is not intended to be used
    outsie of the library.  The function is, in fact, merely an auxiliary
    internal function.

    See Also
    --------
    numpy.round

    """

    return _np.round(ind).ravel().astype(_np.integer)

def _minimal_integral_index_difference (ind):
    """
    Compute the minimal difference of real non-negative indices rounded to nearest integers.

    The function assumes non-negative 1-dimensional indices and returns the
    minimal absolute difference between their values rounded to nearest
    integers.  The rounding is computed using `_round_indices` function.  The
    differences are not only computed between consecutive indices but between
    each pair.

    Parameters
    ----------
    ind : (n,) array_like
        1-dimensional array of real-valued non-negative indices.

    Returns
    -------
    int
        The minimal absolute difference of values in `ind` rounded to nearest
        integers.

    Notes
    -----
    Reduced flexibility of the function (restrictions on the the sign and the
    dimensionality of the index) is intentional since the function is not
    intended to be used outsie of the library.  The function is, in fact,
    merely an auxiliary internal function.

    See Also
    --------
    _round_indices

    """

    return _np.diff(_np.sort(_round_indices(ind))).min()

def _interpolating_indexed_array (a, ind):
    """
    Index an array along its first axis with non-integral indices by interpolating its values.

    The interpolation method is a linear spline.  An element at position
    `n + q`, where `n` is an integer and `q` is a real number such that
    `0 <= q < 1` is computed as `a[n] + q * (a[n + 1] - a[n])` (the case when
    `n` is the last index and `q == 0` is also handled successfully).

    Parameters
    ----------
    a : (m, ...) array_like
        Input array.

    ind : (n,) array_like
        1-dimensional array of real-valued indices.  Both
        `numpy.floor(ind).astype(int)` and `numpy.ceil(ind).astype(int)` must
        be valid indices (positive or negative) for the array `a` along the
        first axis.

    Returns
    -------
    (n, ...) numpy.ndarray
        Array constructed from the values of the input array `a` and the
        indexing array `ind` along the first axis by generating values at
        non-existing indices through interpolation of the input array `a`.

    Notes
    -----
    If both `a` and `ind` are integral-valued arrays (of an integral data type
    (`dtype`)), the resulting array will also be integral-valued; otherwise its
    data type (`dtype`) will be of a floating point type (real or complex,
    according to values in `a`).  However, regardless if `a` or `ind` are
    integral-valued or not, the size of the output array's data type (`dtype`)
    may increase (for instance, from `numpy.int16` to `numpy.int32` or from
    `numpy.complex64` to `numpy.complex128`).

    Reduced flexibility of the function (restrictions on the indexing axis and
    the dimensionality of the index) is intentional since the function is not
    intended to be used outsie of the library.  The function is, in fact,
    merely an auxiliary internal function.

    See Also
    --------
    numpy.floor
    numpy.ceil

    """

    # Prepare parameters.
    a = _np.asarray(a)
    ind = _np.asarray(ind).ravel()

    # Compute auxiliary indices.
    floor_ind = _np.floor(ind).astype(_np.integer)
    ceil_ind = _np.ceil(ind).astype(_np.integer)
    frac_ind = ind - floor_ind
    while frac_ind.ndim < a.ndim:
        frac_ind = _np.expand_dims(frac_ind, frac_ind.ndim)

    # Compute and return the resulting array.
    return a[floor_ind] + frac_ind * (a[ceil_ind] - a[floor_ind])

def _index_optimisation_bounds (a):
    """
    Generate bounds for the optimisation of non-negative indices of an array along the first axis.

    The result of the function can be used for bounds in `scipy.optimize`
    algorithms, such as the parameter `bounds` of `scipy.optimize.minimize`
    function.

    Parameters
    ----------
    a : array_like
        The original array whose indices should be optimised.

    Returns
    -------
    scipy.optimize.Bounds
        Bounds for the optimisation of indices of the array `a`.  The lower
        bound is 0 and the upper bound is `a.shape[0]` - 1.

    Notes
    -----
    Reduced flexibility of the function (restrictions on the indexing axis and
    the dimensionality of the index) is intentional since the function is not
    intended to be used outsie of the library.  The function is, in fact,
    merely an auxiliary internal function.

    See Also
    --------
    scipy.optimize.minimize

    """

    return _spo.Bounds(0, _np.asarray(a).shape[0] - 1)

def _index_optimisation_constraint (**kwargs):
    """
    Generate constraint for the optimisation of non-negative indices of an array along a single axis.

    The result of the function can be used for constraints in `scipy.optimize`
    algorithms, such as the parameter `constraints` of
    `scipy.optimize.minimize` function.

    The constraint assumes non-negative 1-dimensional indices, evaluates them
    using `_minimal_integral_index_difference` function (parameter `fun` of the
    constraint), and checks if the value is in range [1, inf) (parameters `lb`
    and `ub` of the constraint respectively).  For instance, indices
    `[0.49, 0.51]` would be accepted since they would be rounded to integral
    indices `[0, 1]`, but indices `[0.51, 1.49]` would be rejected because they
    would be rounded to integral indices `[1, 1]`.

    Parameters
    ----------
    kwargs
        Optional keyword arguments for `scipy.optimize.NonlinearConstraint`
        function.

    Returns
    -------
    scipy.optimize.NonlinearConstraint
        Constraint for the optimisation of indices of an array.  It evaluates
        the indices as the minimal absolute difference between their values
        rounded to nearest integers, and compares the result to lower bound 1
        and upper bound `inf`.

    Notes
    -----
    Reduced flexibility of the function (restrictions on the sign and the
    dimensionality of the index) is intentional since the function is not
    intended to be used outsie of the library.  The function is, in fact,
    merely an auxiliary internal function.

    See Also
    --------
    scipy.optimize.minimize
    _minimal_integral_index_difference

    """

    return _spo.NonlinearConstraint(
        _minimal_integral_index_difference,
        1,
        float('inf'),
        **kwargs
    )

def diverse_sample (
    Y,
    size,
    axis = -1,
    n_iter = 100,
    early_stop = 16,
    random_state = None,
    return_ind = False,
    return_var = False,
    return_nit = False
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

    return_var : boolean, optional
        If true, variance of the subsample is returned as well.

    return_nit : boolean, optional
        If true, number of iterations run is returned as well.

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

    nit : int
        Number of iterations run.

        *Returned only if `return_nit` is true.*

    See also
    --------
    numpy.random.Random
    tensor_var
    diverse_sample_optd
    diverse_sample_optc

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

    nit = 0
    j = 0
    for nit in range(n_iter):
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

    ret = list()
    if return_ind:
        ret.append(max_ind)
    ret.append(max_Z)
    if return_var:
        ret.append(max_var)
    if return_nit:
        ret.append(nit)

    return ret[0] if len(ret) == 1 else tuple(ret)

def diverse_sample_optd (
    Y,
    size,
    axis = -1,
    random_state = None,
    return_ind = False,
    return_raw = False,
    constraint_kwargs = dict(),
    optimize_kwargs = dict()
):
    """
    Generate a sample as diverse as possible.

    Diversity of a sample is measured by its variance (greater variance means
    greater diversity).  Variance of a tensor-shaped sample is computed using
    `tensor_var` function.

    Unlike `diverse_sample` function, this function does not produce many
    random samples in order to find the optimal one; rather, it uses
    `scipy.optimize.minize` function for the optimisation.  However, unlike
    `diverse_sample_optc` function, at each step it rounds the indices to
    nearest integers and then checks the variance.  The optimisation-based
    approach may be suboptimal since the underlying domain (choice of indices)
    is descrete.

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

        **Note.** The random state is used only to generate the initial guess
        for `scipy.optimize.minimize` function.  For true reproducibility check
        arguments `constraint_kwargs` and `optimize_kwargs`.

    return_ind : boolean, optional
        If true, indices of the subsample (in the original sample) are returned
        as well.  Check *Notes* to see how the result is transformed into
        actual indices.

    return_raw : boolean, optional
        If true, the raw result of `scipy.optimize.minimize` function call is
        returned as well.

    constraint_kwargs : dict, optional
        Optional keyword arguments for `scipy.optimize.NonlinearConstraint`
        class (except parameters `fun`, `lb` and `ub`).  The non-linear
        constraint is used to check if any pair of indices (not necessarily
        integral) rounds to the nearest integer.  This is done by rounding the
        indices, computing the minimal absolute difference between the values
        and checking if it is greater than or equal to 1.

    optimize_kwargs : dict, optional
        Optional keyword arguments for `scipy.optimize.minimize` function
        (except parameters `fun`, `x0`, `args`, `bounds` and `constraints`).

    Returns
    -------
    ind : (size,) numpy.ndarray
        Indices of the returned sample in the original sample.

        *Returned only if `return_ind` is true.*

    Z : (m1, m2, ..., size, ..., mk) numpy.ndarray
        The most diverse sample found.

    raw : scipy.optimize.OptimizeResult
        The result of `scipy.optimize.minimze` function call.

        *Returned only if `return_raw` is true.*

    Notes
    -----
    The result of the optimisation (returned value `raw`) is converted to an
    actual index-array by rounding values of `raw.x` to integers using
    `numpy.round` function and truncating them to fit into range
    [0, `a.shape(axis)`) (also, they are sorted in the end).  This is done
    regardless of the value of `raw.success`, meaning the indices may not be in
    fact optimised.  As a result, it is not guaranteed that all indices are
    mutually different and the user is advised to check the returned values
    themselves.

    See also
    --------
    scipy.optimize.NonlinearConstraint
    scipy.optimize.minimize
    scipy.optimize.OptimizeResult
    numpy.random.Random
    numpy.round
    tensor_var
    diverse_sample
    diverse_sample_optc

    """

    # Prepare parameters.

    Y = _np.moveaxis(Y, axis, 0)

    if random_state is None:
        random_state = _np.random

    # Find the most diverse subsample.
    res = _spo.minimize(
        lambda ind: -tensor_var(
            Y[_round_indices(_truncate_indices(ind, Y))],
            axis = 0
        ),
        random_state.choice(Y.shape[0], size, replace = False),
        tuple(),
        bounds = _index_optimisation_bounds(Y),
        constraints = _index_optimisation_constraint(**constraint_kwargs),
        **optimize_kwargs
    )
    max_ind = _round_indices(_truncate_indices(Y, res.x))
    max_Z = _np.moveaxis(Y[max_ind], 0, axis)

    # Return computed values.

    ret = list()
    if return_ind:
        ret.append(max_ind)
    ret.append(max_Z)
    if return_raw:
        ret.append(res)

    return ret[0] if len(ret) == 1 else tuple(ret)

def diverse_sample_optc (
    Y,
    size,
    axis = -1,
    random_state = None,
    return_ind = False,
    return_raw = False,
    constraint_kwargs = dict(),
    optimize_kwargs = dict()
):
    """
    Generate a sample as diverse as possible.

    Diversity of a sample is measured by its variance (greater variance means
    greater diversity).  Variance of a tensor-shaped sample is computed using
    `tensor_var` function.

    Unlike `diverse_sample` function, this function does not produce many
    random samples in order to find the optimal one; rather, it uses
    `scipy.optimize.minize` function for the optimisation.  However, unlike
    `diverse_sample_optd` function, the indices are not rounded until the end
    of the optimisation but samples are generated from non-integral indices
    through interpolation of the values from the original tensor using a linear
    spline.  This may be valid approach if the original tensor represents a
    relatively dense discretisation of a smooth function (progression of
    values), but the optimisation-based approach may still be suboptimal since
    the underlying domain (choice of indices) is descrete.

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

        **Note.** The random state is used only to generate the initial guess
        for `scipy.optimize.minimize` function.  For true reproducibility check
        arguments `constraint_kwargs` and `optimize_kwargs`.

    return_ind : boolean, optional
        If true, indices of the subsample (in the original sample) are returned
        as well.  Check *Notes* to see how the result is transformed into
        actual indices.

    return_raw : boolean, optional
        If true, the raw result of `scipy.optimize.minimize` function call is
        returned as well.

    constraint_kwargs : dict, optional
        Optional keyword arguments for `scipy.optimize.NonlinearConstraint`
        class (except parameters `fun`, `lb` and `ub`).  The non-linear
        constraint is used to check if any pair of indices (not necessarily
        integral) rounds to the nearest integer.  This is done by rounding the
        indices, computing the minimal absolute difference between the values
        and checking if it is greater than or equal to 1.

    optimize_kwargs : dict, optional
        Optional keyword arguments for `scipy.optimize.minimize` function
        (except parameters `fun`, `x0`, `args`, `bounds` and `constraints`).

    Returns
    -------
    ind : (size,) numpy.ndarray
        Indices of the returned sample in the original sample.

        *Returned only if `return_ind` is true.*

    Z : (m1, m2, ..., size, ..., mk) numpy.ndarray
        The most diverse sample found.

    raw : scipy.optimize.OptimizeResult
        The result of `scipy.optimize.minimize` function call.

        *Returned only if `return_raw` is true.*

    Notes
    -----
    The result of the optimisation (returned value `raw`) is converted to an
    actual index-array by rounding values of `raw.x` to integers using
    `numpy.round` function and truncating them to fit into range
    [0, `a.shape(axis)`) (also, they are sorted in the end).  This is done
    regardless of the value of `raw.success`, meaning the indices may not be in
    fact optimised.  As a result, it is not guaranteed that all indices are
    mutually different and the user is advised to check the returned values
    themselves.

    See also
    --------
    scipy.optimize.NonlinearConstraint
    scipy.optimize.minimize
    scipy.optimize.OptimizeResult
    numpy.random.Random
    numpy.round
    tensor_var
    diverse_sample
    diverse_sample_optd

    """

    # Prepare parameters.

    Y = _np.moveaxis(Y, axis, 0)

    if random_state is None:
        random_state = _np.random

    # Find the most diverse subsample.
    res = _spo.minimize(
        lambda ind: -tensor_var(
            _interpolating_indexed_array(Y, _truncate_indices(Y, ind)),
            axis = 0
        ),
        random_state.choice(Y.shape[0], size, replace = False),
        bounds = _index_optimisation_bounds(Y),
        constraints = _index_optimisation_constraint(**constraint_kwargs),
        **optimize_kwargs
    )
    max_ind = _round_indices(_truncate_indices(Y, res.x))
    max_Z = _np.moveaxis(Y[max_ind], 0, axis)

    # Return computed values.

    ret = list()
    if return_ind:
        ret.append(max_ind)
    ret.append(max_Z)
    if return_raw:
        ret.append(res)

    return ret[0] if len(ret) == 1 else tuple(ret)

### Sample splitting

def _absolute_subsample_sizes (total, size):
    """
    Compute absolute sizes of subsamples from size weights.

    Given the number `total` of elements in the original sample and an array of
    size weights `size`, the absolute weights of subsamples are computed.  The
    absolute sizes are obtained by multiplying the wights and the total number
    (`total * size`) and then rounding using built-in `round` function.  The
    absolute sizes are resolved from the first to last, therefore, in case of
    ties, sizes at the beginning will be rounded up, while sizes at the end
    will be rounded down.  For instance, if `total` is 13 and `size` is
    `[0.5, 0.5]`, the output sizes will be `[7, 6]`.

    Parameters
    ----------
    total : int
        The total number (non-negative) of elements in the original sample.

    size : array_like
        1-dimensional real-valued array of non-negative elements representing
        the weights of sizes of subsamples.  The weights do not need to sum up
        to 1 because they are normalised (by dividing via
        `numpy.true_divide(size, numpy.sum(size))`) regardless of the original
        values.

    Returns
    -------
    int_size : numpy.ndarray
        1-dimensional integral-valued array of absolute sizes corresponding to
        input weights (input array `size`).

    border : numpy.ndarray
        1-dimensional integral-valued array of border indices of the
        subsamples.  The following equations are true: `border[0] == 0`,
        `border[-1] == total` and
        `numpy.array_equal(numpy.diff(border), int_size)` (the `i`-th subsample
        can be obtained by using the elements at positions from `border[i]` to
        `border[i + 1]` in the original sample).

    Notes
    -----
    Reduced flexibility of the function (restrictions on the sign and the
    dimensionality of the sizes) is intentional since the function is not
    intended to be used outsie of the library.  The function is, in fact,
    merely an auxiliary internal function.

    See Also
    --------
    round
    numpy.sum
    numpy.true_divide
    numpy.diff
    numpy.array_equal

    """

    # Compute cummulative relative sizes.
    size = _np.asarray(size).ravel()
    size = _np.true_divide(size, _np.flip(_np.flip(size).cumsum()))

    # Compute absolute sizes and borders.
    n = list()
    for u in size:
        n.append(round(u * total))
        total -= n[-1]
    del size
    n = _np.array(n, dtype = _np.integer)
    r = _np.insert(n.cumsum(), 0, 0)

    # Return computed arrays.
    return (n, r)

def _samples_variance_difference (Xs, var, var_dn, size, border, ind):
    """
    Compute differences in variances between subsamples and the original (total) tensor sample along the first axis.

    Given an iterable `Xs` of samples (see parameter below) aranged in tensors
    all of which are of the same size along the first axis, and a permutation
    `ind` of its indices along the first axis (see parameter below), along with
    a few additional parameters, a matrix of differences in variances between
    subsamples and the original samples is computed and returned.

    Let `k` be the number of tensors in `Xs` and `m` the number of subsamples
    for each `X` in `Xs` (length of the array `size`).
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

    Parameters
    ----------
    Xs : iterable of (n, ...) array_like
        Original samples in the form of tensors, all of which are of the same
        size along the first axis.  Let us denote the number of tensors in `Xs`
        as `m` and the size of the tensors along the first `axis` as `n`.  Then
        each of the tensors' `n` slices along the axis represent a single
        observation.

    var : (m,) array_like
        Variances of tensors in `Xs`.  The variances must be **true**
        variances, meaning `var[i]` should be
        `tensor_var(Xs[i], axis = 0) / (m - 1)` (the result of `tensor_var`
        function evaluated at the `i`-th tensor divided by the length of the
        tensor minus 1).

    var_dn : (m,) array_like
        *Variances* of tensors in `Xs` used for division.  This parameter
        should be `numpy.maximum(var, 1)`.

    size : (k,) array_like
        1-dimensional integral-valued array of sizes of subsamples such that
        each size is greater than or equal to 2 and that
        `numpy.sum(size) == n`.

    border : (k + 1,) array_like
        1-dimensional array of integral-valued border-indices for subsamples.
        The following equalities should be satisfied: `border[0] == 0`,
        `border[-1] == n` and `numpy.array_equal(numpy.diff(border), size)`.

    ind : (n,) array_like
        1-dimensional array of integral-valued indices to extract subsamples.
        This parameter should actually be a permutation of `list(range(m))`
        (all of the legal indices for tensors along their first axis with no
        duplicates).

    Returns
    -------
    (k, m) numpy.ndarray
        A real-valued matrix such that the element at position `(i, j)`
        represents the normalised difference of the `i`-th subsample's of the
        `j`-th tensor variance and the original sample's (`j`-th tensor's)
        variance.  The difference is **not** absolute, it may be negative or
        positive.

    Notes
    -----
    The `i`-th subsample of the `j`-tensor is actually
    `Xs[j][ind[border[i]:border[i + 1]]]`.  That is, the index `ind` is
    divided into parts of sizes provided through the parameter `size`, and its
    `i`-th part is used as indices for the `i`-th subsamples.  To extract the
    `i`-th subsample of the `j`-th tensor, this part of indices is then applied
    to the `j`-th tensor.

    The normalised variance of a subsample is computed by subtracting the
    original sample's variance from the subsample's variance, and dividing the
    difference by `max(var[j], 1)` (`var[j]` is the variance of the `j`-th
    tensor).  The variances are true variances, menaning the result of
    `tensor_var` function is divided by the appropriate tensor's (original or
    the subsample) length.

    Reduced flexibility of the function (restrictions on the indexing axis) is
    intentional since the function is not intended to be used outsie of the
    library.  The function is, in fact, merely an auxiliary internal function.

    See Also
    --------
    numpy.maximum
    numpy.sum
    numpy.diff
    numpy.array_equal
    tensor_var

    """

    return (
        [
            [
                tensor_var(_np.asarray(X)[ind[border[a]:border[a + 1]]]) /
                        (size[a] - 1)
                    for a in range(int(size.size))
            ] for X in Xs
        ] - _np.asarray(var)
    ) / var_dn

def split_sample (
    Xs,
    size = [0.70, 0.15, 0.15],
    axis = -1,
    n_iter = 100,
    early_stop = 16,
    diff_weights = None,
    random_state = None,
    return_ind = False,
    return_nit = False
):
    """
    Split a sample by preserving its variance.

    Given an iterable `Xs` of samples (see parameter below) aranged in tensors
    all of which are of the same size along the axis `axis` and a 1-dimensional
    array `size` of subsamples' sizes, the samples are split into subsamples
    of given sizes along the provided axis by preserving the original variances
    as closely as possible.  If `size` is an array of weights, such as
    `[0.5, 0.5]`, actual sizes are rounded by resolving sizes from beginning to
    end.  For example, if the samples in `Xs` are of size 13, and `size` is
    `[0.5, 0.5]`, subsamples of sizes 7, 6 are generated.

    Actually, all given `sizes` are treated as positive weights, therefore,
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

    size : (m,) array_like, optional
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

    return_nit : boolean, optional
        If true, number of iterations run is returned as well.

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

    nit : int
        Number of iterations run.

        *Returned only if `return_nit` is true.*

    Raises
    ------
    ValueError
        If `Xs` is empty or the tensors are not of the same size along the axis
        `axis`.

    See Also
    --------
    numpy.random.Random
    tensor_var
    split_sample_optc
    split_sample_optd

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

    n, r = _absolute_subsample_sizes(Xs[0].shape[0], size)

    v = _np.expand_dims([tensor_var(X) / (X.shape[0] - 1) for X in Xs], 1)
    vd = _np.maximum(v, 1.0)

    # Initialise values.
    min_ind = None
    min_Xs = None
    min_d = float('inf')

    # Iteratively find the optimal split into subsamples.

    nit = 0
    j = 0
    for nit in range(n_iter):
        # Check if no improvement has happend in the last `early_stop`
        # iterations.
        if j >= early_stop:
            break

        # Extract and inspect new subsamples.
        ind = random_state.permutation(Xs[0].shape[0])
        d = _np.sum(
            diff_weights * _np.square(
                _samples_variance_difference(Xs, v, vd, n, r, ind)
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

    ret = list()
    if return_ind:
        ret.append(min_ind)
    ret.append(min_Xs)
    if return_nit:
        ret.append(nit)

    return ret[0] if len(ret) == 1 else tuple(ret)
