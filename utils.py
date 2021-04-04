# -*- coding: utf-8 -*-

"""
Utilities for preparation of the dataset of songs.

This script is a part of Davor Penzar's *[ESC](http://eurovision.tv/) Score
Predictor* project.

Author: [Davor Penzar `<davor.penzar@gmail.com>`](mailto:davor.penzar@gmail.com)
Date: 2021-04-04
Version: 1.0

"""

# Import standard library.
import math as _math
import numbers as _numbers

# Import SciPy packages.
import numpy as _np
import scipy as _sp
import scipy.optimize as _spo

# Import scikit-image.
import skimage as _ski
import skimage.transform as _skit

# Import TensorLy.
import tensorly as _tl
import tensorly.decomposition as _tld
import tensorly.tenalg as _tla

# Import librosa.
import librosa as _lr


# Define custom functions.


##  MISCELANEOUS FUNCTIONS

def _infinite_iter_singleton (x):
    """
    Infinitely iterate over a singleton.

    The function creates a generator that, when iterated over, returns the
    provided singleton indefinitely.

    Parameters
    ----------
    x
        Singleton to yield.

    Returns
    -------
    iter_x : iterable
        Generator that always returns `x` and never stops; i. e.
        `next(iter_x) is x` is always true and `next(iter_x)` never raises
        `StopIteration`.

    Notes
    -----
    The function is an auxiliary internal function.

    """

    while True:
        yield x

def rank_list (score, centralise = False, normalise = False):
    """
    Compute the rank list according to a score.

    The highest score receives rank 0, the next receives rank 1 and so on.
    If more than one score—let us denote the number `m`—should receive a rank
    `i` (if there is a tie for the rank), the ranks are resolved either by
    setting all ranks to `i` or by setting them all to `i + 0.5 * m`.  The
    behaviour is determined by the parameter `centralise`.  Either way, the
    next rank is `i + m`, and not `i + 1`.

    Parameters
    ----------
    score : (n,) array_like
        Scores to rank.

    centralise : boolean, optional
        If true, ties amongst `m` candidates for rank `i` are resolved by
        setting the ranks to `i + 0.5 * m`; otherwise the ties are resolved by
        setting the ranks to `i`.

    normalise : boolean, optional
        If true, ranks are normalised to set the lowest score to 1.  However,
        if only a single score is observed or if all scores are equal, no
        normalisation is done (all scores receive rank 0 instead).

    Returns
    -------
    rank : (n,) numpy.ndarray
        Ranks of the scores in `score`.  The value `rank[j]` represents the
        rank of `score[j]`.  If all values are integral, the data type
        (`dtype`) of the array is an integral data type; otherwise it is a
        floating point type.  Note that values could only be non-integral if
        parameter `centralise` is true and a tie amongst an odd number of
        candidates must be resolved, or the parameter `normalise` is true and a
        non-normalised score exceeds 1.

    """

    # Prepare parameters.

    score = _np.asarray(score).ravel()
    ind = _np.flip(_np.argsort(score))

    # Compute the ranks.

    rank = list(0 for _ in range(int(score.size)))
    floating = False

    s = float('nan')
    r = 0
    e = set()
    for i in ind:
        if score[i] != s:
            if centralise:
                R = (
                    r +
                    (float(0.5 * len(e)) if (len(e) & 1) else (len(e) >> 1))
                )
                floating = floating or isinstance(R, _numbers.Integral)
                for j in e:
                    rank[j] = R
            s = score[i]
            r += len(e)
            e = set()
        rank[i] = r
        e.add(i)

    rank = _np.array(rank, dtype = _np.float64 if floating else _np.int64)
    if (
        normalise and
        rank.size > 1 and
        _np.max(rank) > 1 and
        not _np.all(_np.isclose(1.0, 1.0 + _np.diff(_np.sort(rank))))
    ):
        rank = _np.true_divide(rank, _np.max(rank))

    # Return the computed ranks.
    return rank

def rank_list_diff (score, normalise = False):
    """
    Compute the differential rank list according to a score.

    The differential rank of the `j`-th score is defined as
    `max(score) - score[j]`.  Hence the highest score receives rank 0 and ties
    amongst scores receive the same rank, as when computing ranks via
    `rank_list` function.  Additionally, the differential rank may be
    normalised so that the lowest score receives rank 1.  The advantage of the
    differential rank list compared to the ordinary rank list is that not only
    do ties receive the same rank, but close scores also receive closer ranks
    than distant scores—in fact, the difference in ranks is proportional to the
    difference in scores.

    Parameters
    ----------
    score : (n,) array_like
        Scores to rank.

    normalise : boolean, optional
        If true, ranks are normalised to set the lowest score to 1.  However,
        if only a single score is observed or if all scores are equal, no
        normalisation is done (all scores receive rank 0 instead).

    Returns
    -------
    rank : (n,) numpy.ndarray
        Differential ranks of the scores in `score`.  The value `rank[j]`
        represents the differential rank of `score[j]`.

    """

    # Prepare parameters.

    score = _np.asarray(score).ravel()

    # Compute and return the differential ranks.

    rank = score.max() - score
    if (
        normalise and
        rank.size > 1 and
        not _np.all(_np.isclose(1.0, 1.0 + _np.diff(_np.sort(rank))))
    ):
        rank = _np.true_divide(rank, _np.max(rank))

    return rank


##  AUDIO MANIPULATION

def compute_params (**kwargs):
    """
    Compute parameters for song preprocessing.

    Parameters `hop_length`, `frame_length`, `n_fft`, `kernel_size`, `width`
    and `win_length` are computed from the parameter `sr` (sample rate).  See
    `librosa.effects.hpss`, `librosa.feature.zero_crossing_rate`,
    `librosa.feature.mfcc`, `librosa.feature.delta`,
    `librosa.feature.chroma_cqt` and `librosa.feature.tempogram` for
    explanation of the parameters.

    Inspect code to see how the parameters are computed.

    Parameters
    ----------
    kwargs
        If `sr` argument is in `kwargs`, then its value is used; otherwise
        `22050` is used.  Other arguments are ignored and perhaps overwritten.

    Returns
    -------
    dict
        Input parameter `kwargs` updated with the computed values.

    See Also
    librosa.effects.hpss
    librosa.feature.zero_crossing_rate
    librosa.feature.chroma_cqt
    librosa.feature.tempogram
    librosa.feature.mfcc
    librosa.feature.delta

    """

    # Get the sample rate.
    sr = kwargs.get('sr', 22050)
    if sr <= 0:
        sr = 1

    # Compute other parameters.

    hop_length = 64 * int(round(sr / 3200.0))
    if hop_length <= 0:
        hop_length = 64

    frame_length = hop_length << 2 # frame_length = 4 * hop_length
    n_fft = hop_length << 2 # n_fft = 4 * hop_length

    kernel_size = int(round(_math.floor(float(hop_length) / 16.0)))
    while not (kernel_size & 1):
        kernel_size -= 1
    while kernel_size < 3 or not (kernel_size & 1):
        # while kernel_size < 3 or not (kernel_size % 2)
        kernel_size += 1

    win_length = int(round(10.0 * sr / hop_length))
    if win_length <= 0:
        win_length = 1

    width = int(round(_math.ceil(float(hop_length) / 64.0)))
    while width < 3 or not (width & 1): # while width < 3 or not (width % 2)
        width += 1

    # Update `kwargs`.
    kwargs.update(
        {
            'sr': sr,
            'hop_length': hop_length,
            'frame_length': frame_length,
            'n_fft': n_fft,
            'kernel_size': kernel_size,
            'win_length': win_length,
            'width': width
        }
    )

    # Return the computed parameters.
    return kwargs

def process_song (
    path,
    return_y = False,
    return_sr = False,
    comp = False,
    **kwargs
):
    """
    Process a song.

    The song's zero-crossing rate, chromagram, tempogram, STFT, MFC, MFC delta
    and MFC delta^2 features are computed and returned.

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
        If true, parameters are computed using `compute_kwargs` function.

    kwargs
        Optional definitions of parameters `sr`, `fmin`, `fmax`, `kernel_size`,
        `hop_length`, `win_length`, `norm`, `n_chroma`, `n_fft`, `n_mels`,
        `n_mfcc`, `bins_per_octave` and `width` for functions
        `librosa.effects.hpss`, `librosa.effects.zero_crossing_rate`,
        `librosa.feature.chroma_cqt`, `librosa.feature.tempogram`,
        `librosa.stft`, `librosa.feature.mfcc` and `librosa.feature.delta`.  If
        any of the parameters is undefined, its default value is used.  If
        `comp` is true, any initial values (except `sr` if it is not `None`)
        are overwritten.

        To specify different values of parameters for each function, the name
        of the parameter must be prepended with the substring
        `'{function_name}_'`, where `function_name` is the name of the function
        without the names of the packages and subpackages.  For instance, to
        set a sepcial value for `hop_length` in `librosa.stft` function, a
        parameter `stft_hop_length'` should be passed.  Such special
        definitions are searched before *global* definitions (definitions
        without prefixes), even if the parameter is used only in one of the
        functions (for instance, `n_chroma` is used solely in
        `librosa.feature.chroma_cqt` but parameter `chroma_cqt_n_chroma` still
        has precedence over it), and they are not overwritten if `comp` is
        true.  However, parameter `sr` is necessarily global and cannot be
        prefixed to use different values in functions.

        Even though some parameters may be passed to `librosa.features.mfcc`
        function if the MFC should be computed from a raw audio signal,
        when using special definitions of parameters, check out if the parameter
        is actually passed to `librosa.stft` function since the MFC is computed
        from a pre-computed STFT. For instance, setting a parameter named
        `mfcc_hop_length` has no effect because `hop_length` is actually used
        by `librosa.stft` function, which is called before
        `librosa.features.mfcc` function.  On the other hand, a globally named
        parameter (e. g. `hop_length`) may be set to avoid such problems.

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

    zcr : (1, n) numpy.ndarray
        Fractions of zero-crossings.  Dimension `n` is the length of the track
        in observed frames of width `hop_length`.

    chromagram : (n_chroma, n) numpy.ndarray
        Chromagram matrix.

    tempogram : (win_length, n) numpy.ndarray
        Tempogram matrix.

    S : (1 + n_fft // 2, n) numpy.ndarray
        STFT matrix.

    mfcc : (n_mfcc, n, 2) numpy.ndarray
        MFC, MFC delta and MFC delta^2 features.  The first slice along the
        third axis is the MFCC matrix, the second slice is the MFCC delta
        matrix and the third slice is the MFCC delta^2 matrix.

    See Also
    --------
    librosa.load
    librosa.effects.hpss
    librosa.effects.zero_crossing_rate
    librosa.feature.chroma_cqt
    librosa.feature.tempogram
    librosa.stft
    librosa.feature.mfcc
    librosa.feature.delta
    compute_kwargs

    """

    # Read raw song.
    y, kwargs['sr'] = _lr.load(
        path = path,
        sr = kwargs.get('sr', 22050),
        mono = True,
        dtype = kwargs.get('load_dtype', kwargs.get('dtype', _np.float32))
    )
    y /= _np.absolute(y).max(axis = None)
    kwargs['sr'] = int(kwargs['sr'])

    # If needed, compute parameters.
    if comp:
        kwargs = compute_params(**kwargs)

    # Compute STFT features.
    S = _lr.stft(
        y,
        n_fft = kwargs.get(
            'stft_n_fft',
            kwargs.get('n_fft', 2048)
        ),
        hop_length = kwargs.get(
            'stft_hop_length',
            kwargs.get('hop_length', None)
        ),
        win_length = kwargs.get(
            'stft_win_length',
            kwargs.get('win_length', None)
        ),
        dtype = kwargs.get(
            'stft_dtype',
            None
        )
    )

    # Separate harmonics and percussives.
    H, P = _lr.decompose.hpss(
        S = S,
        kernel_size = kwargs.get(
            'hpss_kernel_size',
            kwargs.get('kernel_size', 31)
        ),
        power = kwargs.get(
            'hpss_power',
            kwargs.get('power', 2.0)
        )
    )
    y_harmonic = _lr.istft(
        stft_matrix = H,
        hop_length = kwargs.get(
            'stft_hop_length',
            kwargs.get('hop_length', None)
        ),
        win_length = kwargs.get(
            'stft_win_length',
            kwargs.get('win_length', None)
        ),
        dtype = y.dtype,
        length = y.shape[-1]
    )
    y_percussive = _lr.istft(
        stft_matrix = P,
        hop_length = kwargs.get(
            'stft_hop_length',
            kwargs.get('hop_length', None)
        ),
        win_length = kwargs.get(
            'stft_win_length',
            kwargs.get('win_length', None)
        ),
        dtype = y.dtype,
        length = y.shape[-1]
    )
    y_harmonic /= _np.absolute(y_harmonic).max(axis = None)
    y_percussive /= _np.absolute(y_percussive).max(axis = None)
    del H
    del P

    # Compute zero-crossing rate features.
    zcr = _lr.feature.zero_crossing_rate(
        y = y,
        frame_length = kwargs.get(
            'zero_crossing_rate_frame_length',
            kwargs.get('frame_length', 2048)
        ),
        hop_length = kwargs.get(
            'zero_crossing_length_hop_length',
            kwargs.get('hop_length', 512)
        )
    )

    # Compute chroma features.
    chromagram = _lr.feature.chroma_cqt(
        y = y_harmonic,
        sr = kwargs.get('sr', 22050),
        hop_length = kwargs.get(
            'chroma_cqt_hop_length',
            kwargs.get('hop_length', 512)
        ),
        fmin = kwargs.get(
            'chroma_cqt_fmin',
            kwargs.get('fmin', 32.70319566257482933473124919041309)
                # defaul: C1
        ),
        norm = kwargs.get(
            'chroma_cqt_norm',
            kwargs.get('norm', float('inf'))
        ),
        n_chroma = kwargs.get(
            'chroma_cqt_n_chroma',
            kwargs.get('n_chroma', 12)
        ),
        n_octaves = kwargs.get(
            'chroma_cqt_n_octaves',
            kwargs.get('n_octaves', 7)
        ),
        bins_per_octave = kwargs.get(
            'chroma_cqt_bins_per_octave',
            kwargs.get('bins_per_octave', 36)
        )
    )

    # Compute tempo features.
    tempogram = _lr.feature.tempogram(
        y = y_percussive,
        sr = kwargs.get('sr', 22050),
        hop_length = kwargs.get(
            'tempogram_hop_length',
            kwargs.get('hop_length', 512)
        ),
        win_length = kwargs.get(
            'tempogram_win_length',
            kwargs.get('win_length', 384)
        ),
        norm = kwargs.get(
            'tempogram_norm',
            kwargs.get('norm', float('inf'))
        )
    )

    # Compute MFC features and the first- and second-order differences.
    mfcc = _lr.feature.mfcc(
        y = None,
        sr = kwargs.get('sr', 22050),
        S = _lr.power_to_db(
            _lr.feature.melspectrogram(
                y = None,
                sr = kwargs.get('sr', 22050),
                S = (
                    _np.abs(S) **
                    kwargs.get('mfcc_power', kwargs.get('power', 2.0))
                ),
                n_fft = kwargs.get(
                    'stft_n_fft',
                    kwargs.get('n_fft', 2048)
                ),
                hop_length = kwargs.get(
                    'stft_hop_length',
                    kwargs.get('hop_length', None)
                ),
                win_length = kwargs.get(
                    'stft_win_length',
                    kwargs.get('win_length', None)
                ),
                power = kwargs.get(
                    'mfcc_power',
                    kwargs.get('power', 2.0)
                ),
                n_mels = kwargs.get(
                    'mffc_n_mels',
                    kwargs.get('n_mels', 128)
                ),
                fmin = kwargs.get(
                    'mffc_fmin',
                    kwargs.get('fmin', 0.0)
                ),
                fmax = kwargs.get(
                    'mffc_fmax',
                    kwargs.get('fmax', None)
                ),
                dtype = kwargs.get(
                    'mfcc_dtype',
                    kwargs.get('dtype', _np.float32)
                )
            )
        ),
        n_fft = kwargs.get(
            'stft_n_fft',
            kwargs.get('n_fft', 2048)
        ),
        hop_length = kwargs.get(
            'stft_hop_length',
            kwargs.get('hop_length', None)
        ),
        win_length = kwargs.get(
            'stft_win_length',
            kwargs.get('win_length', None)
        ),
        power = kwargs.get(
            'mfcc_power',
            kwargs.get('power', 2.0)
        ),
        n_mfcc = kwargs.get(
            'mfcc_n_mfcc',
            kwargs.get('n_mfcc', 20)
        ),
        n_mels = kwargs.get(
            'mffc_n_mels',
            kwargs.get('n_mels', 128)
        ),
        fmin = kwargs.get(
            'mffc_fmin',
            kwargs.get('fmin', 0.0)
        ),
        fmax = kwargs.get(
            'mffc_fmax',
            kwargs.get('fmax', None)
        ),
        dtype = kwargs.get(
            'mfcc_dtype',
            kwargs.get('dtype', _np.float32)
        )
    )
    mfcc_delta = _lr.feature.delta(
        data = mfcc,
        width = kwargs.get(
            'delta_width',
            kwargs.get('width', 9)
        ),
        order = 1
    )
    mfcc_delta2 = _lr.feature.delta(
        data = mfcc,
        width = kwargs.get(
            'delta_width',
            kwargs.get('width', 9)
        ),
        order = 2
    )
    mfcc = _np.concatenate(
        (
            _np.expand_dims(mfcc, mfcc.ndim),
            _np.expand_dims(mfcc_delta, mfcc_delta.ndim),
            _np.expand_dims(mfcc_delta2, mfcc_delta2.ndim)
        ),
        axis = -1
    )
    del mfcc_delta
    del mfcc_delta2

    # Return computed features.

    ret = list()
    if return_y:
        ret.append(y)
        if return_sr:
            ret.append(kwargs['sr'])
        ret.append(y_harmonic)
        ret.append(y_percussive)
    elif return_sr:
        ret.append(kwargs['sr'])
    ret.append(zcr)
    ret.append(chromagram)
    ret.append(tempogram)
    ret.append(S)
    ret.append(mfcc)

    return tuple(ret)

def _rays_pi_6th (x, y, m, n = None, dtype = _np.float32):
    r"""
    Generate masks for rays between angles of integral multiples of pi/6 in the first quadrant.

    Parameters
    ----------
    x : (m, 1) array_like
        Values along the $ x $-axis strictly greater than 0.

    y : (1, n) array_like
        Values along the $ y $-axis strictly greater than 0.

    m : int
        Transformation downscaling factor (greater than or equal to 1) for the
        $ x $-axis values.  Before computing angles, the input array `x` is
        divided by `m`.

    n : None or int, optional
        Transformation downscaling factor (greater than or equal to 1) for the
        $ y $-axis values.  If `n` is not set (or if it is `None`), it is set
        to the value of `m`.  Before computing angles, the input array `y` is
        divided by `n`.

    dtype : numpy.dtype, optional
        Type of the values in the resulting mask array.

    Returns
    -------
    (m, n, 3) numpy.ndarray
        Mask of the `x` × `y` set.  For each index `i` (0 through 2) the `i`-th
        slice along the third axis (at index 2) represents the mask for the
        ray between angles $ i\pi / 6 $ and $ (i + 1)\pi / 6 $.  Indices along
        the first (at index 0) and second (at index 1) axes are derived from
        input arrays `x` and `y`.

    Notes
    -----
    Reduced flexibility of the function (restrictions on the shapes and values
    of the input parameters) is intentional since the function is not intended
    to be used outside of the library.  The function is, in fact, merely an
    auxiliary internal function.

    """

    # Define the approximation of the square root of 3.
    sqrt_3 = \
        1.7320508075688772935274463415058723669428052538103806280558069795

    # Prepare parameters.

    if n is None:
        n = m

    x = _np.asarray(x)
    y = _np.asarray(y)

    # Compute transformed tangents and cotangents.
    tan = (m * y) / (n * x)
    cot = (n * x) / (m * y) # not `tan ** -1` because of numeric errors

    # Initialise the resulting mask.
    mask = _np.zeros((m, n, 3), dtype = dtype)

    # Compute masks for the cumulative unions of the rays.
    mask[:, :, 0] = (sqrt_3 * tan <= 1.0)
    mask[:, :, 1] = (sqrt_3 * cot >= 1.0)
        # ^ not `tan <= sqrt_3` because of numeric errors
    mask[:, :, 2] = _np.ones((m, n), dtype = dtype)

    # Compute actual masks from cumulative masks.
    for i in range(2, 0, -1):
        mask[:, :, i] -= mask[:, :, i - 1]

    # Return the computed mask.
    return mask

def _circ (x, y, m, n = None, dtype = _np.float32):
    r"""
    Generate masks for a circle (ellipse).

    Parameters
    ----------
    x : (m, 1) array_like
        Values along the $ x $-axis strictly greater than 0.

    y : (1, n) array_like
        Values along the $ y $-axis strictly greater than 0.

    m : int
        Transformation downscaling factor (greater than or equal to 1) for the
        $ x $-axis values.  Before computing distances from the origin, the
        input array `x` is divided by `m`.

    n : None or int, optional
        Transformation downscaling factor (greater than or equal to 1) for the
        $ y $-axis values.  If `n` is not set (or if it is `None`), it is set
        to the value of `m`.  Before computing distances from the origin, the
        input array `y` is divided by `n`.

    dtype : numpy.dtype, optional
        Type of the values in the resulting mask array.

    Returns
    -------
    (m, n) numpy.ndarray
        Mask of the `x` × `y` set for the unit circle.  Indices along the first
        (at index 0) and second (at index 1) axes are derived from input arrays
        `x` and `y`.

    Notes
    -----
    Reduced flexibility of the function (restrictions on the shapes and values
    of the input parameters) is intentional since the function is not intended
    to be used outside of the library.  The function is, in fact, merely an
    auxiliary internal function.

    """

    # Prepare parameters.

    if n is None:
        n = m

    x = _np.asarray(x)
    y = _np.asarray(y)

    # Compute transformed distances from the origin.
    dist = _np.sqrt(_np.square(x / m) + _np.square(y / n))

    # Compute the mask for the circle (ellipse).
    mask = _np.asarray((dist <= 1.0), dtype = dtype)

    # Return the computed mask.
    return mask

def circ_12ths (
    m,
    n = None,
    dtype = _np.float32,
    super_scale = None,
    mem_econ = True
):
    r"""
    Generate masks for twelftths of a circle (ellipse).

    A mask for a measurable set (e. g. a twelfth of a circle) is an array of
    real values from the interval [0, 1].  An actual mask would only have zeros
    and ones as entries; however, since arrays are discrete, a more realistic
    masks can be achieved using anti-aliasing by setting entries partially in
    the set to a value between 0 and 1.  Such a mask indicates the following:

    *   entries with the value of 0 are completely outside of the set,
    *   entries with the value of 1 are completely inside of the set,
    *   entries with a value of $ p $ from the interval (0, 1) represent a
        measurable region with a finite non-zero area such that the ratio of
        the area of its intersection with the set and the complete region's
        area equals $ p $.

    A twelfth of a circle in the context of this function is not *any* subset
    of a circle with the area of a twelfth of the circle's area.  A twelfth of
    a circle is defined by an integer $ k $ and consits of all the points
    $ (x, y) $ for which there exist numbers $ r $ from the interval [0, 1] and
    $ \varphi $ from the interval [$ k\pi / 6 $, $ (k + 1)\pi / 6 $) such that
    $ x = r\cos\varphi $ and $ y = r\sin\varphi $.  Effectively there exist
    only 12 such twelfths since sine and cosine functions are periodic.

    Parameters
    ----------
    m : int
        Number of discretisation points of the radius along the $ x $-axis.
        The complete circle is discretised along the $ x $-axis by `2 * m`
        points.

    n : None or int, optional
        Number of discretisation points of the radius along the $ y $-axis.
        If `n` is not set (or if it is `None`), it is set to the value of `m`.
        The complete circle is discretised along the $ y $-axis by `2 * n`
        points.

    dtype : numpy.dtype, optional
        Type of the values in the resulting mask array.

    super_scale : None or int or tuple of int
        If set (if it is not `None`), the value(s) indicate the factor(s) of
        supersampling for each entry in the resulting mask array to numerically
        calculate their optimal values.  Setting `super_scale` to `None` is the
        same as setting it to 1, and it results in no supersampling, i. e. the
        resulting mask array is not anti-aliased.  Alternatively, the value(s)
        must be strictly positive (greater than or equal to 1), and it may be a
        single integer or a tuple of 2 integers: in the former case the value
        is used for both $ x $- and $ y $-axes supersampling, in the latter
        case the first number is used for $ x $-axis supersampling while the
        second number is used for the $ y $-axis supersampling.

    mem_econ : boolean, optional
        If true, each twelfth in the resulting mask array is position only in
        its representative quadrant, resulting in 4 times fewer entries.
        However, subsequently additional transformations have to be made before
        using the mask(s).  See returning values for more information.

    Returns
    -------
    (..., 12) numpy.ndarray
        Mask of the [-1, 1] × [-1, 1] square.  Each entry represents a
        rectangle acquired by discretising the unit length along the $ x $-axis
        in `m` subsegments and along the $ y $-axis in `n` subsegments.  Hence
        the shape of the resulting mask array is `(m, n, 12)` if `mem_econ` is
        true and `(2 * m, 2 * n, 12)` otherwise (the first axis, at index 0,
        corresponds to the $ x $-axis; the second one, at index 1, corresponds
        to the $ y $-axis).  The higher the index of the entry along the
        $ x $-axis, the higher the $ x $-axis' value of the corresponding point
        (centroid of the rectangle); the same applies to the $ y $-axis.  For
        instance, entry at `[r, s, i]` is to the left of the entry at
        `[r + 1, s, i]`.  Finally, for each index `i` (0 through 11) the `i`-th
        slice along the third axis (at index 2) represents the mask for the
        twelfth of the unit circle defined by the integer `i`.

        If `mem_econ` is false, all slices along the last axis mask the
        complete square [-1, 1] × [-1, 1].  Otherwise each slice masks only the
        square [0, +/-1] × [0, +/-1] to which the corresponding circle twelfth
        belongs.

    """

    # Prepare parameters.

    if n is None:
        n = m

    M = m
    N = n

    if super_scale is not None:
        if hasattr(super_scale, '__iter__'):
            super_scale = list(super_scale)
        else:
            super_scale = [super_scale, super_scale]
        super_scale = tuple(super_scale + [1])

        # Augment dimensions `m` and `n` by the corresponding factors.
        M = super_scale[0] * m
        N = super_scale[1] * n

    # Discretise the rectangle [0, M] × [0, N].
    x, y = _np.ogrid[
        0.5 : M - 0.5 : complex(imag = M),
        0.5 : N - 0.5 : complex(imag = N)
    ]

    # Compute masks of the pi/6-rays and the circle (ellipse).
    mask_rays = _rays_pi_6th(x, y, M, N, dtype)
    mask_circ = _circ(x, y, M, N, dtype)

    # In case of supersampling, downscale the computed masks.
    if super_scale is not None:
        aux_mask = _skit.downscale_local_mean(
            _np.concatenate(
                (mask_rays, _np.expand_dims(mask_circ, mask_circ.ndim)),
                axis = -1
            ),
            factors = super_scale
        ).astype(dtype)
        mask_rays = aux_mask[:, :, :-1]
        mask_circ = aux_mask[:, :, -1]

        # Free memory.
        del aux_mask

    # Initialise the final mask and compute masks for the first 3 twelfths.
    mask = None
    if mem_econ:
        mask = _np.zeros((m, n, 12), dtype = dtype)
        for i in range(3):
            mask[:, :, i] = mask_rays[:, :, i] * mask_circ
    else:
        mask = _np.zeros(((m << 1), (n << 1), 12), dtype = dtype)
        for i in range(3):
            mask[m:, n:, i] = mask_rays[:, :, i] * mask_circ

    # Free memory.
    del mask_rays
    del mask_circ

    # Compute the masks of the remaining twelfths by flipping and rotating the
    # first 3 twelfths.
    for q in range(1, 4):
        for i in range(3):
            mask[:, :, 3 * q + i] = _np.flip(
                mask[:, :, 3 * q - i - 1],
                axis = 1 - (q & 1)
            )

    # Return the computed mask.
    return mask

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
    r"""
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

    return _np.sum(
        _np.square(Y - _np.mean(Y, axis = axis, keepdims = True)),
        axis = None
    )

def tensor_var_vec (Y, axis = -1, **kwargs):
    r"""
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
    r"""
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

def tensor_cov (Xs, axis = -1):
    r"""
    Compute covariance of a sample arranged in tensors.

    The covariance of tensor-shaped samples
    $ X_{0} , X_{1} , \dotsc , X_{m - 1} $ is defined as the sum of all
    elements in the inner product of tensors $ X_{i} - \bar{X_{i}} $ for
    indices $ i = 0 , 1 , \dotsc , m - 1 $ (the mean of all observations in a
    single tensor is subtracted from each observation, and then the transformed
    observations are joined in the transformed sample in the same way as the
    original observations were joined into $ X_{i} $; the mean is computed
    pointwise, i. e. the "mean observation" is of the same shape as all
    original observations, but each of its elements is the mean of the elements
    at the corresponding positions in the original observations) divided by the
    appropriate denominator.  The denominator is $ n - df $, where $ n $ is the
    number of observations and $ df $ are the degrees of freedom.

    Parameters
    ----------
    Xs : iterable of (m1, m2, ..., n, ..., mk) array
        Original samples in the form of tensors, all of which are of the same
        sshape.  Let us denote the size of the tensors along the axis `axis` as
        `n`.  Then each of the tensors' `n` slices along the axis represent a
        single observation.

        If `Xs` contains only a single element, its variance computed via
        `tensor_var` function is returned.

    axis : int or iterable of int, optional
        Axis along which the observations are joined into the tensor `Y`. If
        `axis` is negative, it counts from the last to the first axis, -1 being
        the last.

        If a single value is passed, the value is used for all tensors in `Xs`.
        Otherwise the iterable must contain exactly the same number of elements
        in as `Xs` so that the `j`-th axis can be used for the `j`-th tensor.

    Returns
    -------
    float
        Sum of elements in the inner product of $ X_{i} - \bar{X_{i}} $ for
        $ X_{i} $ in `Xs`.  This value can be considered as the total cvariance
        of the samples, multiplied by `n` minus the degrees of freedom.

    Raises
    ------
    ValueError
        If `Xs` is empty or the tensors are not of the same size along the axis
        `axis` or the number of provided axes in `axis` does not match the
        number of tensors in `Xs`.

    See also
    --------
    tensor_var

    """

    # Prepare parameters.

    single_axis = False
    if _np.isscalar(axis):
        axis = _infinite_iter_singleton(axis)
        single_axis = True
    else:
        axis = tuple(axis)
    Xs = tuple(
        _np.moveaxis(X, ax, 0) for X, ax in zip(Xs, axis)
    )
    if not (
        len(Xs) and
        all(X.shape == Xs[0] for X in Xs) and
        (single_axis or len(axis) == len(Xs))
    ):
        raise ValueError('Either no tensor is provided or they are not of the same shape or the number of axes provided does not match the number of tensors.')

    # Compute and return the covariance.

    if len(Xs) == 1:
        return tensor_var(Xs[0], axis = 0)
    
    return _np.sum(
        _np.prod(
            list(
                X - _np.mean(X, axis = ax, keepdims = True)
                    for X, ax in zip(Xs, axis)
            ),
            axis = 0
        ),
        axis = 0
    )


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
        truncated into range [0, `m`) (actually, range [0, `m` - 1] union {0}).

    Notes
    -----
    Reduced flexibility of the function (restrictions on the indexing axis, the
    sign and the dimensionality of the index) is intentional since the function
    is not intended to be used outside of the library.  The function is, in
    fact, merely an auxiliary internal function.

    """

    return _np.maximum(
        _np.minimum(_np.asarray(ind).ravel(), _np.asarray(a).shape[0] - 1),
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
    outside of the library.  The function is, in fact, merely an auxiliary
    internal function.

    See Also
    --------
    numpy.round

    """

    return _np.round(ind).ravel().astype(_np.int32)

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
    intended to be used outside of the library.  The function is, in fact,
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
    `n` is the last index and `q == 0` is also handled corectly).

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
    intended to be used outside of the library.  The function is, in fact,
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
    floor_ind = _np.floor(ind).astype(_np.int32)
    ceil_ind = _np.ceil(ind).astype(_np.int32)
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
    intended to be used outside of the library.  The function is, in fact,
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
    intended to be used outside of the library.  The function is, in fact,
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

    random_state : None or numpy.random.Generator, optional
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
         random_state = _np.random.default_rng()

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

        # Free memory.
        del ind
        del var
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

    random_state : None or numpy.random.Generator, optional
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
         random_state = _np.random.default_rng()

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

    random_state : None or numpy.random.Generator, optional
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
         random_state = _np.random.default_rng()

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
    intended to be used outside of the library.  The function is, in fact,
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
    n = _np.array(n, dtype = _np.int32)
    r = _np.insert(n.cumsum(), 0, 0)

    # Return computed arrays.
    return (n, r)

def _samples_variance_normalisation (Xs):
    """
    Compute variances and their normalisation denominators for tensors samples.

    Given an iterable `Xs` of samples (see parameter below) aranged in tensors
    all of which are of the same size along the first axis, variances along
    their first axes are returned, as well as denominators *safe* for variance
    normalisation.  The denominator is computed as the maximum between a normal
    variance and the number 1 (neutral denominator).

    Parameters
    ----------
    Xs : iterable of (n, ...) array_like
        Original samples in the form of tensors, all of which are of the same
        size along the first axis.  Let us denote the number of tensors in `Xs`
        as `m` and the size of the tensors along the first `axis` as `n`.  Then
        each of the tensors' `n` slices along the axis represent a single
        observation.

    Returns
    -------
    var : (m,) numpy.ndarray
        1-dimensional array of true variances of samples in `Xs`.  The
        variances are computed using `tensor_var` function, but the result is
        then divided by `n` - 1 to obtain the true sample variance.

    var_dn : (m,) numpy.ndarray
        1-dimensional denominator *safe* for variance normalisation.  This
        array is computed as `numpy.maximum(var, 1)`.

    Reduced flexibility of the function (restrictions on the sampling axis) is
    intentional since the function is not intended to be used outside of the
    library.  The function is, in fact, merely an auxiliary internal function.

    See Also
    --------
    numpy.maximum
    tensor_var

    """

    var = _np.asarray([tensor_var(X) / (X.shape[0] - 1) for X in Xs])
    var_dn = _np.maximum(var, 1)

    return (var, var_dn)

def _samples_variance_difference (Xs, var, var_dn, size, border, ind):
    """
    Compute differences in variances between subsamples and the original (total) tensors sample along the first axis.

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
        1-dimensional array of real-valued indices to extract subsamples.
        This parameter should actually be a permutation of `list(range(m))`
        (all of the legal indices for tensors along their first axis with no
        duplicates), or at least when rounded to integral values via
        `_round_indices` function.

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
    `_interpolating_indexed_array(Xs[j], ind[border[i]:border[i + 1]])`.  That
    is, the index `ind` is divided into parts of sizes provided through the
    parameter `size`, and its `i`-th part is used as indices for the `i`-th
    subsamples as in `_interpolating_indexed_array` function.  To extract the
    `i`-th subsample of the `j`-th tensor, this part of indices is then applied
    to the `j`-th tensor.

    The normalised variance of a subsample is computed by subtracting the
    original sample's variance from the subsample's variance, and dividing the
    difference by `max(var[j], 1)` (`var[j]` is the variance of the `j`-th
    tensor).  The variances are true variances, menaning the result of
    `tensor_var` function is divided by the appropriate tensor's (original or
    the subsample) length.

    Reduced flexibility of the function (restrictions on the indexing axis) is
    intentional since the function is not intended to be used outside of the
    library.  The function is, in fact, merely an auxiliary internal function.

    See Also
    --------
    numpy.maximum
    numpy.sum
    numpy.diff
    numpy.array_equal
    tensor_var
    _round_indices
    _interpolating_indexed_array

    """

    # Prepare parameters.
    size = _np.asarray(size).ravel()
    border = _np.asarray(border).ravel()
    ind = _np.asarray(ind).ravel()

    # Compute and return normalised variances.
    return (
        [
            [
                tensor_var(
                    _interpolating_indexed_array(
                        X,
                        ind[border[a]:border[a + 1]]
                    )
                ) / (size[a] - 1)
                    for a in range(size.size)
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
        size of the tensors along the axis `axis` as `n`.  Then each of the
        tensors' `n` slices along the axis represent a single observation.

        Passing more than one sample set is enabled to allow splitting a
        dataset according to both inputs and outputs (for example, a dataset
        of `n` inputs of shape `(256, 256)` as matrices and `n` 1-dimensional
        outputs as labels).  However, this is generalised to enable passing an
        arbitrary number of parts of observations in a sample.

    size : (m,) array_like, optional
        Weighted sizes of the resulting subsamples.  The resulting subsamples'
        sizes are computed by rounding the values of
        `size / numpy.sum(size) * n` so that the sum equals to `n`.

    axis : int or iterable of int, optional
        Axis along which the observations are joined into the tensors in `Xs`.
        If `axis` is negative, it counts from the last to the first axis, -1
        being the last.

        If a single value is passed, the value is used for all tensors in `Xs`.
        Otherwise the iterable must contain exactly the same number of elements
        in as `Xs` so that the `j`-th axis can be used for the `j`-th tensor.

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

    random_state : None or numpy.random.Generator, optional
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
        `axis` or the number of provided axes in `axis` does not match the
        number of tensors in `Xs`.

    See Also
    --------
    numpy.random.Random
    tensor_var
    split_sample_optd
    split_sample_optc

    """

    # Prepare parameters.

    single_axis = False
    if _np.isscalar(axis):
        axis = _infinite_iter_singleton(axis)
        single_axis = True
    else:
        axis = tuple(axis)
    Xs = tuple(
        _np.moveaxis(X, ax, 0) for X, ax in zip(Xs, axis)
    )
    if not (
        len(Xs) and
        all(X.shape[ax] == Xs[0].shape[axis[0]] for X, ax in zip(Xs, axis)) and
        (single_axis or len(axis) == len(Xs))
    ):
        raise ValueError('Either no tensor is provided or they are not of the same shape along the provided axes or the number of axes provided does not match the number of tensors.')

    if early_stop is None:
        early_stop = n_iter
    if diff_weights is None:
        diff_weights = 1
    if random_state is None:
         random_state = _np.random.default_rng()

    n, r = _absolute_subsample_sizes(Xs[0].shape[0], size)
    del size

    v, vd = _samples_variance_normalisation(Xs)
    v = _np.expand_dims(v, 1)
    vd = _np.expand_dims(vd, 1)

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

        # Free memory.
        del ind
        del d
    min_ind = tuple(
        _np.sort(min_ind[r[a]:r[a + 1]]) for a in range(n.size)
    )
    min_Xs = tuple(
        tuple(_np.moveaxis(X[I], 0, axis) for I in min_ind)
            for X, ax in zip(Xs, axis)
    )

    # Return computed values.

    ret = list()
    if return_ind:
        ret.append(min_ind)
    ret.append(min_Xs)
    if return_nit:
        ret.append(nit)

    return ret[0] if len(ret) == 1 else tuple(ret)

def split_sample_optd (
    Xs,
    size = [0.70, 0.15, 0.15],
    axis = -1,
    diff_weights = None,
    random_state = None,
    return_ind = False,
    return_raw = False,
    constraint_kwargs = dict(),
    minimize_kwargs = dict()
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

    Unlike `split_sample` function, this function does not produce many
    random subsamples in order to find the optimal split; rather, it uses
    `scipy.optimize.minize` function for the optimisation.  However, unlike
    `split_sample_optc` function, at each step it rounds the indices to nearest
    integers and then checks the variance.  The optimisation-based approach may
    be suboptimal since the underlying domain (choice of indices) is descrete.

    Parameters
    ----------
    Xs : iterable of array_like
        Original samples in the form of tensors, all of which are of the same
        size along the axis `axis` (see parameter below).  Let us denote the
        size of the tensors along the axis `axis` as `n`.  Then each of the
        tensors' `n` slices along the axis represent a single observation.

        Passing more than one sample set is enabled to allow splitting a
        dataset according to both inputs and outputs (for example, a dataset
        of `n` inputs of shape `(256, 256)` as matrices and `n` 1-dimensional
        outputs as labels).  However, this is generalised to enable passing an
        arbitrary number of parts of observations in a sample.

    size : (m,) array_like, optional
        Weighted sizes of the resulting subsamples.  The resulting subsamples'
        sizes are computed by rounding the values of
        `size / numpy.sum(size) * n` so that the sum equals to `n`.

    axis : int or iterable of int, optional
        Axis along which the observations are joined into the tensors in `Xs`.
        If `axis` is negative, it counts from the last to the first axis, -1
        being the last.

        If a single value is passed, the value is used for all tensors in `Xs`.
        Otherwise the iterable must contain exactly the same number of elements
        in as `Xs` so that the `j`-th axis can be used for the `j`-th tensor.

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

    random_state : None or numpy.random.Generator, optional
        Random state of the algorithm (for reproducibility of results).

        **Note.** The random state is used only to generate the initial guess
        for `scipy.optimize.minimize` function.  For true reproducibility check
        arguments `constraint_kwargs` and `optimize_kwargs`.

    return_ind : boolean, optional
        If true, indices of the subsample (in the original sample) are returned
        as well.

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
    subsamples : tuple of tuple of numpy.ndarray
        Tuple of tuples of the generated subsamples.  The `j`-th tensor in the
        `i`-th tuple is a subsample of the `j`-th size extracted from the
        `i`-th original tensor (sample part), represented as a `numpy.ndarray`.

    ind : tuple of (m,) numpy.ndarray
        Indices of the returned subsamples in the original samples.  The `j`-th
        array represents the indices of the subsamples of the `j`-th size.

        *Returned only if `return_ind` is true.*

    raw : scipy.optimize.OptimizeResult
        The result of `scipy.optimize.minimize` function call.

        *Returned only if `return_raw` is true.*

    Raises
    ------
    ValueError
        If `Xs` is empty or the tensors are not of the same size along the axis
        `axis` or the number of provided axes in `axis` does not match the
        number of tensors in `Xs`.

    Notes
    -----
    The result of the optimisation (returned value `raw`) is converted to an
    actual index-array by rounding values of `raw.x` to integers using
    `numpy.round` function and truncating them to fit into range
    [0, `Xs[0].shape(axis)`).  The obtained array is then split into parts of
    sizes of the output subsamples (computed from the input parameter `size`);
    in the end, indices of each subsample (each part of the indices array) are
    sorted.  This is done regardless of the value of `raw.success`, meaning the
    indices may not be in fact optimised.  As a result, it is not guaranteed
    that all indices are mutually different or that all observations appear in
    the output subsamples and the user is advised to check the returned values
    themselves.

    See Also
    --------
    numpy.random.Random
    numpy.round
    tensor_var
    split_sample
    split_sample_optd

    """

    # Prepare parameters.

    single_axis = False
    if _np.isscalar(axis):
        axis = _infinite_iter_singleton(axis)
        single_axis = True
    else:
        axis = tuple(axis)
    Xs = tuple(
        _np.moveaxis(X, ax, 0) for X, ax in zip(Xs, axis)
    )
    if not (
        len(Xs) and
        all(X.shape[ax] == Xs[0].shape[axis[0]] for X, ax in zip(Xs, axis)) and
        (single_axis or len(axis) == len(Xs))
    ):
        raise ValueError('Either no tensor is provided or they are not of the same shape along the provided axes or the number of axes provided does not match the number of tensors.')

    if diff_weights is None:
        diff_weights = 1
    if random_state is None:
         random_state = _np.random.default_rng()

    n, r = _absolute_subsample_sizes(Xs[0].shape[0], size)
    del size

    v, vd = _samples_variance_normalisation(Xs)
    v = _np.expand_dims(v, 1)
    vd = _np.expand_dims(vd, 1)

    # Find the optimal split into subsamples.
    res = _spo.minimize(
        lambda ind: _np.sum(
            diff_weights * _np.square(
                _samples_variance_difference(
                    Xs,
                    v,
                    vd,
                    n,
                    r,
                    _round_indices(_truncate_indices(Xs[0], ind))
                )
            ),
            axis = None
        ),
        random_state.permutation(Xs[0].shape[0]),
        tuple(),
        bounds = _index_optimisation_bounds(Xs[0]),
        constraints = _index_optimisation_constraint(**constraint_kwargs),
        **minimize_kwargs
    )
    min_ind = _round_indices(_truncate_indices(Xs[0], res.x))
    min_ind = tuple(
        _np.sort(min_ind[r[a]:r[a + 1]]) for a in range(n.size)
    )
    min_Xs = tuple(
        tuple(_np.moveaxis(X[I], 0, ax) for I in min_ind)
            for X, ax in zip(Xs, axis)
    )

    # Return computed values.

    ret = list()
    if return_ind:
        ret.append(min_ind)
    ret.append(min_Xs)
    if return_raw:
        ret.append(res)

    return ret[0] if len(ret) == 1 else tuple(ret)

def split_sample_optc (
    Xs,
    size = [0.70, 0.15, 0.15],
    axis = -1,
    diff_weights = None,
    random_state = None,
    return_ind = False,
    return_raw = False,
    constraint_kwargs = dict(),
    minimize_kwargs = dict()
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

    Unlike `split_sample` function, this function does not produce many
    random subsamples in order to find the optimal split; rather, it uses
    `scipy.optimize.minize` function for the optimisation.  However, unlike
    `split_sample_optd` function, the indices are not rounded until the end
    of the optimisation but subsamples are generated from non-integral indices
    through interpolation of the values from the original tensors using a
    linear spline.  This may be valid approach if the original tensors
    represent a relatively dense discretisation of smooth functions
    (progressions of values), but the optimisation-based approach may still be
    suboptimal since the underlying domain (choice of indices) is descrete.

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

    axis : int or iterable of int, optional
        Axis along which the observations are joined into the tensors in `Xs`.
        If `axis` is negative, it counts from the last to the first axis, -1
        being the last.

        If a single value is passed, the value is used for all tensors in `Xs`.
        Otherwise the iterable must contain exactly the same number of elements
        in as `Xs` so that the `j`-th axis can be used for the `j`-th tensor.

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

    random_state : None or numpy.random.Generator, optional
        Random state of the algorithm (for reproducibility of results).

        **Note.** The random state is used only to generate the initial guess
        for `scipy.optimize.minimize` function.  For true reproducibility check
        arguments `constraint_kwargs` and `optimize_kwargs`.

    return_ind : boolean, optional
        If true, indices of the subsample (in the original sample) are returned
        as well.

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
    subsamples : tuple of tuple of numpy.ndarray
        Tuple of tuples of the generated subsamples.  The `j`-th tensor in the
        `i`-th tuple is a subsample of the `j`-th size extracted from the
        `i`-th original tensor (sample part), represented as a `numpy.ndarray`.

    ind : tuple of (m,) numpy.ndarray
        Indices of the returned subsamples in the original samples.  The `j`-th
        array represents the indices of the subsamples of the `j`-th size.

        *Returned only if `return_ind` is true.*

    raw : scipy.optimize.OptimizeResult
        The result of `scipy.optimize.minimize` function call.

        *Returned only if `return_raw` is true.*

    Raises
    ------
    ValueError
        If `Xs` is empty or the tensors are not of the same size along the axis
        `axis` or the number of provided axes in `axis` does not match the
        number of tensors in `Xs`.

    Notes
    -----
    The result of the optimisation (returned value `raw`) is converted to an
    actual index-array by rounding values of `raw.x` to integers using
    `numpy.round` function and truncating them to fit into range
    [0, `Xs[0].shape(axis)`).  The obtained array is then split into parts of
    sizes of the output subsamples (computed from the input parameter `size`);
    in the end, indices of each subsample (each part of the indices array) are
    sorted.  This is done regardless of the value of `raw.success`, meaning the
    indices may not be in fact optimised.  As a result, it is not guaranteed
    that all indices are mutually different or that all observations appear in
    the output subsamples and the user is advised to check the returned values
    themselves.

    See Also
    --------
    numpy.random.Random
    numpy.round
    tensor_var
    split_sample
    split_sample_optd

    """

    # Prepare parameters.

    single_axis = False
    if _np.isscalar(axis):
        axis = _infinite_iter_singleton(axis)
        single_axis = True
    else:
        axis = tuple(axis)
    Xs = tuple(
        _np.moveaxis(X, ax, 0) for X, ax in zip(Xs, axis)
    )
    if not (
        len(Xs) and
        all(X.shape[ax] == Xs[0].shape[axis[0]] for X, ax in zip(Xs, axis)) and
        (single_axis or len(axis) == len(Xs))
    ):
        raise ValueError('Either no tensor is provided or they are not of the same shape along the provided axes or the number of axes provided does not match the number of tensors.')

    if diff_weights is None:
        diff_weights = 1
    if random_state is None:
         random_state = _np.random.default_rng()

    n, r = _absolute_subsample_sizes(Xs[0].shape[0], size)
    del size

    v, vd = _samples_variance_normalisation(Xs)
    v = _np.expand_dims(v, 1)
    vd = _np.expand_dims(vd, 1)

    # Find the optimal split into subsamples.
    res = _spo.minimize(
        lambda ind: _np.sum(
            diff_weights * _np.square(
                _samples_variance_difference(
                    Xs,
                    v,
                    vd,
                    n,
                    r,
                    _truncate_indices(Xs[0], ind)
                )
            ),
            axis = None
        ),
        random_state.permutation(Xs[0].shape[0]),
        tuple(),
        bounds = _index_optimisation_bounds(Xs[0]),
        constraints = _index_optimisation_constraint(**constraint_kwargs),
        **minimize_kwargs
    )
    min_ind = _round_indices(_truncate_indices(Xs[0], res.x))
    min_ind = tuple(
        _np.sort(min_ind[r[a]:r[a + 1]]) for a in range(n.size)
    )
    min_Xs = tuple(
        tuple(_np.moveaxis(X[I], 0, ax) for I, ax in min_ind)
            for X, ax in zip(Xs, axis)
    )

    # Return computed values.

    ret = list()
    if return_ind:
        ret.append(min_ind)
    ret.append(min_Xs)
    if return_raw:
        ret.append(res)

    return ret[0] if len(ret) == 1 else tuple(ret)
