#   [ESC](http://eurovision.tv/) Score Predictor

Creation of a machine-learning model for predicting scores in the [*Eurovision Song Contest*](http://eurovision.tv/).

##  Elaborate

The elaborate of the project is available at [`Elaborate.pdf`](Elaborate.pdf). It is **not** yet **finished**.

##  To Do:

1.  Update/add inline documentation for [`wpca.py`](wpca.py) and [`delayed_early_stopping.py`](delayed_early_stopping.py).
2.  Finish [`Modelling.ipynb`](Modelling.ipynb) notebook (for construction of model(s)) and add text to it for easier reading.
3.  Add parts about weighted PCA of [*musicnn*](http://github.com/jordipons/musicnn) outputs that was used for custom model inputs to [`Elaborate.pdf`](Elaborate.pdf).
4.  After constructing the model(s), explain it/them in details, and display and comment the results in [`Elaborate.pdf`](Elaborate.pdf).
5.  Remove unnecessary parts from [`Elaborate.pdf`](Elaborate.pdf) and overall finish the elaborate.
6.  Upload data that may and can be uploaded (both leaglly and practically).

##  Used Values

Ideally, the following values would be used when calling [*libROSA*'s](http://librosa.org/) functions for music preprocessing:

```python
import numpy as np

params = {
	'sr': 24576,
	'hop_length': 960,
	'chroma_cqt_hop_length': 3840,
	'tempogram_hop_length': 960,
	'n_fft': 3840,
	'n_mels': 128,
	'n_mfcc': 16,
	'fmin': 0.0,
	'fmax': 11839.82152677230076587824670536366572,
	'chroma_cqt_fmin': 32.70319566257482933473124919041309,
	'frame_length': 1920,
	'kernel_size': 59,
	'win_length': 384,
	'width': 15,
	'dtype': np.float64,
	'stft_dtype': np.complex128
}

```

However, [*musicnn*'s](http://github.com/jordipons/musicnn) and [*libROSA*'s](http://librosa.org/) defaults were used in the end.
