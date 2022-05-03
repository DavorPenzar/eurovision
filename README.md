#   [ESC](http://eurovision.tv/) Score Predictor

Creation of a machine-learning model for predicting scores in the [*Eurovision Song Contest*](http://eurovision.tv/).

Unfortunately, no useful model was successfully developed. However, the idea and techniques conducted through the project are extensively reported in the [`Elaborate.pdf`](Elaborate.pdf).

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
