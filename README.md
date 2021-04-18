#   [ESC](http://eurovision.tv/) Score Predictor

Creation of a machine-learning model for predicting scores in the [*Eurovision Song Contest*](http://eurovision.tv/).

##  Used Values

Function [`utils.compute_params`](utils.py#L218) was not actually used in the production.  Instead, the parameter dictionary was created manually by setting

```Python
params = {
    'sr': 24576,
    'hop_length': 960,
    'chroma_cqt_hop_length': 3840,
    'tempogram_hop_length': 960,
    'n_fft': 3840,
    'fmin': 0.0,
    'fmax': 11839.82152677230076587824670536366572,
    'chroma_cqt_fmin': 32.70319566257482933473124919041309,
    'frame_length': 1920, # instead of 3840 (half of it)
    'kernel_size': 59,
    'win_length': 256,
    'width': 15
}

```

Somewhere, of course, additional parameters were set, but their values would not be set by the function, anyway.
