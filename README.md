#   [ESC](http://eurovision.tv/) Score Predictor

Creation of a machine-learning model for predicting scores in the [*Eurovision Song Contest*](http://eurovision.tv/).

##  Used Values

Function [`utils.compute_params`](utils.py#L41) was not actually used in the production.  Instead, the parameters dictionary was created manually by setting

```Python
params = {
    'sr': 19661, # the optimal value is 19660.8
    'hop_length': 768,
    'chromagram_hop_length': 3072,
    'tempogram_hop_length': 768,
    'frame_length': 1536, # instead of 3072 (half of it)
    'kernel_size': 47,
    'win_length': 256,
    'width': 13
}

```

Somewhere, of course, additional parameters were set, but their values would not be set by the function, anyway.
