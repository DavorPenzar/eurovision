#   [ESC](http://eurovision.tv/) Score Predictor

Creation of a machine-learning model for predicting scores in the [*Eurovision Song Contest*](http://eurovision.tv/).

##  Used Values

Function [`utils.compute_params`](utils.py#L33) was not actually used in the production.  Instead, the parameters dictionary was created manually, by setting

```Python
params = {
    'sr': 26215,
    'hop_length': 512,
    'kernel_size': 31,
    'win_length': 512,
    'width': 9
}

```

Somewhere, of course, additional parameters were set, but their values would not be set by the function, anyway.
