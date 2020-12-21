#   [ESC](http://eurovision.tv/) Score Predictor

Creation of a machine-learning model for predicting scores in the [*Eurovision Song Contest*](http://eurovision.tv/).

##  Used Values

Function [`compute_params`](utils.py#L33) was not actually used in the production.  Instead, the parameters dictionary was created manually, as setting

```Python
params = {
    'sr': 26215,
    'hop_length': 512,
    'kernel_size': 31,
    'win_length': 512,
    'width': 9,
    'dtype': np.float64
}

```
