"""All functions and modules related to model definition.
"""

import torch
import sde_lib
import numpy as np

_MODELS = {}


def register_model(cls=None, *, name=None):
    """A decorator for registering model classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _MODELS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _MODELS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_model(name):
    return _MODELS[name]


def get_sigmas(config):
    """Get sigmas --- the set of noise levels for SMLD from config files.
    Args:
      config: A ConfigDict object parsed from the config file
    Returns:
      sigmas: a jax numpy arrary of noise levels
    """
    sigmas = np.exp(
        np.linspace(np.log(config.sde.sigma_max), np.log(config.sde.sigma_min), config.sde.num_scales))

    return sigmas


def create_model(config):
    """Create the score model."""
    model_name = config.model.name
    score_model = get_model(model_name)(config)
    return score_model


def get_model_fn(model, train=False, ):
    """Create a function to give the output of the score-based model.

    Args:
      model: The score model.
      train: `True` for training and `False` for evaluation.

    Returns:
      A model function.
    """

    def model_fn(x, time_cond, class_labels=None):
        """Compute the output of the score-based model.

        Args:
          x: A mini-batch of input data.
          labels: A mini-batch of conditioning variables for time steps. Should be interpreted differently
            for different models.

        Returns:
          A tuple of (model output, new mutable states)
        """
        if train:
            model.train()
        else:
            model.eval()
        
        return model(x, time_cond, class_labels=class_labels)

    return model_fn


def get_score_fn(sde, model, train=False):
    """Wraps `score_fn` so that the model output corresponds to a real time-dependent score function.

    Args:
      sde: An `sde_lib.SDE` object that represents the forward SDE.
      model: A score model.
      train: `True` for training and `False` for evaluation.

    Returns:
      A score function.
    """
    model_fn = get_model_fn(model, train=train)

    def score_fn(x, t, class_labels=None):
        time_cond = sde.marginal_prob(torch.zeros_like(x), t)[1]
        score = model_fn(x, time_cond, class_labels=class_labels)
        return score

    return score_fn


def get_cf_score_fn(sde, model, class_labels, weight):
    """Wraps `score_fn` with weighting.

    Args:
        sde: A `sde_lib.SDE` object
        model: the score model

    Returns:
        A weighted score function. Input of x, t, class_labels, and weight.
    """
    score_fn = get_score_fn(sde, model, train=False)

    def weighted_score_fn(x, t):
        concat_x = x.repeat(2, 1, 1, 1)
        concat_t = t.repeat(2)
        concat_cl = torch.cat([class_labels, torch.zeros_like(class_labels)], dim=0)

        concat_score = score_fn(concat_x, concat_t, concat_cl)
        score_conditioned = concat_score[:x.shape[0]]
        score_clean = concat_score[x.shape[0]:]

        return (1 + weight)[:, None, None, None] * score_conditioned - weight[:, None, None, None] * score_clean

    return weighted_score_fn


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))
