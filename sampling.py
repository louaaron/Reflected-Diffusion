"""Various sampling methods."""

import torch
import torch.nn as nn
import numpy as np
import abc

from models.utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
from models import utils as mutils
import cube

_CORRECTORS = {}
_PREDICTORS = {}
_DENOISERS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)

def register_denoiser(cls=None, *, name=None):
    """A decorator for registering corrector classes."""
    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _DENOISERS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _DENOISERS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]

def get_denoiser(name):
    return _DENOISERS[name]


def get_sampling_fn(config, sde, shape, eps, device):
    """Create a sampling function.

    Args:
      config: A `ml_collections.ConfigDict` object that contains all configuration information.
      sde: A `sde_lib.SDE` object that represents the forward SDE.
      shape: A sequence of integers representing the expected shape of a single sample.
      inverse_scaler: The inverse data normalizer function.
      eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
      A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """

    sampler_name = config.sampling.method
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == 'ode':
        denoiser = get_denoiser(config.sampling.denoiser.lower())
        sampling_fn = get_ode_sampler(sde=sde,
                                      shape=shape,
                                      eps=eps,
                                      moll=config.sampling.moll,
                                      side_eps=config.sampling.side_eps,
                                      device=device)

    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == 'pc':
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        denoiser = get_denoiser(config.sampling.denoiser.lower())
        sampling_fn = get_pc_sampler(sde=sde,
                                     shape=shape,
                                     predictor=predictor,
                                     corrector=corrector,
                                     denoiser=denoiser,
                                     snr=config.sampling.snr,
                                     n_steps=config.sampling.n_steps_each,
                                     eps=eps,
                                     device=device)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(score_fn, probability_flow)
        self.score_fn = score_fn

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the predictor.

        Args:
          x: A PyTorch tensor representing the current state
          t: A Pytorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__()
        self.sde = sde
        self.score_fn = score_fn
        self.snr = snr
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t):
        """One update of the corrector.

        Args:
          x: A PyTorch tensor representing the current state
          t: A PyTorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass

class Denoiser(abc.ABC):
    """The abstract class for a denoiser"""
    def __init__(self, denoiser):
        super().__init__()
        self.denoiser = denoiser

    @abc.abstractmethod
    def update_fn(self, x, x_mean, t):
        pass


@register_predictor(name='euler_maruyama')
class ReflectedEulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, score_fn, probability_flow=False):
        super().__init__(sde, score_fn, probability_flow)

    def update_fn(self, x, t):
        dt = -1. / self.rsde.N
        z = torch.randn_like(x)
        drift, diffusion = self.rsde.sde(x, t)
        x_mean = x + drift * dt
        x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z

        x, x_mean = cube.reflect(x), cube.reflect(x_mean)

        return x, x_mean


@register_corrector(name='langevin')
class ReflectedLangevinCorrector(Corrector):
    def __init__(self, sde, score_fn, snr, n_steps):
        super().__init__(sde, score_fn, snr, n_steps)

    def update_fn(self, x, t):
        sde = self.sde
        score_fn = self.score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        alpha = torch.ones_like(t)

        for i in range(n_steps):
            grad = score_fn(x, t)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

            x, x_mean = cube.reflect(x), cube.reflect(x_mean)

        return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def update_fn(self, x, t):
        return x, x


@register_denoiser(name='network')
class TrainedDenoiser(Denoiser):
    """Apply network to denoise input"""
    def update_fn(self, x, x_mean, t):
        return (x - self.denoiser(x, t)).clamp(min=0, max=1)


@register_denoiser(name="mean")
class MeanDenoiser(Denoiser):
    def update_fn(self, x, x_mean, t):
        return x_mean


@register_denoiser(name="none")
class NoneDenoiser(Denoiser):
    def update_fn(self, x, x_mean, t):
        return x


def shared_predictor_update_fn(x, t, sde, model, predictor, probability_flow):
    """A wrapper that configures and returns the update function of predictors."""
    score_fn = mutils.get_score_fn(sde, model, train=False)
    if predictor is None:
        predictor_obj = NonePredictor(sde, score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, score_fn, probability_flow)
    return predictor_obj.update_fn(x, t)


def shared_corrector_update_fn(x, t, sde, model, corrector, snr, n_steps):
    """A wrapper tha configures and returns the update function of correctors."""
    score_fn = mutils.get_score_fn(sde, model, train=False)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, score_fn, snr, n_steps)
    else:
        corrector_obj = corrector(sde, score_fn, snr, n_steps)
    return corrector_obj.update_fn(x, t)


def shared_denoiser_update_fn(x, x_mean, t, denoiser, denoise_model):
    if denoiser is None:
        denoiser_obj = NoneDenoiser(denoise_model)
    else:
        denoiser_obj = denoiser(denoise_model)
    return denoise_obj.denoise(x, x_mean, t)

    
def get_pc_sampler(sde, shape, predictor, corrector, denoiser, snr, 
                   n_steps=1, eps=1e-3, device='cuda'):
    """Create a Predictor-Corrector (PC) sampler."""
    def pc_sampler(model, z=None, noise_removal_model=None, weight=0, class_labels=None):
        """ The PC sampler funciton.

        Args:
          model: A score model.
          noise_removal_model: A noise removal model (if used).
          weight: Weight used for CF guidance.
          class_labels: Class labels used for CF guidance.
        Returns:
          Samples, number of function evaluations.
        """
        # Initial sample
        if z is None:
            x = torch.rand(shape).to(device)
        else:
            x = z

        if class_labels is None:
            score_fn = mutils.get_score_fn(sde, model, train=False)
        else:
            score_fn = mutils.get_cf_score_fn(sde, model, class_labels, weight)

        # Create update functions
        pred = predictor(sde, score_fn)
        corr = corrector(sde, score_fn, snr, n_steps)
        deno = denoiser(noise_removal_model)

        with torch.no_grad():
            # Initial sample
            x = torch.rand(shape).to(device)
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            for i in range(sde.N):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                if i < sde.N - 1:
                    x, _ = corr.update_fn(x, vec_t)
                    x, x_mean = pred.update_fn(x, vec_t)

            vec_t = torch.ones(shape[0], device=t.device) * eps
            deno.update_fn(x, x_mean, vec_t)

            return x, sde.N * (n_steps + 1)

    return pc_sampler


def get_ode_sampler(sde, shape, rtol=1e-5, atol=1e-5, method='RK45', eps=1e-3, moll=200, side_eps=1e-2, device='cuda'):
    """Probability flow ODE sampler with the black-box ODE solver."""

    def drift_fn(score_fn, x, t):
        """Get the drift function of the reverse-time SDE."""
        rsde = sde.reverse(score_fn, probability_flow=True)
        return rsde.sde(x, t)[0]

    def ode_sampler(model, z=None, noise_removal_model=None, weight=0, class_labels=None):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
          model: A score model.
          z: If present, generate samples from latent code `z`.
        Returns:
          samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            if z is None:
                x = (1 - 2 * side_eps) * torch.rand(shape).to(device) + side_eps
            else:
                x = z

            if class_labels is None:
                score_fn = mutils.get_score_fn(sde, model, train=False)
            else:
                score_fn = mutils.get_cf_score_fn(sde, model, class_labels, weight)

            def bump(x):
                if moll > 0:
                    return ((- 1/ (0.5 ** 2 - (0.5 - x).pow(2)) + 4) / moll).exp()
                else:
                    return x

            def ode_func(t, x):
                x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
                vec_t = torch.ones(shape[0], device=x.device) * t
                drift = drift_fn(score_fn, x, vec_t) * bump(x)
                return to_flattened_numpy(drift)

            solution = integrate.solve_ivp(ode_func, (sde.T, eps), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

            vec_t = torch.ones(shape[0], device=x.device) * eps

            return x, nfe

    return ode_sampler
