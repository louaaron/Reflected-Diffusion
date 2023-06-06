"""All functions related to loss computation and optimization.
"""

import torch
import torch.optim as optim
import numpy as np
from models import utils as mutils

import cube


def get_optimizer(config, params):
    if config.optim.optimizer == 'Adam':
        optimizer = optim.Adam(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    elif config.optim.optimizer == 'AdamW':
        optimizer = optim.AdamW(params, lr=config.optim.lr, betas=(config.optim.beta1, config.optim.beta2), eps=config.optim.eps,
                               weight_decay=config.optim.weight_decay)
    else:
        raise NotImplementedError(
            f'Optimizer {config.optim.optimizer} not supported yet!')

    return optimizer


def optimization_manager(config):
    """Returns an optimize_fn based on `config`."""

    def optimize_fn(optimizer, params, step, lr=config.optim.lr,
                    warmup=config.optim.warmup,
                    grad_clip=config.optim.grad_clip,
                    scaler=None):
        """Optimizes with warmup and gradient clipping (disabled if negative)."""
        if scaler is not None:
            scaler.unscale_(optimizer)

        if warmup > 0:
            for g in optimizer.param_groups:
                g['lr'] = lr * np.minimum(step / warmup, 1.0)
        if grad_clip >= 0:
            torch.nn.utils.clip_grad_norm_(params, max_norm=grad_clip)

        if scaler is None:
            optimizer.step()
        else:
            scaler.step(optimizer)
            scaler.update()

    return optimize_fn


def get_sde_loss_fn(sde, train, reduce_mean=True, likelihood_weighting=True, eps=1e-5):
    """Create a loss function for training with arbitrary SDEs.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        train: `True` for training loss and `False` for evaluation loss.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        likelihood_weighting: If `True`, outputs the diffusion variational bound term.
        eps: A `float` number. The smallest time step to sample from.

    Returns:
        A loss function.
    """
    reduce_op = torch.mean if reduce_mean else lambda *args, **kwargs: 0.5 * \
        torch.sum(*args, **kwargs)

    def loss_fn(model, batch, class_labels=None):
        """Compute the loss function.

        Args:
            model: A score model.
            batch: A mini-batch of training data.

        Returns:
            loss: A scalar that represents the average loss value across the mini-batch.
        """
        score_fn = mutils.get_score_fn(sde, model, train=train)
        t = torch.rand(batch.shape[0], device=batch.device) * (sde.T - eps) + eps
        z = torch.randn_like(batch)

        mean, std = sde.marginal_prob(batch, t)
        perturbed_data = cube.reflect(mean + std[:, None, None, None] * z)
        score = score_fn(perturbed_data, t, class_labels=class_labels)
        score_hk = cube.score_hk(perturbed_data, mean, std)

        if not likelihood_weighting:
            losses = (std ** 2)[:, None, None, None] * (score - score_hk).pow(2)
        else:
            g2 = sde.sde(torch.zeros_like(batch), t)[1] ** 2
            losses = g2[:, None, None, None] * (score - score_hk).pow(2)

        losses = reduce_op(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss

    return loss_fn


def get_step_fn(sde, train, optimize_fn=None, reduce_mean=False, likelihood_weighting=False):
    """Create a one-step training/evaluation function.

    Args:
        sde: An `sde_lib.SDE` object that represents the forward SDE.
        optimize_fn: An optimization function.
        reduce_mean: If `True`, average the loss across data dimensions. Otherwise sum the loss across data dimensions.
        likelihood_weighting: If `True`, outputs the diffusion variational bound term.

    Returns:
        A one-step function for training or evaluation.
    """
    loss_fn = get_sde_loss_fn(sde, train, reduce_mean=reduce_mean, likelihood_weighting=likelihood_weighting)

    def step_fn(state, batch, class_labels=None):
        """Running one step of training or evaluation.

        This function will undergo `jax.lax.scan` so that multiple steps can be pmapped and jit-compiled together
        for faster execution.

        Args:
            state: A dictionary of training information, containing the score model, optimizer,
            EMA status, and number of optimization steps.
            batch: A mini-batch of training/evaluation data.

        Returns:
            loss: The average loss value of this state.
        """
        model = state['model']
        if train:
            optimizer = state['optimizer']
            optimizer.zero_grad()
            loss = loss_fn(model, batch, class_labels=class_labels)
            if state['scaler'] is None:
                loss.backward()
            else:
                state['scaler'].scale(loss).backward()
            optimize_fn(optimizer, model.parameters(), step=state['step'], scaler=state['scaler'])
            state['step'] += 1
            state['ema'].update(model.parameters())
        else:
            with torch.no_grad():
                ema = state['ema']
                ema.store(model.parameters())
                ema.copy_to(model.parameters())
                loss = loss_fn(model, batch, class_labels=class_labels)
                ema.restore(model.parameters())

        return loss

    return step_fn
