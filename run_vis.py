import io
import os
import os.path

import numpy as np
import torch
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from torchvision.utils import make_grid, save_image
from omegaconf import open_dict

import losses
import sampling
import sde_lib
import utils
from models import utils as mutils
from models import adm, ncsnpp, vdm # needed for creating the model
from models.ema import ExponentialMovingAverage


torch.backends.cudnn.benchmark = True


def visualize(cfg, load_cfg, noise_removal_cfg, log_dir):
    # set up
    logger = utils.get_logger(os.path.join(log_dir, "logs"))
    work_dir = cfg.load_dir

    device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    sde = sde_lib.RVESDE(sigma_min=load_cfg.sde.sigma_min, sigma_max=load_cfg.sde.sigma_max, N=load_cfg.sde.num_scales)
    sampling_eps = 1e-5

    sampling_shape = (cfg.eval.batch_size, load_cfg.data.num_channels, load_cfg.data.image_size, load_cfg.data.image_size)
    sampling_fn = sampling.get_sampling_fn(load_cfg, sde, sampling_shape, sampling_eps, device)
    
    # load in models
    score_model = mutils.create_model(load_cfg).to(device)
    ema = ExponentialMovingAverage(score_model.parameters(), decay=load_cfg.model.ema_rate)
    optimizer = losses.get_optimizer(load_cfg, score_model.parameters())
    scaler = torch.cuda.amp.GradScaler() if load_cfg.model.name == "adm" else None
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, scaler=scaler)

    if noise_removal_cfg is not None:
        noise_removal_model = mutils.create_model(noise_removal_cfg).to(device)
        utils.load_denoising_model(os.path.join(cfg.denoiser_path, "checkpoints/checkpoint.pth"), noise_removal_model)
    else:
        noise_removal_model = None

    ckpt = cfg.eval.ckpt
    if ckpt == -1:
        ckpts = os.listdir(os.path.join(work_dir, "checkpoints"))
        ckpts = [int(x.split(".")[0].split("_")[1]) for x in ckpts]
        ckpt = max(ckpts)

    checkpoint_dir = os.path.join(work_dir, "checkpoints", f"checkpoint_{ckpt}.pth")
    state = utils.restore_checkpoint(checkpoint_dir, state, device, ddp=False)
    ema.copy_to(score_model.parameters())

    # generate images
    this_sample_dir = os.path.join(log_dir, "images")
    utils.makedirs(this_sample_dir)

    if load_cfg.model.name == "adm":
        w = cfg.w * torch.ones(sampling_shape[0], device=device)
        labels = cfg.label * torch.ones(sampling_shape[0], device=device).long()
    else:
        w = None
        labels = None

    logger.info(f"Generating samples for checkpoint {ckpt}")
    for r in range(cfg.eval.rounds):
        logger.info(f"Round {r}")
        samples = sampling_fn(score_model, noise_removal_model=noise_removal_model, weight=w, class_labels=labels)[0]
        samples_np = np.round(samples.clip(min=0, max=1).permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
       
        nrow = int(np.sqrt(samples.shape[0]))
        image_grid = make_grid(samples, nrow, padding=0)
        save_image(image_grid, os.path.join(this_sample_dir, f"samples_{r}.png"))

        with open(os.path.join(this_sample_dir, f"samples_{r}.npz"), "wb") as fout:
            io_buffer = io.BytesIO()
            np.savez_compressed(io_buffer, samples=samples_np)
            fout.write(io_buffer.getvalue())

    logger.info("Finished generating samples.")


from run_vis import *
@hydra.main(version_base=None, config_path="configs", config_name="vis")
def main(cfg):
    hydra_cfg = HydraConfig.get()
    load_cfg = utils.load_hydra_config_from_run(cfg.load_dir)

    log_dir = hydra_cfg.run.dir if hydra_cfg.mode == RunMode.RUN else os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
    utils.makedirs(log_dir)

    # overwrite the sampling instructions
    with open_dict(load_cfg):
        load_cfg.sampling = cfg.sampling

    if cfg.sampling.denoiser == "network":
        noise_removal_cfg = utils.load_hydra_config_from_run(cfg.denoiser_path)
    else:
        noise_removal_cfg = None

    logger = utils.get_logger(os.path.join(log_dir, "logs"))
    logger.info(cfg)
    logger.info(f"loaded in config from {cfg.load_dir}")
    logger.info(load_cfg)
    logger.info(f"Denoising with config?")
    logger.info(noise_removal_cfg)

    try:
        visualize(cfg, load_cfg, noise_removal_cfg, log_dir)
    except Exception as e:
        logger.critical(e, exc_info=True)

if __name__ == "__main__":
    main()