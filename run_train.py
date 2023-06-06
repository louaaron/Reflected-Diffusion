import datetime
import gc
import os
import os.path

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import make_grid, save_image

import datasets
import losses
import sampling
import sde_lib
import utils
from models import adm, ncsnpp, vdm
from models import utils as mutils
from models import vdm
from models.ema import ExponentialMovingAverage

torch.backends.cudnn.benchmark = True


def setup(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)

    # initialize the process group
    dist.init_process_group(
        "nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30)
    )


def cleanup():
    dist.destroy_process_group()


def run_multiprocess(rank, world_size, cfg, work_dir, port):
    try:
        setup(rank, world_size, port)
        _run(rank, world_size, work_dir, cfg)
    finally:
        cleanup()


def _run(rank, world_size, work_dir, cfg):

    # Create directories for experimental logs
    sample_dir = os.path.join(work_dir, "samples")
    checkpoint_dir = os.path.join(work_dir, "checkpoints")
    checkpoint_meta_dir = os.path.join(work_dir, "checkpoints-meta", "checkpoint.pth")
    if rank == 0:
        utils.makedirs(sample_dir)
        utils.makedirs(checkpoint_dir)
        utils.makedirs(os.path.dirname(checkpoint_meta_dir))

    # logging
    if rank == 0:
        logger = utils.get_logger(os.path.join(work_dir, "logs"))

    def mprint(msg):
        if rank == 0:
            logger.info(msg)

    # construct models etc...
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    score_model = mutils.create_model(cfg).to(device)
    score_model = DDP(score_model, device_ids=[rank], static_graph=True, find_unused_parameters=True)
    if torch.__version__.startswith('1.14'):
        score_model = torch.compile(score_model)
    ema = ExponentialMovingAverage(
        score_model.parameters(), decay=cfg.model.ema_rate)
    scaler = torch.cuda.amp.GradScaler() if cfg.model.name == "adm" else None
    optimizer = losses.get_optimizer(cfg, score_model.parameters())

    mprint(score_model)
    mprint(f"EMA: {ema}")
    mprint(f"Optimizer: {optimizer}")
    mprint(f"Scaler: {scaler}.")
    
    state = dict(optimizer=optimizer, model=score_model, ema=ema, step=0, scaler=scaler) 

    state = utils.restore_checkpoint(checkpoint_meta_dir, state, device)
    initial_step = int(state['step'])

    # Build data iterators
    train_ds, eval_ds = datasets.get_dataset(cfg)

    train_iter = iter(train_ds)
    eval_iter = iter(eval_ds)

    sde = sde_lib.RVESDE(sigma_min=cfg.sde.sigma_min, sigma_max=cfg.sde.sigma_max, N=cfg.sde.num_scales)
    sampling_eps = 1e-5

    # Build one-step training and evaluation functions
    optimize_fn = losses.optimization_manager(cfg)
    reduce_mean = cfg.training.reduce_mean
    likelihood_weighting = cfg.training.likelihood_weighting
    train_step_fn = losses.get_step_fn(sde, 
                                       train=True, 
                                       optimize_fn=optimize_fn,
                                       reduce_mean=reduce_mean, 
                                       likelihood_weighting=likelihood_weighting)
    eval_step_fn = losses.get_step_fn(sde, 
                                      train=False, 
                                      optimize_fn=optimize_fn,
                                      reduce_mean=reduce_mean, 
                                      likelihood_weighting=likelihood_weighting)

    # Build samping functions
    if cfg.training.snapshot_sampling:
        sampling_shape = (cfg.training.batch_size // cfg.ngpus, 
                          cfg.data.num_channels,
                          cfg.data.image_size, 
                          cfg.data.image_size)
        sampling_fn = sampling.get_sampling_fn(
            cfg, sde, sampling_shape, sampling_eps, device)

    num_train_steps = cfg.training.n_iters
    mprint(f"Starting training loop at step {initial_step}.")

    for step in range(initial_step, num_train_steps + 1):
        # clear out memory
        torch.cuda.empty_cache()
        gc.collect()

        batch = next(train_iter)
        batch_imgs = batch[0].to(device)
        batch_class = batch[1].to(device) if cfg.data.classes else None
        loss = train_step_fn(state, batch_imgs, class_labels=batch_class)

        if step % cfg.training.log_freq == 0:
            mprint("step: %d, training_loss: %.5e" % (step, loss.item()))
        
        # save checkpoint periodically
        if step != 0 and step % cfg.training.snapshot_freq_for_preemption == 0 and rank == 0:
            utils.save_checkpoint(checkpoint_meta_dir, state)

        # print out eval loss
        if step % cfg.training.eval_freq == 0:
            eval_batch = next(eval_iter)
            batch_imgs = eval_batch[0].to(device)
            batch_class = eval_batch[1].to(device) if cfg.data.classes else None
            eval_loss = eval_step_fn(state, batch_imgs)
            mprint("step: %d, evaluation_loss: %.5e" % (step, eval_loss.item()))

        if step != 0 and step % cfg.training.snapshot_freq == 0 or step == num_train_steps:
            # Save the checkpoint.
            save_step = step // cfg.training.snapshot_freq
            if rank == 0:
                utils.save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)

            # Generate and save samples
            if cfg.training.snapshot_sampling:
                mprint(f"Generating images at step: {step}")

                if cfg.data.classes:
                    weight = 4 * torch.rand(sampling_shape[0]).to(device)
                    class_labels = torch.randint(0, cfg.data.num_classes, (sampling_shape[0],)).to(device)
                else:
                    weight = None
                    class_labels = None

                ema.store(score_model.parameters())
                ema.copy_to(score_model.parameters())
                sample, n = sampling_fn(score_model, weight=weight, class_labels=class_labels)
                ema.restore(score_model.parameters())

                this_sample_dir = os.path.join(sample_dir, "iter_{}".format(step))
                utils.makedirs(this_sample_dir)
                nrow = int(np.sqrt(sample.shape[0]))
                image_grid = make_grid(sample, nrow, padding=2)
                sample = np.clip(np.round(sample.permute(0, 2, 3, 1).cpu().numpy() * 255), 0, 255).astype(np.uint8)
                np.save(os.path.join(this_sample_dir, f"sample_{rank}"), sample)
                save_image(image_grid, os.path.join(this_sample_dir, f"sample_{rank}.png"))
                dist.barrier()


from run_train import run_multiprocess
@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg):
    hydra_cfg = HydraConfig.get()
    work_dir = hydra_cfg.run.dir if hydra_cfg.mode == RunMode.RUN else os.path.join(hydra_cfg.sweep.dir, hydra_cfg.sweep.subdir)
    utils.makedirs(work_dir)

	# Run the training pipeline
    port = int(np.random.randint(10000, 20000))
    logger = utils.get_logger(os.path.join(work_dir, "logs"))

    hydra_cfg = HydraConfig.get()
    if hydra_cfg.mode != RunMode.RUN:
        logger.info(f"Run id: {hydra_cfg.job.id}")

    try:
        mp.set_start_method("forkserver")
        mp.spawn(run_multiprocess, args=(cfg.ngpus, cfg, work_dir, port), nprocs=cfg.ngpus, join=True)
    except Exception as e:
        logger.critical(e, exc_info=True)


if __name__ == "__main__":
    main()