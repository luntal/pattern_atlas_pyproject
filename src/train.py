from hparams import hparams
from modules.model import TransformerVAE, train_vae
import torch
from torch.utils.data import DataLoader
import numpy as np
from types import SimpleNamespace
import os
import glob
import math
from modules.branch_paths import latest_in_branch, safe_branch_name, get_current_branch
from modules.utilities import load_dataset, load_scheduler, load_model

if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    branch_name = safe_branch_name(get_current_branch())
    # Validation split configuration
    VALIDATION_SPLIT = hparams.validation_split  # 10% of data for validation


    # retrain = True
    retrain = False
    resume_checkpoint = None
    if retrain:
        vae, hparams, resume_checkpoint = load_model(previous=0, onnx=False, return_checkpoint=True)

        hparams.n_epochs = 20 # override n_epochs for retraining
        # update learning rate
        hparams.scheduler_start_lr = hparams.learning_rate
        hparams.scheduler_end_lr = hparams.learning_rate * 0.9
        # update annealing list to continue from current beta
        w_beta_start = hparams.weighted_beta
        w_beta_end = w_beta_start
        hparams.annealing_list = [[0, w_beta_start], [1,w_beta_end]]
        
    else:
        vae = TransformerVAE(hparams)


    # VOCAB = hparams.vocab_size
    # SEQ_LEN = hparams.max_seq_len
    BATCH = hparams.batch_size
    # latent_dim = hparams.latent_dim
    # seed = 42
    torch.manual_seed(hparams.seed)

    total_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f'Total parameters in VAE: {total_params}')
    weight_decay = getattr(hparams, 'weight_decay', 0.0)
    print(f'Optimizer weight decay: {weight_decay}')

    optim = torch.optim.AdamW(
        vae.parameters(),
        lr=hparams.learning_rate,
        weight_decay=weight_decay,
    )

    data, metadata = load_dataset(hparams)


    # Split into train and validation
    val_size = int(len(data) * VALIDATION_SPLIT)
    train_size = len(data) - val_size
    train_data = data[:train_size]
    val_data = data[train_size:]

    print(f'Train samples: {len(train_data)}')
    print(f'Validation samples: {len(val_data)}')
    print('---'*5)

    # pin_memory is only useful for CUDA host->device transfers.
    # On MPS, enabling it triggers a warning and has no effect.
    use_pin_memory = torch.cuda.is_available()

    # DataLoader with worker tuning for efficient data loading on H100
    loader = DataLoader(
        train_data,
        batch_size=BATCH,
        shuffle=True,
        num_workers=4,
        pin_memory=use_pin_memory,
        persistent_workers=True,
        prefetch_factor=2
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH,
        shuffle=False,
        num_workers=4,
        pin_memory=use_pin_memory,
        persistent_workers=True,
        prefetch_factor=2
    )
    scheduler = None

    if retrain:
        resume_optim_state = None
        if resume_checkpoint is not None:
            resume_optim_state = resume_checkpoint.get('optimizer_state_dict', None)
        if resume_optim_state is None:
            resume_optim_state = getattr(hparams, 'optimizer_state_dict', None)

        if resume_optim_state is not None:
            optim.load_state_dict(resume_optim_state)

        if hparams.learning_rate is not None:
            optim.param_groups[0]['lr'] = hparams.learning_rate


    scheduler = load_scheduler(optim, loader, hparams)

    if retrain and resume_checkpoint is not None:
        resume_scheduler_state = resume_checkpoint.get('scheduler_state_dict', None)
        if resume_scheduler_state is not None:
            scheduler.load_state_dict(resume_scheduler_state)


    train_vae(vae, loader, hparams, optim,
              scheduler=scheduler,
              val_loader=val_loader,
              resume_state=resume_checkpoint,
              )

    