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
from modules.utilities import load_dataset

if __name__ == '__main__':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # retrain = True
    retrain = False

    # Validation split configuration
    VALIDATION_SPLIT = hparams.validation_split  # 10% of data for validation

    branch_name = safe_branch_name(get_current_branch())
    if retrain:
        # Find the latest model in branch-specific models directory
        latest_model_path = latest_in_branch('models', 'model_*.pt')
        if latest_model_path is None:
            raise FileNotFoundError(f"No model files found in models/{branch_name}/")
        latest_model = str(latest_model_path)
        print(f'Loading checkpoint from: {latest_model}')

        checkpoint = torch.load(latest_model)
        model_state_dict = checkpoint['model_state_dict']
        hparams = SimpleNamespace(**checkpoint['hparams'])
        checkpoint_optimizer_state = checkpoint.get('optimizer_state_dict', getattr(hparams, 'optimizer_state_dict', None))
        checkpoint_scheduler_state = checkpoint.get('scheduler_state_dict', getattr(hparams, 'scheduler_state_dict', None))
        checkpoint_current_lr = getattr(hparams, 'current_lr', None)

        # Keep these on hparams for downstream compatibility.
        hparams.optimizer_state_dict = checkpoint_optimizer_state
        hparams.scheduler_state_dict = checkpoint_scheduler_state
        if checkpoint_current_lr is not None:
            hparams.current_lr = checkpoint_current_lr

        hparams.annealing_steps = 0
        # hparams.n_epochs = 10
        hparams.learning_rate = 1e-3
        # hparams.n_samples = 1_000
        # hparams.batch_size = 50

        vae = TransformerVAE(hparams)
        vae.load_state_dict(model_state_dict)
        print('batch size: ', hparams.batch_size, '')
        print('annealing steps: ', hparams.annealing_steps, '')
        print('n_samples: ', hparams.n_samples, '')
        print('d_model: ', hparams.d_model, '')
        print('global epoch: ', hparams.global_epoch, '')
        print('fixed current lr: ', hparams.current_lr, '')
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

    optim = torch.optim.Adam(vae.parameters(), lr=hparams.learning_rate)

    data = load_dataset(hparams, project_root)


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
        if hparams.optimizer_state_dict is not None:
            optim.load_state_dict(hparams.optimizer_state_dict)
        if hparams.current_lr is not None:
            optim.param_groups[0]['lr'] = hparams.current_lr
        if hparams.scheduler_state_dict is not None:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optim,
                max_lr=hparams.learning_rate,
                steps_per_epoch=len(loader),
                epochs=hparams.n_epochs
            )
            # scheduler.load_state_dict(hparams.scheduler_state_dict)
    else:
        start_lr = getattr(hparams, 'scheduler_start_lr', hparams.learning_rate)
        end_lr = getattr(hparams, 'scheduler_end_lr', hparams.learning_rate * 0.1)
        exp_curve_factor = getattr(hparams, 'scheduler_exp_curve_factor', 1.0)
        total_scheduler_steps = max(len(loader) * hparams.n_epochs, 1)

        for param_group in optim.param_groups:
            param_group['lr'] = start_lr

        def lr_lambda(current_step):
            if total_scheduler_steps <= 1:
                return end_lr / start_lr

            progress = min(max(current_step / (total_scheduler_steps - 1), 0.0), 1.0)

            if math.isclose(exp_curve_factor, 0.0, abs_tol=1e-12):
                curved_progress = progress
            else:
                numerator = math.exp(exp_curve_factor * progress) - 1.0
                denominator = math.exp(exp_curve_factor) - 1.0
                curved_progress = numerator / denominator

            current_lr = start_lr + (end_lr - start_lr) * curved_progress
            return current_lr / start_lr

        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
    # print('scheduler infos: ')
    # print('steps at last epoch: ', scheduler.last_epoch)
    # print('total steps: ', scheduler.total_steps)
    # print('---' * 5)
    # exit()

    


    train_vae(vae, loader, hparams, optim,
              scheduler=scheduler,
              val_loader=val_loader,
              )

    