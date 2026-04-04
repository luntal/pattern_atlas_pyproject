from hparams import hparams
from modules.model import TransformerVAE, train_vae
import torch
from torch.utils.data import DataLoader
import numpy as np
from types import SimpleNamespace
import os
import glob

if __name__ == '__main__':
    # retrain = True
    retrain = False

    # Validation split configuration
    VALIDATION_SPLIT = hparams.validation_split  # 10% of data for validation

    if retrain:
        # Find the latest model in the models directory
        model_files = glob.glob('../models/model_*.pt')
        if not model_files:
            raise FileNotFoundError("No model files found in ../models/ directory")
        latest_model = max(model_files, key=os.path.getmtime)
        print(f'Loading checkpoint from: {latest_model}')

        checkpoint = torch.load(latest_model)
        model_state_dict = checkpoint['model_state_dict']
        hparams = SimpleNamespace(**checkpoint['hparams'])
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


    VOCAB = hparams.vocab_size
    SEQ_LEN = hparams.max_seq_len
    BATCH = hparams.batch_size
    latent_dim = hparams.latent_dim
    seed = 42
    torch.manual_seed(seed)

    total_params = sum(p.numel() for p in vae.parameters() if p.requires_grad)
    print(f'Total parameters in VAE: {total_params}')



    optim = torch.optim.Adam(vae.parameters(), lr=hparams.learning_rate)

    # data = torch.randint(low=1, high=VOCAB, size=(BATCH, SEQ_LEN))
    dataset_name = '../data/' + hparams.dataset_name
    data = np.load(dataset_name)['data']
    print('Total amount of samples: ', data.shape)
    data = data[:hparams.n_samples]
    print('Using ', data.shape[0])

    print('---'*5)

    # Shuffle data before splitting
    np.random.seed(seed)
    np.random.shuffle(data)

    # Split into train and validation
    val_size = int(len(data) * VALIDATION_SPLIT)
    train_size = len(data) - val_size
    train_data = data[:train_size]
    val_data = data[train_size:]

    print(f'Train samples: {len(train_data)}')
    print(f'Validation samples: {len(val_data)}')
    print('---'*5)

    loader = DataLoader(train_data, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=BATCH, shuffle=False)
    scheduler = None

    if retrain:
        optim.load_state_dict(hparams.optimizer_state_dict)
        optim.param_groups[0]['lr'] = hparams.current_lr
    else:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optim,
            max_lr=hparams.learning_rate,
            steps_per_epoch=len(loader),
            epochs=hparams.n_epochs
        )
    # print('scheduler infos: ')
    # print('steps at last epoch: ', scheduler.last_epoch)
    # print('total steps: ', scheduler.total_steps)
    # print('---' * 5)
    # exit()

    # Device selection with priority: CUDA > MPS > CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f'Using CUDA device: {torch.cuda.get_device_name(0)}')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print('Using MPS device (Apple Silicon)')
    else:
        device = torch.device('cpu')
        print('Using CPU device')

    train_vae(vae, loader, hparams, optim,
              device=device,
              scheduler=scheduler,
              step_start=hparams.global_step,
              val_loader=val_loader
              )

    # sample = vae.sample(max_len=SEQ_LEN, temperature=0, sample=False)
    # print('sample shape', sample.shape)