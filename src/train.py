from hparams import hparams
from modules.model import TransformerVAE, train_vae
import torch
from torch.utils.data import DataLoader
import numpy as np
from types import SimpleNamespace

if __name__ == '__main__':
    # retrain = True
    retrain = False
    model_name = '../models/model-test.pt'

    if retrain:
        checkpoint = torch.load(model_name)
        model_state_dict = checkpoint['model_state_dict']
        hparams = SimpleNamespace(**checkpoint['hparams'])
        hparams.annealing_steps = 0
        # hparams.n_epochs = 10
        hparams.learning_rate = 1e-3
        # hparams.n_samples = 1_000
        # hparams.batch_size = 50


        vae = TransformerVAE(hparams)
        vae.load_state_dict(model_state_dict)
        # retrain()
        # for param in hparams.__dict__:
        #     print(f'{param}: {getattr(hparams, param)}')
        print('batch size: ', hparams.batch_size, '')
        print('annealing steps: ', hparams.annealing_steps, '')
        print('n_samples: ', hparams.n_samples, '')
        print('d_model: ', hparams.d_model, '')
        print('global epoch: ', hparams.global_epoch, '')
        print('fixed current lr: ', hparams.current_lr, '')

        # model_name = 'model_1.pt'
    else:
        vae = TransformerVAE(hparams)
        # model_name = 'model.pt'


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
    dataset_name = '../data/dataset-groove.npz'
    data = np.load(dataset_name)['tensor']
    print('Total amount of training samples: ', data.shape)
    data = data[:hparams.n_samples]
    print('Using ', data.shape[0])

    print('---'*5)

    loader = DataLoader(data, batch_size=BATCH, shuffle=True)
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

    train_vae(vae, loader, hparams, optim,
              device=torch.device('cuda' if torch.cuda.is_available() else 'mps'),
              model_name=model_name,
              scheduler=scheduler,
              step_start=hparams.global_step
              )

    # sample = vae.sample(max_len=SEQ_LEN, temperature=0, sample=False)
    # print('sample shape', sample.shape)