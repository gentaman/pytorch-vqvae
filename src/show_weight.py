import sys
sys.path.append('../')

import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from modules.modules import VectorQuantizedVAE
from modules.functions import Classifier
from datasets.datasets import get_dataset
from utils import accuracy, get_args


def make_dir(out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

def plot_hist(em_weight, out_dir):
    make_dir(out_dir)

    for i, w in enumerate(em_weight):
        plt.figure(figsize=(8, 8))
        plt.hist(w, range=(em_weight.min(), em_weight.max()), bins=100)
        plt.tight_layout()
        out_path = os.path.join(out_dir, 'hist_{}_emweight.png'.format(str(i).zfill(4)))
        plt.savefig(out_path)
        plt.close()

def reconstruct_sample(model, images, labels, predictor, out_dir, prefix=None, max_samples=10):
    make_dir(out_dir)
    result = model(images)
    preds = predictor(result[-1])
    stack = zip(preds.argmax(1).cpu().numpy(), images.cpu().numpy(), result[0].detach().cpu().numpy())
    for i, (pred, im, r_im) in enumerate(stack):
        if i > max_samples:
            break
        plt.subplot(1, 4, 1)
        plt.title('original')
        plt.imshow(im[0], vmax=1.0, vmin=-1.0)
        plt.subplot(1, 4, 2)
        plt.title('reconstruction')
        plt.imshow(r_im[0], vmax=1.0, vmin=-1.0)
        plt.subplot(1, 4, 3)
        plt.title('abs diff')
        plt.imshow(abs(im[0] - r_im[0]), vmax=1.0, vmin=-1.0)
        plt.subplot(1, 4, 4)
        ax = plt.gca()
        plt.text(0.5, 0.5, 'True:{}\nPred:{}\nMSE:{:.4f}'.format(
            labels[i],
            pred,
            F.mse_loss(torch.Tensor(im), torch.Tensor(r_im)).item(),
            ),
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes
            )
        if prefix is None:
            out_path = os.path.join(out_dir, '{}_rec.png'.format(str(i).zfill(3)))
        else:
            out_path = os.path.join(out_dir, '{}_{}_rec.png'.format(prefix, str(i).zfill(3)))
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()


def plot_codebooks(model, fmap_size, out_dir, device='cuda', each_normalize=True):
    make_dir(out_dir)
    em_weight = model.codebook.embedding.weight.cpu().detach().numpy()
    k, hidden_size = em_weight.shape
    vmin = np.inf
    vmax = -np.inf
    outs = []
    for j in range((fmap_size)**2):
        d = torch.zeros(k, hidden_size, fmap_size, fmap_size)
        h_index, w_index = np.unravel_index(j, (fmap_size, fmap_size))
        d[:, :, h_index, w_index] = torch.Tensor(em_weight)
        out = model.decoder(d.to(device))
        outs.append(out.cpu())
        if not each_normalize:
            if vmin > out1.min():
                vmin = out1.min()
            if vmax < out1.max():
                vmax = out1.max()

    for i in range(k):
        plt.figure(figsize=(16, 16))
        for j in range((fmap_size)**2):
            # d = torch.zeros(k, hidden_size, fmap_size, fmap_size)
            # h_index, w_index = np.unravel_index(j, (fmap_size, fmap_size))
            # d[:, :, h_index, w_index] = torch.Tensor(em_weight)
            # out1 = model.decoder(d.to(device))
            plt.subplot(fmap_size, fmap_size, j+1)
            plt.title('index: {}-{}'.format(h_index, w_index))
            if each_normalize:
                plt.imshow(outs[j][i, 0].cpu().detach().numpy())
            else:
                plt.imshow(outs[j][i, 0].cpu().detach().numpy(), vmin=vmin, vmax=vmax)
            plt.axis('off')
        plt.tight_layout()
        out_path = os.path.join(out_dir, '{}-{}-{}.png'.format(str(i).zfill(3), h_index, w_index))
        plt.savefig(out_path)
        plt.close()

def plot_perform_model(args, num_channels, n_out, data_loader, model_path, predictor_path=None):

    root = args.root
    dirname = os.path.dirname(model_path)
    dirname = os.path.basename(dirname)
    path = os.path.join(root, args.output_folder, dirname)


    model = VectorQuantizedVAE(
        num_channels, args.hidden_size, args.k, pred=True, transpose=args.resblock_transpose, BN=args.BN
        ).to(args.device)

    if args.hidden_fmap_size is None:
        hidden_fmap_size = args.image_size // 4
    else:
        hidden_fmap_size = args.hidden_fmap_size
    if args.gap:
        n_in = args.hidden_size
    else:
        n_in = int(args.hidden_size * hidden_fmap_size * hidden_fmap_size)
    
    predictor = Classifier(n_in, n_out, gap=args.gap).to(args.device)

    if model_path:
        print("load model ==> {}".format(model_path))
        with open(model_path, 'rb') as f:
            state_dict = torch.load(f)
            try:
                model = copy_model(state_dict, model, verbose=1)
            except:
                model.load_state_dict(state_dict)
    if predictor_path:
        print("load predictor ==> {}".format(predictor_path))
        with open(predictor_path, 'rb') as f:
            state_dict = torch.load(f)
            predictor.load_state_dict(state_dict)
    
    # plot histogram
    print('plot histogram')
    em_weihgt = model.codebook.embedding.weight.cpu().detach().numpy()
    plot_hist(em_weihgt, os.path.join(path, 'hist'))

    # reconstruct test sample
    print('reconstruct test sample')
    images, labels = next(iter(data_loader))
    out_dir = os.path.join(path, 'rec_imgs')
    reconstruct_sample(model, images.to(args.device), labels, predictor, out_dir)

    print('generate codebook')
    if args.each_normalize:
        out_dir = os.path.join(path, 'gen_codebook')
    else:
        out_dir = os.path.join(path, 'gen_codebook_norm')
    plot_codebooks(model, args.image_size//8, out_dir, device=args.device, each_normalize=args.each_normalize)


def main(args):
    result = get_dataset(args.dataset, args.data_folder, image_size=args.image_size)

    train_dataset = result['train']
    test_dataset = result['test']
    valid_dataset = result['valid']
    num_channels = result['num_channels']

    # Define the data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
        batch_size=args.batch_size, shuffle=False, drop_last=True,
        num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
        batch_size=args.batch_size, shuffle=False)
    
    n_out = len(train_dataset._label_encoder)
    for model_path in glob(os.path.join(args.root, 'models', '*', 'best.pt')):
        dirname = os.path.dirname(model_path)
        pred_path = os.path.join(dirname, 'best_predictor.pt')
        print('model performance: {}'.format(model_path))
        if not os.path.exists(pred_path):
            pred_path = None
        else:
            print('predictor performance: {}'.format(pred_path))
        
        plot_perform_model(args, num_channels, n_out, test_loader, model_path, pred_path)

if __name__ == '__main__':
    import os
    import sys

    args = get_args(description='Performance VQ-VAE')
    
    # Create logs and models folder if they don't exist
    root = args.root
    log_path = os.path.join(root, 'logs')
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    models_path = os.path.join(root, 'models')
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    path = os.path.join(root, 'configs', args.output_folder)
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, 'config'), 'w') as f:
        f.write('\n'.join(sys.argv))
    
    # Device
    args.device = torch.device(args.device
        if torch.cuda.is_available() else 'cpu')
    # Slurm
    if 'SLURM_JOB_ID' in os.environ:
        args.output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])
    path = os.path.join(models_path, args.output_folder)
    if not os.path.exists(path):
        os.makedirs(path)
    args.steps = 0

    main(args)