import os
import pickle

import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from modules.modules import VectorQuantizedVAE, AE
from modules.functions import Classifier
from datasets.datasets import get_dataset
from utils.get_argument import get_args, parse_config
del sys.path[-1]


def plot_activate_rate(count_ks, dnames, out_dir='.'):
    figsize = (15, 10)
    zero_acts = []
    for count_k in count_ks:
        np_count_k = np.asarray(count_k)
        zero_act = np.sum(np_count_k == 0, axis=1)
        # print(zero_act)
        zero_acts.append(zero_act)
    plt.figure(figsize=figsize)
    plt.boxplot(zero_acts, labels=dnames)
    plt.xlabel('K')
    plt.ylabel('Number of inactivated code')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'boxplot_check_inactivate.png'))
    plt.close()

    plt.figure(figsize=figsize)
    rand_x = [np.random.normal(loc=i, scale=0.1, size=len(zero_act)) for i, zero_act in enumerate(zero_acts)]
    plt.scatter(rand_x, np.asarray(zero_acts).reshape(-1), s=5, alpha=0.8)
    plt.xticks(range(len(zero_acts)), dnames)
    plt.xlabel('K')
    plt.ylabel('Number of inactivated code')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'scatter_check_inactivate.png'))
    plt.close()


    non_zero_acts = []
    for count_k in count_ks:
        np_count_k = np.asarray(count_k)
        non_zero_act = np.sum(np_count_k != 0, axis=1)
        # print(non_zero_act)
        non_zero_acts.append(non_zero_act)
    plt.figure(figsize=figsize)
    plt.boxplot(non_zero_acts, labels=dnames)
    plt.xlabel('K')
    plt.ylabel('Number of activated code')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'boxplot_check_activate.png'))
    plt.close()

    plt.figure(figsize=figsize)
    rand_x = [np.random.normal(loc=i, scale=0.1, size=len(zero_act)) for i, zero_act in enumerate(non_zero_acts)]
    plt.scatter(rand_x, np.asarray(non_zero_acts).reshape(-1), s=5, alpha=0.8)
    plt.xticks(range(len(non_zero_acts)), dnames)
    plt.xlabel('K')
    plt.ylabel('Number of inactivated code')
    plt.grid(True)
    plt.savefig(os.path.join(out_dir, 'scatter_check_activate.png'))
    plt.close()


def plot_contribute_clf_rate(count_k_labels, effect_codes_ks, thr=0.5, out_dir='.'):
    figsize = (15, 10)

    pss = []
    for count_k_label, effect_codes_k in zip(count_k_labels, effect_codes_ks):
        np_count_k_label = np.asarray(count_k_label)
        np_effective_code = np.asarray(effect_codes_k)
        ps = []
        for np_count, np_effect in zip(np_count_k_label, np_effective_code):
            tmp_index = np_count.sum(axis=0)
            tmp = np.linalg.norm(np_effect, axis=0)
            p = np.mean(tmp[tmp_index > 0] >= thr * tmp.max())
            ps.append(p)
        ps = np.asarray(ps)
        pss.append(ps)
    pss = np.asarray(pss)

    plt.figure(figsize=figsize)
    plt.boxplot(pss.T, labels=[16, 32, 64, 128, 256, 512])
    plt.title('threshold {}'.format(thr))
    plt.xlabel('K')
    plt.ylabel('Rate of contributed codes for classification')
    plt.savefig(os.path.join(out_dir, 'boxplot_check_used_clfy_{}.png'.format(thr)))
    plt.show()

def main(main_args):
    dnames = [
    'mnist_im32_gap_k16_e100',
    'mnist_im32_gap_k32_e100',
    'mnist_im32_gap_k64_e100',
    'mnist_im32_gap_k128_e100',
    'mnist_im32_gap_k256_e100',
    'mnist_im32_gap_k512_e100'
    ]
    n = len(dnames)
    count_ks = [list() for i in range(n)]
    count_k_labels = [list() for i in range(n)]
    effect_codes_ks = [list() for i in range(n)]

    for configs in main_args.configs:
        configs_path = configs
        # dnames  = sorted(os.listdir(configs_path))
        for i, dname in enumerate(dnames):
            print(dname)
            config_path = os.path.join(configs_path, dname, 'config')

            args = get_args(description='VQ-VAE', inputs=['--config', config_path])
            device = args.device
            image_size = args.image_size
            dataset = args.dataset
            data_folder = args.data_folder
            result = get_dataset(dataset, data_folder, image_size=image_size)
            test_dataset = result['test']
            num_channels = result['num_channels']
            hidden_size = args.hidden_size

            # TODO: no gap, BN, bias - model
            if hasattr(args, 'k'):
                k = args.k
                model = VectorQuantizedVAE(num_channels, hidden_size, k, pred=True, transpose=args.resblock_transpose).to(device)
                n_in = hidden_size
                n_out = 10
                predictor = Classifier(n_in, n_out, gap=args.gap).to(device)
            else:
                model = AE(num_channels, hidden_size, pred=True, transpose=args.resblock_transpose).to(device)
                n_in = hidden_size
                n_out = 10
                predictor = Classifier(n_in, n_out, gap=args.gap).to(device)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=main_args.n_image, shuffle=False)

            for n_cv in range(args.kfold):
                m_name = 'models_{}'.format(n_cv)
                model_path = os.path.join(args.root, m_name, args.output_folder, 'best.pt')
                print('load model ==> {}'.format(model_path))
                with open(model_path, 'rb') as f:
                    model.load_state_dict(torch.load(f))

                model_path = os.path.join(args.root, m_name, args.output_folder, 'best_predictor.pt')
                print('load model ==> {}'.format(model_path))
                with open(model_path, 'rb') as f:
                    predictor.load_state_dict(torch.load(f))
                
                fc = predictor.fc
                em_weight = model.codebook.embedding.weight
                em_effect = torch.mm(em_weight, fc.weight.transpose(0, 1)) + fc.bias.unsqueeze(0)
                np_effective_code = em_effect.transpose(1, 0).cpu().detach().numpy()
                effect_codes_ks[i].append(np_effective_code)

                count_k = [0 for _ in range(args.k)]
                count_k_label = [[0 for _ in range(args.k)] for _ in range(10)]
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device)

                    latents = model.encode(images)

                    for k in range(args.k):
                        count_k[k] += torch.sum(latents == k).item()
                        for l in range(10):
                            count_k_label[l][k] += torch.sum(latents[labels == l] == k).item()
                    
                count_ks[i].append(np.asarray(count_k))
                count_k_labels[i].append(np.asarray(count_k_label))
    out_dir = os.path.join(os.path.dirname(configs_path), main_args.output_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    print('write == > {}'.format(out_dir))
    fontsize = 24
    plt.rcParams["font.size"] = fontsize
    plot_activate_rate(count_ks, [16, 32, 64, 128, 256, 512], out_dir)

    thr = 0.5
    print('write == > {}'.format(out_dir))
    plot_contribute_clf_rate(count_k_labels, effect_codes_ks, thr, out_dir)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot codebook effect')
    parser.add_argument('-c', '--configs', type=str, nargs='+', required=True,
        help='path of configs directory')
    parser.add_argument('-m', '--model-type', choices=['AE', 'VQ-VAE'], required=True,
        help='type of model [AE or VQ-VAE]')
    parser.add_argument('-n', '--n-image', type=int, default=256,
        help='number of output images')
    parser.add_argument('--output-folder', type=str, default='activate_rate',
        help='name of the output folder')
    args = parser.parse_args()
    
    main(args)