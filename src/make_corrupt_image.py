import os
import pickle

from tqdm import tqdm
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

from codebook_effect import damage_latent_code
import sys
sys.path.append('..')
from modules.modules import VectorQuantizedVAE, AE
from modules.functions import Classifier
from datasets.datasets import get_dataset
from utils.get_argument import get_args, parse_config
del sys.path[-1]


def make_corrupt_images(images, rec_imgs, corrupt_img, corrupt_label, re_pred, labels, out_dir='.', mse=False, original_label=None):
    fontsize = 15
    plt.rcParams["font.size"] = fontsize
    for i in range(len(images)):
        plt.figure(figsize=(3 * 3, 3))
        plt.subplot(1, 3, 1)
        plt.title('input: class {}'.format(labels[i]))
        tmp = images[i].detach().cpu().permute(1, 2, 0).numpy()
        if tmp.shape[-1] == 1:
            tmp = tmp[:, :, 0]
            plt.imshow(tmp, cmap='gray', vmin=-1, vmax=1)
        else:
            plt.imshow(tmp, vmin=-1, vmax=1)
        plt.axis('off')

        plt.subplot(1, 3, 2)
        if mse:
            loss = F.mse_loss(corrupt_img[i], images[i])
            plt.title('corrupted: {:.4f}'.format(loss))
        else:
            plt.title('corrupted {}, re-pred {}'.format(corrupt_label[i], re_pred[i]))
        tmp = corrupt_img[i].detach().cpu().permute(1, 2, 0).numpy()
        if tmp.shape[-1] == 1:
            tmp = tmp[:, :, 0]
            plt.imshow(tmp, cmap='gray', vmin=-1, vmax=1)
        else:
            plt.imshow(tmp, vmin=-1, vmax=1)
        plt.axis('off')

        plt.subplot(1, 3, 3)
        if mse:
            loss = F.mse_loss(rec_imgs[i], images[i])
            plt.title('reconstruct: {:.4f}'.format(loss))
        else:
            plt.title('reconstruct')
        tmp = rec_imgs[i].detach().cpu().permute(1, 2, 0).numpy()
        if tmp.shape[-1] == 1:
            tmp = tmp[:, :, 0]
            plt.imshow(tmp, cmap='gray', vmin=-1, vmax=1)
        else:
            plt.imshow(tmp, vmin=-1, vmax=1)
        plt.axis('off')
        if original_label is not None:
            i = original_label[i]
        if mse:
            path = os.path.join(out_dir, 'corrupted_image_{:03}_mse.png'.format(i))
        else:
            path = os.path.join(out_dir, 'corrupted_image_{:03}.png'.format(i))
        plt.savefig(path, transparent=True)
        plt.close()

def main(main_args):
    configs_path = main_args.configs
    dnames  = sorted(os.listdir(configs_path))
    for dname in dnames:
        print(dname)
        config_path = os.path.join(configs_path, dname, 'config')

        description = main_args.model_type
        args = get_args(description=description, inputs=['--config', config_path])
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
        images, labels = next(iter(test_loader))
        images = images.to(device)
        labels = labels.to(device)

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

            if isinstance(model, VectorQuantizedVAE):
                rec_imgs, _, latents, _ = model(images)
                preds = predictor(latents)
            elif isinstance(model, AE):
                rec_imgs, latents = model(images)
                preds = predictor(latents)
            
            correct_index = preds.argmax(1) == labels
            result = damage_latent_code(model, predictor, images[correct_index], labels[correct_index], device=args.device)

            out_dir = os.path.join(os.path.dirname(configs_path), main_args.output_folder, dname, 'cv{}'.format(n_cv))
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            print('write == > {}'.format(out_dir))

            corrupt_img = result['corrupt_img']
            corrupt_label = result['corrupt_label']
            re_pred = result['re_pred']
            _images = images[correct_index]
            _labels = labels[correct_index]
            _rec_imgs = rec_imgs[correct_index]

            make_corrupt_images(_images, _rec_imgs, corrupt_img, corrupt_label, re_pred, _labels, 
                                out_dir=out_dir, mse=False, original_label=torch.arange(len(labels), dtype=torch.long)[correct_index])
            make_corrupt_images(_images, _rec_imgs, corrupt_img, corrupt_label, re_pred, _labels,
                                out_dir=out_dir, mse=True, original_label=torch.arange(len(labels), dtype=torch.long)[correct_index])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot codebook effect')
    parser.add_argument('-c', '--configs', type=str, required=True,
        help='path of configs directory')
    parser.add_argument('-m', '--model-type', choices=['AE', 'VQ-VAE'], required=True,
        help='type of model [AE or VQ-VAE]')
    parser.add_argument('-n', '--n-image', type=int, default=25,
        help='number of output images')
    parser.add_argument('--output-folder', type=str, default='corrupted_images',
        help='name of the output folder')
    args = parser.parse_args()
    
    main(args)