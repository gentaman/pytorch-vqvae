import os
import pickle

from tqdm import tqdm
import torch.nn.functional as F
import torch

import sys
sys.path.append('..')
from modules.modules import VectorQuantizedVAE, AE
from modules.functions import Classifier
from datasets.datasets import get_dataset
from utils.get_argument import get_args, parse_config
del sys.path[-1]


def damage_latent_code(model, predictor, input_imgs, labels, device='cpu'):
    with torch.no_grad():
        fc = predictor.fc
        fc_weight = fc.weight
        fc_bias = fc.bias

        if isinstance(model, VectorQuantizedVAE):
            em_weight = model.codebook.embedding.weight
            hidden_size = fc_weight.shape[1]
            em_effect = torch.mm(em_weight, fc_weight.transpose(0, 1)) + fc_bias.unsqueeze(0)
            effective_codes = torch.argsort(em_effect.transpose(0, 1), dim=1, descending=True)
            latents = model.encode(input_imgs)
            zpad_pos = torch.zeros_like(latents, dtype=torch.uint8, device=device)
            effectived_index = torch.ones_like(em_effect[0, labels], dtype=torch.uint8)
            effect_code_a_class = effective_codes[labels.long(), :].float()
            batch_size, fmap_size, fmap_size = latents.shape
            # max iteration = k
            for i in range(effective_codes.shape[1]):
                index = effect_code_a_class[:, i].long()
                effectived_index = effectived_index.long() - (em_effect[index, labels] <= 0).long()
                effectived_index = effectived_index > 0
                if effectived_index.sum() == 0:
                    break
                tmp = effect_code_a_class[:, i].unsqueeze(-1).unsqueeze(-1) * torch.ones(batch_size, fmap_size, fmap_size).to(device)
                zpad_pos[effectived_index, :] += (latents == tmp.long())[effectived_index, :]
                zpad_pos = (zpad_pos > 0)
                z_q_x = model.codebook.embedding(latents.view(batch_size, -1)).view(batch_size, fmap_size, fmap_size, hidden_size).permute(0, 3, 1, 2)
                z_q_x = z_q_x.permute(0, 2, 3, 1)
                z_q_x[zpad_pos, :] = 0
                z_q_x = z_q_x.permute(0, 3, 1, 2)
                pred0 = predictor(z_q_x)
                effectived_index = effectived_index.long() - (pred0.argmax(1) != labels).long()
                effectived_index = effectived_index > 0

            corrupt_img = model.decoder(z_q_x)
            _, _, z, _ = model(corrupt_img)
            pred1 = predictor(z)
            result = {
                'corrupt_img': corrupt_img,
                'corrupt_label': pred0.argmax(1),
                're_pred': pred1.argmax(1),
            }
        elif isinstance(model, AE):
            _, codes = model(input_imgs)
            batch_size, hidden_size, fmap_size, fmap_size = codes.shape
            a = codes.view(batch_size, hidden_size, int(fmap_size**2)).permute(0, 2, 1)
            effect_code = torch.matmul(a, fc.weight.transpose(0, 1))
            effect_code = effect_code.permute(0, 2, 1)

            # must use long()
            effect_code_a_class = effect_code[range(len(labels)), labels.long(), :]

            values, indeces = torch.sort(effect_code_a_class, dim=-1, descending=True)
            effectived_index = torch.ones(batch_size, dtype=torch.uint8, device=device)
            zpad_pos = torch.zeros_like(codes, dtype=torch.uint8, device=device)
            # max iteration = feature map size ** 2
            for value, index in zip(values.transpose(0, 1), indeces.transpose(0, 1)):
                effectived_index = effectived_index.long() - (value <= 0).long()
                effectived_index = effectived_index > 0
                if effectived_index.sum() <= 0:
                    break
                w_index = (index % fmap_size)[effectived_index]
                h_index = (index // fmap_size)[effectived_index]
                zpad_pos[effectived_index, :, h_index, w_index] = 1
                codes[zpad_pos] = 0.
                pred0 = predictor(codes)
                effectived_index = effectived_index.long() - (pred0.argmax(1) != labels).long()
                effectived_index = effectived_index > 0
            corrupt_img = model.decoder(codes)
            _, pred1 = model(corrupt_img)
            pred1 = predictor(pred1)
            result = {
                'corrupt_img': corrupt_img,
                'corrupt_label': pred0.argmax(1),
                're_pred': pred1.argmax(1),
            }
        else:
            raise ValueError('unknow model type {}'.format(type(model)))

    return result

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
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        scalars = [dict() for i in range(args.kfold)]
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

            mse_loss = 0.
            re_mse_loss = 0.
            re_corrects = 0.
            n_corrects = 0
            for images, labels in tqdm(test_loader, total=len(test_loader)):
                images = images.to(device)
                labels = labels.to(device)
                if isinstance(model, VectorQuantizedVAE):
                    rec_imgs, _, latents, _ = model(images)
                    preds = predictor(latents)
                elif isinstance(model, AE):
                    rec_imgs, latents = model(images)
                    preds = predictor(latents)

                correct_index = preds.argmax(1) == labels
                result = damage_latent_code(model, predictor, images[correct_index], labels[correct_index], device=args.device)
                re_mse_loss += F.mse_loss(result['corrupt_img'], images[correct_index])
                mse_loss += F.mse_loss(rec_imgs[correct_index], images[correct_index])
                re_corrects += torch.sum(result['re_pred'] == labels[correct_index]).float()
                n_corrects += torch.sum(correct_index).float()
            scalars[n_cv]['mse_loss'] = mse_loss.item()
            scalars[n_cv]['re_mse_loss'] = re_mse_loss.item()
            scalars[n_cv]['n_corrects'] = n_corrects.item()
            scalars[n_cv]['re_corrects'] = re_corrects.item()

        dname = os.path.dirname(config_path)
        out_dir = os.path.join(dname, main_args.output_folder, dname)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        
        out_path = os.path.join(out_dir, 'scalars.pkl')
        with open(out_path, 'wb') as f:
            pickle.dump(scalars, f)
            print('write scalars ==> {}'.format(out_path))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot codebook effect')
    parser.add_argument('-c', '--configs', type=str, required=True,
        help='path of configs directory')
    parser.add_argument('-m', '--model-type', type=str, required=True,
        help='type of model [AE or VQ-VAE]')
    parser.add_argument('--output-folder', type=str, default='codebook_effect',
        help='name of the output folder')
    args = parser.parse_args()
    
    main(args)
