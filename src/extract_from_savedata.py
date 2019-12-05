
import os
from glob import glob
import pickle

import numpy as np
import cv2
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import sys
sys.path.append('..')
from utils.get_argument import get_args, parse_config
del sys.path[-1]

def main(root_path, arg_out_dir=None, output_scalar='scalars_data'):
    if arg_out_dir is not None:
        # eg. root_path: '/{some directory}/logs/'
        # eg. out_dir: '/{some directory}/datas/'
        fnames = os.listdir(root_path)
        paths = map(lambda fname: [root_path, arg_out_dir, fname], fnames)
        if not os.path.exists(arg_out_dir):
            os.makedirs(arg_out_dir)
    else:
        # eg. root_path: '/{some directory}/logs/{model name}*'
        root_paths = map(lambda x: os.path.dirname(x), glob(root_path))
        fnames = map(lambda x: os.path.basename(x), glob(root_path))
        out_dirs = map(lambda x: 
                    os.path.join(os.path.dirname(os.path.dirname(x)), 'datas'),
                    glob(root_path))
        paths = zip(root_paths, out_dirs, fnames)


    # Not change coeffs
    recon_coeff = 1.0
    vq_coeff = 1.0
    beta = 1.0
    gamma = 1.0


    datas = {}
    for root_path, out_dir, fname in paths:
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        config_path = os.path.join(os.path.dirname(root_path), 'configs', fname, 'config')
        config = parse_config(config_path)
        args = get_args(description='VQ-VAE', inputs=config)
        print(args)

        path = os.path.join(root_path, fname)
        event_path = glob(os.path.join(path, 'events*'))[0]
        print('load ==> {}'.format(event_path))
        event_acc = EventAccumulator(event_path)
        event_acc.Reload()
        # save scalar data
        scalars = {}
        # print(event_acc.Tags()['scalars'])
        for s_tag in event_acc.Tags()['scalars']:
            events = event_acc.Scalars(s_tag)
            scalars[s_tag] = np.asarray([event.value for event in events])

        # calc total loss
        if '_ae_' in fname:
            # ae
            a = recon_coeff
            b = 0
            c = 0
            d = gamma
        else:
            # vqvae
            a = recon_coeff
            b = vq_coeff
            c = beta
            d = gamma

        if 'no_recon' in fname:
            a = 0
        if 'no_pred' in fname:
            d = 0

        if args.kfold > 0:
            for cnt in range(args.kfold):
                tag = 'cv{:02}'.format(cnt)
                # print(scalars.keys())
                if hasattr(args, 'k'):
                    phase = 'train'
                    scalars['{}/loss/{}'.format(tag, phase)] = a * scalars['{}/loss/{}/reconstruction'.format(tag, phase)] \
                                                             + (b + c) * scalars['{}/loss/{}/quantization'.format(tag, phase)] \
                                                             + d * scalars['{}/loss/{}/prediction'.format(tag, phase)]
                    phase = 'test'
                    scalars['{}/loss/{}'.format(tag, phase)] = a * scalars['{}/loss/{}/reconstruction'.format(tag, phase)] \
                                                             + (b + c) * scalars['{}/loss/{}/quantization'.format(tag, phase)] \
                                                             + d * scalars['{}/loss/{}/prediction'.format(tag, phase)]
                else:
                    phase = 'train'
                    scalars['{}/loss/{}'.format(tag, phase)] = a * scalars['{}/loss/{}/reconstruction'.format(tag, phase)] \
                                                             + d * scalars['{}/loss/{}/prediction'.format(tag, phase)]
                    phase = 'test'
                    scalars['{}/loss/{}'.format(tag, phase)] = a * scalars['{}/loss/{}/reconstruction'.format(tag, phase)] \
                                                             + d * scalars['{}/loss/{}/prediction'.format(tag, phase)]
        else:            
            if 'loss/train/quantization' in scalars:
                scalars['loss/train'] = a * scalars['loss/train/reconstruction'] + (b + c) * scalars['loss/train/quantization'] + d * scalars['loss/train/prediction']
            else:
                scalars['loss/train'] = a * scalars['loss/train/reconstruction'] + d * scalars['loss/train/prediction']

            if 'loss/test/quantization' in scalars:
                scalars['loss/test'] = a * scalars['loss/test/reconstruction'] + (b + c) * scalars['loss/test/quantization'] + d * scalars['loss/test/prediction']
            else:
                scalars['loss/test'] = a * scalars['loss/test/reconstruction'] + d * scalars['loss/test/prediction']

        datas[fname] = scalars

        out_dir_fname = os.path.join(out_dir, fname)
        if not os.path.exists(out_dir_fname):
            os.makedirs(out_dir_fname)
        # save image data
        for i_tag in event_acc.Tags()['images']:
            events = event_acc.Images(i_tag)
            tag_name = i_tag.replace('/', '_')
            for index, event in enumerate(events):
                s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
                image = cv2.imdecode(s, cv2.IMREAD_COLOR)
                out_name = '{}_{:03}.jpg'.format(tag_name, event.step)
                outpath = os.path.join(out_dir_fname, out_name)
                print('write image ==> {}'.format(outpath))
                cv2.imwrite(outpath, image)
        if arg_out_dir is None:
            pkl_path = os.path.join(out_dir_fname, '{}.pkl'.format(output_scalar))
            with open(pkl_path, 'wb') as f:
                pickle.dump(datas, f)
            datas = {}

    if arg_out_dir is not None:
        pkl_path = os.path.join(out_dir, '{}.pkl'.format(output_scalar))
        with open(pkl_path, 'wb') as f:
            pickle.dump(datas, f)

if __name__ == '__main__':
    import sys
    import argparse
    description = 'extract data'
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('-g', '--glob', type=str, default=None,
        help='glob path')
    parser.add_argument('-r', '--log-root', type=str, default=None,
        help='log directory path')
    parser.add_argument('-o', '--output-scalar', type=str, default='scalars_data',
        help='output name of scalar data')

    args = parser.parse_args()

    if args.glob is None and args.log_root is None:
        raise ValueError

    # root_path = sys.argv[1]
    # if len(sys.argv) <= 2:
    #     dirname = os.path.dirname(root_path.rstrip('/'))
    #     out_dir = os.path.join(dirname, 'datas')
    # else:
    #     out_dir = sys.argv[2]

    if args.glob is None:
        root_path = args.log_root.rstrip('/')
        dirname = os.path.dirname(args.log_root.rstrip('/'))
        out_dir = os.path.join(dirname, 'datas')
        main(root_path, out_dir, output_scalar=args.output_scalar)
    elif args.log_root is None:
        main(os.path.expanduser(args.glob), output_scalar=args.output_scalar)
    else:
        raise ValueError
    

