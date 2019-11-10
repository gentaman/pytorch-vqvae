import pickle
import os
from glob import glob
import re
import random
import sys
sys.path.append('..')
from utils.get_argument import get_args, parse_config
del sys.path[-1]
from fnmatch import fnmatch

import numpy as np
import matplotlib.pyplot as plt

def parse_ksize(name):
    s_key = '_k'
    e_key = '_'
    s_index = name.find(s_key)
    if s_index == -1:
        return -1
    sep_name = name[s_index + len('_k'):]
    e_index = sep_name.find(e_key)
    if e_index == -1:
        return -1
    
    return int(name[s_index + len('_k') :s_index + len('_k') + e_index])


def get_plotargs(name, train_vs_test=False):
    label_name = ''
    ksize = parse_ksize(name)
    if 'k512' in name:
        label_name = label_name + 'k512'
        c1 = 224
        c2 = 80
        c3 = 224
        vq = True
    elif 'k256' in name:
        label_name = label_name + 'k256'
        c1 = 224
        c2 = 224
        c3 = 80
        vq = True
    elif 'k128' in name:
        label_name = label_name + 'k128'
        c1 = 224
        c2 = 80
        c3 = 80
        vq = True
    elif 'k32' in name:
        label_name = label_name + 'k32'
        c1 = 80
        c2 = 224
        c3 = 80
        vq = True
    elif 'k16' in name:
        label_name = label_name + 'k16'
        c1 = 80
        c2 = 224
        c3 = 224
        vq = True
    elif 'k4' in name:
        label_name = label_name + 'k4'
        if 'k4-2' in name:
            c1 = 224
            c2 = 80
            c3 = 224
        else:
            c1 = 80
            c2 = 80
            c3 = 224
        vq = True
    elif ksize > -1:
        label_name = label_name + 'k' + str(ksize)
        random.seed(ksize)
        c1 = random.randrange(0, 255, 20)
        c2 = random.randrange(0, 255, 20)
        c3 = random.randrange(0, 255, 20)
        vq = True

    else:
        c1 = 0
        c2 = 0
        c3 = 0
        vq = False
            
    if 'no_recon' in name:
        if vq:
            label_name = ' '.join(['VQ-CNN', label_name])
        else:
            label_name = ' '.join(['CNN', label_name])
        ls = '--'
    elif 'no_pred' in name:
        if vq:
            label_name = ' '.join(['VQ-VAE', label_name])
        else:
            label_name = ' '.join(['AE', label_name])
        ls = ':'
    else:
        if vq:
            label_name = ' '.join(['multi-VQ-VAE', label_name])
        else:
            label_name = ' '.join(['multi-AE', label_name])
        ls = '-'

    if 'fashion-mnist' in name:
        r = c3
        g = c1
        b = c2
        color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        label_name = ' '.join([label_name, 'fashion-MNIST'])
    elif 'mnist' in name:
        r = c1
        g = c2
        b = c3
        color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
        label_name = ' '.join([label_name, 'MNIST'])
    else:
        r = c1
        g = c2
        b = 0
        color = '#{:02x}{:02x}{:02x}'.format(r, g, b)
    
    
    args = {
        'linestyle': ls,
        'color': color,
        'label': label_name
    }
    return args

def global_summary(scalar_data, out_dir, kfold=-1, plot_mean_value=False):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fontsize = 24
    plt.rcParams["font.size"] = fontsize
    num_epoch = 100
    figsize = (16, 9)

    figs_num = {}
    y_min_max = {}

    def rm_key(key):
        if 'k4-2' in key:
            return True
        
        if 'k4' in key:
            return False
        
        return True

    file_keys = list(filter(rm_key, scalar_data.keys()))
    file_keys = sorted(file_keys, key=lambda x: parse_ksize(x))
    for file_key in file_keys:
        scalar = scalar_data[file_key]
        plotargs = get_plotargs(file_key)

        if kfold > 0:
            s_keys = scalar.keys()
            set_keys = set(map(lambda x: x[x.find('/') + 1:], s_keys))
            dict_keys = {i: list(filter(lambda x: fnmatch(x, '*/{}'.format(i)), s_keys)) for i in set_keys}
            for data_key in dict_keys:
                cv_keys = dict_keys[data_key]

                key_without_train_test = re.sub('train|test|val', '', data_key)
                if data_key not in figs_num:
                    fig = plt.figure(figsize=figsize)
                    figs_num[data_key] = fig.number
                    ax = fig.add_subplot(1, 1, 1)
                    ax.set_title(data_key)
                else:
                    fig = plt.figure(figs_num[data_key])
                    ax = fig.axes[0]
                data = []
                for cv_key in cv_keys:
                    data.append(scalar[cv_key])
                data = np.asarray(data)

                if 'train' in data_key:
                    print(file_key, data_key)
                    data = data.reshape(kfold, num_epoch, -1).mean(-1)
                if 'label' not in plotargs:
                    plotargs["label"] = file_key
                
                if plot_mean_value:
                    plot_value = data.mean(0)
                else:
                    q75, q25, plot_value = np.percentile(data, [75, 25, 50], axis=0)
                    fill_min = q25
                    fill_max = q75
                                    
                ax.plot(plot_value, alpha=0.7, **plotargs)
                if plot_mean_value:
                    pass
                else:
                    tmp_plotargs = {i: plotargs[i] for i in plotargs}
                    del tmp_plotargs['label']
                    ax.fill_between(range(num_epoch), fill_min, fill_max, alpha=0.1, **tmp_plotargs)
                
                if key_without_train_test not in y_min_max:
                    y_min_max[key_without_train_test] = [data.min(), data.max()]
                else:
                    if y_min_max[key_without_train_test][0] > data.min():
                        y_min_max[key_without_train_test][0] = data.min()
                    if y_min_max[key_without_train_test][1] < data.max():
                        y_min_max[key_without_train_test][1] = data.max()   
        else:
            for data_key in scalar:
                key_without_train_test = re.sub('train|test|val', '', data_key)
                if data_key not in figs_num:
                    fig = plt.figure(figsize=figsize)
                    figs_num[data_key] = fig.number
                    ax = fig.add_subplot(1, 1, 1)
                    ax.set_title(data_key)
                else:
                    fig = plt.figure(figs_num[data_key])
                    ax = fig.axes[0]
                data = scalar[data_key]
                if 'train' in data_key:
                    print(file_key, data_key)
                    data = data.reshape(num_epoch, -1).mean(1)
                if 'label' not in plotargs:
                    plotargs["label"] = file_key
                ax.plot(data, alpha=0.7, **plotargs)
                
                if key_without_train_test not in y_min_max:
                    y_min_max[key_without_train_test] = [data.min(), data.max()]
                else:
                    if y_min_max[key_without_train_test][0] > data.min():
                        y_min_max[key_without_train_test][0] = data.min()
                    if y_min_max[key_without_train_test][1] < data.max():
                        y_min_max[key_without_train_test][1] = data.max()
                

    for key in figs_num:
        print(key)
        key_without_train_test = re.sub('train|test|val', '', key)
        fig = plt.figure(figs_num[key])
        ax = fig.axes[0]
        e0 = 1
        e1 = 1
        if 'loss' in key:
            ax.set_yscale('log')
            e0 = 1e-10
            e1 = y_min_max[key_without_train_test][1]
        ax.set_ylim(y_min_max[key_without_train_test][0]-e0, y_min_max[key_without_train_test][1]+e1)
        ax.grid()
        ax.legend(fontsize=int(fontsize * 3 / 5), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

        if plot_mean_value:
            out_path = os.path.join(out_dir, 'mean-{}.png'.format(key.replace('/', '-')))
        else:
            out_path = os.path.join(out_dir, '{}.png'.format(key.replace('/', '-')))
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

def train_test_summary(scalar_data, out_dir, kfold=-1, adjust_y_scale=True, plot_mean_value=False, **kwargs):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fontsize = 24
    plt.rcParams["font.size"] = fontsize
    num_epoch = 100
    figsize = (16, 9)

    figs_num = {}
    y_min_max = {}
    file_keys = sorted(list(scalar_data.keys()))
    for file_key in file_keys:
        scalar = scalar_data[file_key]
        plotargs = get_plotargs(file_key)
        if 'label' not in plotargs:
            plotargs["label"] = file_key
        label_name = plotargs["label"]

        if kfold > 0:
            s_keys = scalar.keys()
            set_keys = set(map(lambda x: x[x.find('/') + 1:], s_keys))
            dict_keys = {i: list(filter(lambda x: fnmatch(x, '*/{}'.format(i)), s_keys)) for i in set_keys}
            for data_key in dict_keys:
                cv_keys = dict_keys[data_key]
                key_without_train_test = re.sub('train|test|val', '', data_key)
                fig_key = '/'.join([label_name, key_without_train_test])
                if fig_key not in figs_num:
                    fig = plt.figure(figsize=figsize)
                    figs_num[fig_key] = fig.number
                    ax = fig.add_subplot(1, 1, 1)
                    ax.set_title(fig_key)
                else:
                    fig = plt.figure(figs_num[fig_key])
                    ax = fig.axes[0]
                data = []
                for cv_key in cv_keys:
                    data.append(scalar[cv_key])
                data = np.asarray(data)

                if 'train' in data_key:
                    data = data.reshape(kfold, num_epoch, -1).mean(-1)
                    plotargs['linestyle'] = ':'
                    plotargs['label'] = 'train'
                if 'test' in data_key:
                    plotargs['linestyle'] = '-'
                    plotargs['label'] = 'test'
                
                if plot_mean_value:
                    plot_value = data.mean(0)
                else:
                    q75, q25, plot_value = np.percentile(data, [75, 25, 50], axis=0)
                    fill_min = q25
                    fill_max = q75
                                    
                ax.plot(plot_value, alpha=0.7, **plotargs)
                if plot_mean_value:
                    pass
                else:
                    tmp_plotargs = {i: plotargs[i] for i in plotargs}
                    del tmp_plotargs['label']
                    ax.fill_between(range(num_epoch), fill_min, fill_max, alpha=0.1, **tmp_plotargs)
                
                if key_without_train_test not in y_min_max:
                    y_min_max[key_without_train_test] = [data.min(), data.max()]
                else:
                    if y_min_max[key_without_train_test][0] > data.min():
                        y_min_max[key_without_train_test][0] = data.min()
                    if y_min_max[key_without_train_test][1] < data.max():
                        y_min_max[key_without_train_test][1] = data.max()   
        else:
            for data_key in scalar:
                key_without_train_test = re.sub('/train|/test|/val', '', data_key)
                fig_key = '/'.join([label_name, key_without_train_test])
                if fig_key not in figs_num:
                    fig = plt.figure(figsize=figsize)
                    figs_num[fig_key] = fig.number
                    ax = fig.add_subplot(1, 1, 1)
                    ax.set_title(fig_key)
                else:
                    fig = plt.figure(figs_num[fig_key])
                    ax = fig.axes[0]
                data = scalar[data_key]
                if 'train' in data_key:
                    data = data.reshape(num_epoch, -1).mean(1)
                    plotargs['linestyle'] = ':'
                    plotargs['label'] = 'train'
                if 'test' in data_key:
                    plotargs['linestyle'] = '-'
                    plotargs['label'] = 'test'
                
                ax.plot(data, alpha=0.7, **plotargs)
                
                if key_without_train_test not in y_min_max:
                    y_min_max[key_without_train_test] = [data.min(), data.max()]
                else:
                    if y_min_max[key_without_train_test][0] > data.min():
                        y_min_max[key_without_train_test][0] = data.min()
                    if y_min_max[key_without_train_test][1] < data.max():
                        y_min_max[key_without_train_test][1] = data.max()
                

    y_keys = list(y_min_max.keys())
    for key in figs_num:
        print(key)
        key_without_train_test = list(filter(lambda x: x in key, y_keys))[0]
        fig = plt.figure(figs_num[key])
        ax = fig.axes[0]
        e0 = 1
        e1 = 1
        if 'loss' in key:
            ax.set_yscale('log')
            e0 = 1e-10
            e1 = y_min_max[key_without_train_test][1]
        if adjust_y_scale:
            ax.set_ylim(y_min_max[key_without_train_test][0]-e0, y_min_max[key_without_train_test][1]+e1)
        ax.grid()
        ax.legend(fontsize=int(fontsize * 3 / 5), bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

        if plot_mean_value:
            out_path = os.path.join(out_dir, 'mean-{}.png'.format(key.replace('/', '-')))
        else:
            out_path = os.path.join(out_dir, '{}.png'.format(key.replace('/', '-')))
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

def plot_quantile(scalar_data, out_dir, kfold, epoch=100):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fontsize = 24
    plt.rcParams["font.size"] = fontsize
    num_epoch = 100
    figsize = (16, 9)

    figs_num = {}
    y_min_max = {}

    def rm_key(key):
        if 'k4-2' in key:
            return True
        
        if 'k4' in key:
            return False
        
        return True

    file_keys = list(filter(rm_key, scalar_data.keys()))
    file_keys = sorted(file_keys, key=lambda x: parse_ksize(x))
    box_data = {}
    for file_key in file_keys:
        scalar = scalar_data[file_key]
        plotargs = get_plotargs(file_key)
        plotargs = get_plotargs(file_key)
        if 'label' not in plotargs:
            plotargs["label"] = file_key
        label_name = plotargs["label"]
        
        assert kfold > 0
        s_keys = scalar.keys()
        set_keys = set(map(lambda x: x[x.find('/') + 1:], s_keys))
        dict_keys = {i: list(filter(lambda x: fnmatch(x, '*/{}'.format(i)), s_keys)) for i in set_keys}
        x_cnt = 0
        for data_key in dict_keys:
            cv_keys = dict_keys[data_key]

            key_without_train_test = re.sub('train|test|val', '', data_key)
            fig_key = '/'.join([label_name, key_without_train_test])

            if data_key not in figs_num:
                fig = plt.figure(figsize=figsize)
                figs_num[data_key] = fig.number
                ax = fig.add_subplot(1, 1, 1)
                ax.set_title(data_key)
                box_data[data_key] = []
            else:
                fig = plt.figure(figs_num[data_key])
                ax = fig.axes[0]
            data = []
            for cv_key in cv_keys:
                data.append(scalar[cv_key])
            data = np.asarray(data)

            if 'train' in data_key:
                print(file_key, data_key)
                data = data.reshape(kfold, num_epoch, -1).mean(-1)
            if 'label' not in plotargs:
                plotargs["label"] = file_key
            
            data = data[:, epoch-1]

            box_data[data_key].append({
                'data': data,
                'label': parse_ksize(file_key),
            })

            if key_without_train_test not in y_min_max:
                y_min_max[key_without_train_test] = [data.min(), data.max()]
            else:
                if y_min_max[key_without_train_test][0] > data.min():
                    y_min_max[key_without_train_test][0] = data.min()
                if y_min_max[key_without_train_test][1] < data.max():
                    y_min_max[key_without_train_test][1] = data.max()
                

    for key in figs_num:
        print(key)
        key_without_train_test = re.sub('train|test|val', '', key)
        fig = plt.figure(figs_num[key])
        ax = fig.axes[0]
        e0 = 1
        e1 = 1
        if 'loss' in key:
            ax.set_yscale('log')
            e0 = 1e-10
            e1 = y_min_max[key_without_train_test][1]
            ax.set_ylabel('Loss value')
        if 'accuracy' in key:
            ax.set_ylabel('Accuracy rate')
        ax.set_ylim(y_min_max[key_without_train_test][0]-e0, y_min_max[key_without_train_test][1]+e1)
        datas = [d['data'] for d in box_data[key]]
        labels =[d['label'] for d in box_data[key]]
        ax.boxplot(datas, labels=labels)
        ax.set_xlabel('K')
        ax.grid()

        out_path = os.path.join(out_dir, '{}.png'.format(key.replace('/', '-')))
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

def update_cvcnt(dict_data, offset=5, verbose=False):
    keys = list(dict_data.keys())
    for key in keys:
        extracted = re.search('cv[0-9]+', key)
        if extracted:
            str_number = extracted.group(0)[2:]
            cnt = int(str_number) + offset
            update_key = re.sub('cv[0-9]+', 'cv' + str(cnt).zfill(len(str_number)), key)
            dict_data[update_key] = dict_data[key]
            if verbose:
                print('{}==>{}'.format(key, update_key))
            del dict_data[key]

def main(scalar_path, out_dir, kfold, **kwargs):

    if isinstance(scalar_path, list):
        scalar_data = {}
        for s_path in scalar_path:
            with open(s_path, 'rb') as f:
                scalar_data2 = pickle.load(f)
            keys2 = list(scalar_data2.keys())
            keys1 = list(scalar_data.keys())
            for key in keys2:
                if key in keys1:
                    update_cvcnt(scalar_data2[key], offset=kfold)
                    scalar_data[key].update(scalar_data2[key])
                else:
                    scalar_data[key] = scalar_data2[key]
        kfold = int(len(scalar_path) * kfold)
    elif isinstance(scalar_path, str):
        with open(scalar_path, 'rb') as f:
            scalar_data = pickle.load(f)

    # global summary
    global_summary(scalar_data, os.path.join(out_dir, 'global'), kfold)
    global_summary(scalar_data, os.path.join(out_dir, 'global'), kfold, plot_mean_value=True)

    # train-test summary
    train_test_summary(scalar_data, os.path.join(out_dir, 'train-vs-test'), kfold, **kwargs)
    train_test_summary(scalar_data, os.path.join(out_dir, 'train-vs-test'), kfold, plot_mean_value=True, **kwargs)

    # boxplot
    if kfold > 0:
        plot_quantile(scalar_data, os.path.join(out_dir, 'boxplot'), kfold)

    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Plot some graphes')

    parser.add_argument('-s', '--scalars', type=str, nargs='+',
        help='name of the scalar data')
    parser.add_argument('--output-folder', type=str, default=None,
        help='name of the output folder')
    parser.add_argument('--adjust-y-scale', action='store_true',
        help='adjust y scale over same kind data')
    parser.add_argument('--kfold', type=int, default=-1,
        help='if kfold > 0 then do k-fold Cross Validation. if < 1 then do not use. (default: -1)')
    args = parser.parse_args()

    if args.output_folder is None:
        dirname = os.path.dirname(args.scalars[0])
        while os.path.basename(dirname) != 'datas':
            dirname = os.path.dirname(dirname)
        dirname = os.path.dirname(dirname)
        dir_path = dirname

        if args.adjust_y_scale:
            args.output_folder = os.path.join(dir_path, 'adjust_y_imgs')
        else:
            args.output_folder = os.path.join(dir_path, 'imgs')
    
    print(args.output_folder)
    main(args.scalars, args.output_folder, args.kfold,
        adjust_y_scale=args.adjust_y_scale)
