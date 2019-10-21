import os
import csv
from glob import glob
import torch.utils.data as data
from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder
from PIL import Image

from .my_folder import MyImageFolder

PYTORCH_IMAGENET_MEAN = [0.485, 0.456, 0.406]
PYTORCH_IMAGENET_STD = [0.229, 0.224, 0.225]


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    # Borrowed from https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def get_dataset(dataset, data_folder, image_size=None):
    if dataset in ['mnist', 'fashion-mnist', 'cifar10']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        if dataset == 'mnist':
            # Define the train & test datasets
            if image_size is None:
                image_size = 28
            transform = transforms.Compose([
                # transforms.RandomResizedCrop((image_size, image_size)),
                transforms.Resize((image_size+ image_size//16, image_size + image_size//16)),
                transforms.RandomResizedCrop((image_size, image_size), scale=(0.8, 1.0)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
            train_dataset = datasets.MNIST(data_folder, train=True,
                download=True, transform=transform)
            test_dataset = datasets.MNIST(data_folder, train=False,
                transform=transform)
            num_channels = 1
            unique_labels = set(train_dataset.classes)
            train_dataset._label_encoder = dict((label, idx) for (idx, label) in enumerate(unique_labels))
            test_dataset._label_encoder = train_dataset._label_encoder
            valid_dataset = test_dataset
        elif dataset == 'fashion-mnist':
            # Define the train & test datasets
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5), (0.5))
            ])
            train_dataset = datasets.FashionMNIST(data_folder,
                train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(data_folder,
                train=False, transform=transform)
            num_channels = 1
        elif dataset == 'cifar10':
            # Define the train & test datasets
            train_dataset = datasets.CIFAR10(data_folder,
                train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(data_folder,
                train=False, transform=transform)
            num_channels = 3
        valid_dataset = test_dataset
    elif dataset == 'miniimagenet':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        # Define the train, valid & test datasets
        train_dataset = MiniImagenet(
                                os.path.join(data_folder, 'train'),
                                transform=transform
                                )
        valid_dataset = MiniImagenet(
                                os.path.join(data_folder, 'val'),
                                transform=transform
                                )
        test_dataset = MiniImagenet(
                                os.path.join(data_folder, 'test'),
                                transform=transform
                                )
        num_channels = 3
    elif dataset == 'imagenet':
        if image_size is None:
            image_size = 128
        
        normalize = transforms.Normalize(PYTORCH_IMAGENET_MEAN, PYTORCH_IMAGENET_STD)

        transform = transforms.Compose([
            transforms.RandomResizedCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
        # Define the train, valid & test datasets
        data_path = os.path.join(data_folder, 'ilsvrc2012')
        if not os.path.exists(data_path):
            data_path = os.path.join(data_folder, 'ILSVRC2012')
        if not os.path.exists(data_path):
            data_path = data_folder

        traindir = os.path.join(data_path, 'train')
        train_dataset = datasets.ImageFolder(traindir, transform)
        unique_labels = set(train_dataset.classes)
        train_dataset._label_encoder = dict((label, idx) for (idx, label) in enumerate(unique_labels))

        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize])
        
        valdir = os.path.join(data_path, 'val')
        txt_path = os.path.join(data_path, 'val.txt')
        dirs = glob(os.path.join(valdir, '**')+'/')
        if len(dirs) > 0:
            valid_dataset = datasets.ImageFolder(valdir, transform)
            unique_labels = set(valid_dataset.classes)
            valid_dataset._label_encoder = dict((label, idx) for (idx, label) in enumerate(unique_labels))
        else:
            valid_dataset = MyImageFolder(data_path, txt_path, transform)

        test_dataset = valid_dataset

        num_channels = 3
    
    elif dataset == 'kylberg':
        if image_size is None:
            image_size = 64

        kylberg_mean = (0.49794143, 0.49794143, 0.49794143)
        kylberg_std = (0.15645953, 0.15645953, 0.15645953)
        transform = transforms.Compose([
            transforms.Resize(image_size+32),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=30),
            transforms.ToTensor(),
            transforms.Normalize(kylberg_mean, kylberg_std)
        ])
        train_dataset = MyDataset(
                                os.path.join(data_folder, 'train'),
                                transform=transform
        )
        transform = transforms.Compose([
            transforms.Resize(image_size+32),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(kylberg_mean, kylberg_std)
        ])
        test_dataset = MyDataset(
                                os.path.join(data_folder, 'test'),
                                transform=transform
        )
        valid_dataset = test_dataset
        num_channels = 3
    result_dict = {
        'train': train_dataset,
        'test': test_dataset,
        'valid': valid_dataset,
        'num_channels': num_channels
    }

    return result_dict

class MyDataset(ImageFolder):
    def __init__(self, root, **kwargs):
        super(MyDataset, self).__init__(root, **kwargs)
        self._fit_label_encoding()

    def _fit_label_encoding(self):
        unique_labels = set(self.classes)
        self._label_encoder = dict((label, idx) for (idx, label) in enumerate(unique_labels))

class MiniImagenet(ImageFolder):
    def __init__(self, root, **kwargs):
        super(MiniImagenet, self).__init__(root, **kwargs)
        self._fit_label_encoding()

    def _fit_label_encoding(self):
        unique_labels = set(self.classes)
        self._label_encoder = dict((label, idx) for (idx, label) in enumerate(unique_labels))

class OldMiniImagenet(data.Dataset):

    base_folder = '~/dataset/miniimagenet'
    filename = 'miniimagenet.zip'
    splits = {
        'train': 'train.csv',
        'valid': 'val.csv',
        'test': 'test.csv'
    }

    def __init__(self, root, train=False, valid=False, test=False,
                 transform=None, target_transform=None, download=False):
        super(MiniImagenet, self).__init__()
        self.root = root
        self.train = train
        self.valid = valid
        self.test = test
        self.transform = transform
        self.target_transform = target_transform

        if not (((train ^ valid ^ test) ^ (train & valid & test))):
            raise ValueError('One and only one of `train`, `valid` or `test` '
                'must be True (train={0}, valid={1}, test={2}).'.format(train,
                valid, test))

        self.image_folder = os.path.join(os.path.expanduser(root), 'images')
        if train:
            split = self.splits['train']
        elif valid:
            split = self.splits['valid']
        elif test:
            split = self.splits['test']
        else:
            raise ValueError('Unknown split.')
        self.split_filename = os.path.join(os.path.expanduser(root), split)
        if download:
            self.download()
        if not self._check_exists():
            raise RuntimeError('Dataset not found. You can use `download=True` '
                               'to download it')

        # Extract filenames and labels
        self._data = []
        with open(self.split_filename, 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip the header
            for line in reader:
                self._data.append(tuple(line))
        self._fit_label_encoding()

    def __getitem__(self, index):
        filename, label = self._data[index]
        image = pil_loader(os.path.join(self.image_folder, filename))
        label = self._label_encoder[label]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image, label

    def _fit_label_encoding(self):
        _, labels = zip(*self._data)
        unique_labels = set(labels)
        self._label_encoder = dict((label, idx) for (idx, label) in enumerate(unique_labels))

    def _check_exists(self):
        return (os.path.exists(self.image_folder) 
            and os.path.exists(self.split_filename))

    def download(self):
        from shutil import copyfile
        from zipfile import ZipFile

        # If the image folder already exists, break
        if self._check_exists():
            return True

        # Create folder if it does not exist
        root = os.path.expanduser(self.root)
        if not os.path.exists(root):
            os.makedirs(root)

        # Copy the file to root
        path_source = os.path.join(self.base_folder, self.filename)
        path_dest = os.path.join(root, self.filename)
        print('Copy file `{0}` to `{1}`...'.format(path_source, path_dest))
        copyfile(path_source, path_dest)

        # Extract the dataset
        print('Extract files from `{0}`...'.format(path_dest))
        with ZipFile(path_dest, 'r') as f:
            f.extractall(root)

        # Copy CSV files
        for split in self.splits:
            path_source = os.path.join(self.base_folder, self.splits[split])
            path_dest = os.path.join(root, self.splits[split])
            print('Copy file `{0}` to `{1}`...'.format(path_source, path_dest))
            copyfile(path_source, path_dest)
        print('Done!')

    def __len__(self):
        return len(self._data)
