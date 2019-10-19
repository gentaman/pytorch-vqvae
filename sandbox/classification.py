
# coding: utf-8

# In[1]:

import sys
sys.path.append('../')


# In[2]:

from modules.modules import VectorQuantizedVAE


# In[3]:

import numpy as np
import torch


# In[4]:

# get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
from datasets import datasets


# In[5]:

import torch.optim as optim

from tqdm import tqdm as tqdm


# In[6]:

from torch import nn
class Classifier(nn.Module):
    def __init__(self, in_f, out_f):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_f, out_f)
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return x


# In[7]:

model = VectorQuantizedVAE(3, 256, 256)


# In[8]:

model.load_state_dict(torch.load('/home/genta/data2/vqvae/models/vqvae_im128_k256/best.pt'))


# In[9]:

dataset = datasets.get_dataset('imagenet', '~/dataset/', image_size=128)


# In[10]:

train_dataset = dataset['train']
test_dataset = dataset['test']
valid_dataset = dataset['valid']
num_channels = dataset['num_channels']


# In[11]:

batch_size = 256
num_workers = 2
train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
    batch_size=batch_size, shuffle=False, drop_last=True,
    num_workers=num_workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_dataset,
    batch_size=16, shuffle=True)


# In[12]:

predictor = Classifier(int(256*32*32), len(train_dataset.classes))


# In[13]:

predictor.cuda()
model.cuda()


# In[14]:

optimizer = optim.SGD(predictor.parameters(), lr=0.001, momentum=0.9)


# In[17]:

loss_fn = nn.CrossEntropyLoss()


def train(data_loader, model, clfy, optimizer, args=None, writer=None, loss_fn=None):
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    for images, labels in tqdm(data_loader, total=len(data_loader)):
        # print(images.shape)
        images = images.to('cuda')
        labels = labels.to('cuda')

        optimizer.zero_grad()
        with torch.no_grad():
            latents = model.encode(images)
            latents = model.codebook.embedding(latents).permute(0, 3, 1, 2)
        out = clfy(latents)
        loss = loss_fn(out, labels)
        loss.backward()

#         if writer is not None:
#             # Logs
#             writer.add_scalar('loss/train/reconstruction', loss_recons.item(), args.steps)
#             writer.add_scalar('loss/train/quantization', loss_vq.item(), args.steps)

        optimizer.step()
#         args.steps += 1


# In[ ]:
train(train_loader, model, predictor, optimizer, loss_fn=loss_fn)

torch.save(predictor.state_dict(), './clfy.model')
# get_ipython().run_cell_magic(u'time', u'', u'train(train_loader, model, predictor, optimizer, loss_fn=loss_fn)')
