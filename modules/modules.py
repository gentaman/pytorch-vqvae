import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.uniform import Uniform
from torch.distributions import kl_divergence

from .functions import vq, vq_st

def to_scalar(arr):
    if type(arr) == list:
        return [x.item() for x in arr]
    else:
        return arr.item()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
        except AttributeError:
            print("Skipping weight initialization of ", classname)

        try:
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping bias initialization of ", classname)

class AE(nn.Module):
    def __init__(self, input_dim, dim, pred=False, transpose=False, BN=True):
        super().__init__()

        connections = [
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim, BN=BN),
            ResBlock(dim, BN=BN),
        ]
        if not BN:
            connections = list(filter(lambda x: not isinstance(x, nn.BatchNorm2d), connections))
        self.encoder = nn.Sequential(*connections)

        connections = [
            ResBlock(dim, transpose, BN=BN),
            ResBlock(dim, transpose, BN=BN),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()]
        if not BN:
            connections = list(filter(lambda x: not isinstance(x, nn.BatchNorm2d), connections))
        self.decoder = nn.Sequential(*connections)

        self.apply(weights_init)
        self.pred = pred

    def forward(self, x):
        h = self.encoder(x)
        x_tilde = self.decoder(h)
        if self.pred:
            return x_tilde, h
        else:
            return x_tilde,

class VAE(nn.Module):
    def __init__(self, input_dim, dim, dist=Normal, pred=False, transpose=False, BN=True):
        super().__init__()

        connections = [
            nn.Conv2d(input_dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1),
            ResBlock(dim, BN=BN),
            ResBlock(dim, BN=BN),
        ]
        if not BN:
            connections = list(filter(lambda x: not isinstance(x, nn.BatchNorm2d), connections))
        self.encoder = nn.Sequential(*connections)

        connections = [
            ResBlock(dim, transpose, BN=BN),
            ResBlock(dim, transpose, BN=BN),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1),
            nn.Tanh()]
        if not BN:
            connections = list(filter(lambda x: not isinstance(x, nn.BatchNorm2d), connections))
        self.decoder = nn.Sequential(*connections)

        self.apply(weights_init)
        self.pred = pred
        if isinstance(dist, str):
            if dist == 'uniform':
                self.dist = lambda s, l: Uniform(-s + l, s + l)
            if dist == 'normal':
                self.dist = Normal
        else:
            self.dist = dist

    def forward(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)

        q_z_x = self.dist(mu, logvar.mul(.5).exp())
        p_z = self.dist(torch.zeros_like(mu), torch.ones_like(logvar))
        kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()
        sample = q_z_x.rsample()
        x_tilde = self.decoder(sample)
        if self.pred:
            return x_tilde, kl_div, sample
        else:
            return x_tilde, kl_div


class VQEmbedding(nn.Module):
    def __init__(self, K, D, ema=False, gamma=0.99):
        super().__init__()
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1. / K, 1. / K)
        self.ema = ema
        self.N = 0.0
        self.M = 0.0
        self.gamma = gamma
        self.index = None

    def forward(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        latents = vq(z_e_x_, self.embedding.weight)
        return latents

    def straight_through(self, z_e_x):
        z_e_x_ = z_e_x.permute(0, 2, 3, 1).contiguous()
        z_q_x_, indices = vq_st(z_e_x_, self.embedding.weight.detach())
        z_q_x = z_q_x_.permute(0, 3, 1, 2).contiguous()

        self.index = indices

        z_q_x_bar_flatten = torch.index_select(self.embedding.weight,
            dim=0, index=indices)
        z_q_x_bar_ = z_q_x_bar_flatten.view_as(z_e_x_)
        z_q_x_bar = z_q_x_bar_.permute(0, 3, 1, 2).contiguous()

        return z_q_x, z_q_x_bar

    def update(self, n, z_e_x, device='cpu'):
        if self.ema:
            with torch.no_grad():
                batch, ch, h, w = z_e_x.shape
                z_e = z_e_x.permute(0, 2, 3, 1).contiguous().view(-1, ch)
                K = self.embedding.weight.shape[0]
                onehots = torch.zeros(int(batch * h * w), K, device=device)
                onehots.scatter_(1, self.index.unsqueeze(1), 1)
                sum_z = torch.mm(z_e.transpose(1, 0), onehots)
                self.N = self.N * self.gamma + n * (1 - self.gamma)
                self.M = self.M * self.gamma + sum_z * (1 - self.gamma)
                self.embedding.weight.data = (self.M / self.N).transpose(1, 0)
        else:
            raise ValueError('ema is False')


class ResBlock(nn.Module):
    def __init__(self, dim, transpose=False, BN=True, bias=True):
        super().__init__()
        self.transpose = transpose
        if self.transpose:
            connections = [
                nn.ReLU(True),
                nn.ConvTranspose2d(dim, dim, 1, bias=bias),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.ConvTranspose2d(dim, dim, 3, 1, bias=bias),
                nn.BatchNorm2d(dim)
            ]
        else:
            connections = [
                nn.ReLU(True),
                nn.Conv2d(dim, dim, 3, 1, 1, bias=bias),
                nn.BatchNorm2d(dim),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, 1, bias=bias),
                nn.BatchNorm2d(dim)
            ]
        if not BN:
            connections = list(filter(lambda x: not isinstance(x, nn.BatchNorm2d), connections))

        self.block = nn.Sequential(*connections)

    def forward(self, x):
        h = x
        cnt = 0
        for b in self.block:
            if self.transpose and isinstance(b, nn.ConvTranspose2d):
                # h = b(h, output_size=x.size())
                h = b(h)
                if cnt == 1:
                    h = h[:, :, 0:x.size(2), 0:x.size(3)]
                cnt += 1
            else:
                h = b(h)
        return x + h


class VectorQuantizedVAE(nn.Module):
    def __init__(self, input_dim, dim, K=512, pred=False, transpose=False, BN=True, bias=True, ema=False):
        super().__init__()
        self.pred = pred

        self.codebook = VQEmbedding(K, dim, ema)

        connections = [
            nn.Conv2d(input_dim, dim, 4, 2, 1, bias=bias),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, dim, 4, 2, 1, bias=bias),
            ResBlock(dim, BN=BN, bias=bias),
            ResBlock(dim, BN=BN, bias=bias),
        ]
        if not BN:
            connections = list(filter(lambda x: not isinstance(x, nn.BatchNorm2d), connections))
        self.encoder = nn.Sequential(*connections)

        connections = [
            ResBlock(dim, transpose, BN=BN, bias=bias),
            ResBlock(dim, transpose, BN=BN, bias=bias),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, dim, 4, 2, 1, bias=bias),
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.ConvTranspose2d(dim, input_dim, 4, 2, 1, bias=bias),
            nn.Tanh()]
        if not BN:
            connections = list(filter(lambda x: not isinstance(x, nn.BatchNorm2d), connections))
        self.decoder = nn.Sequential(*connections)

        self.apply(weights_init)

    def encode(self, x):
        z_e_x = self.encoder(x)
        latents = self.codebook(z_e_x)
        return latents

    def decode(self, latents):
        z_q_x = self.codebook.embedding(latents).permute(0, 3, 1, 2)  # (B, D, H, W)
        x_tilde = self.decoder(z_q_x)
        return x_tilde

    def forward(self, x):
        z_e_x = self.encoder(x)
        z_q_x_st, z_q_x = self.codebook.straight_through(z_e_x)
        x_tilde = self.decoder(z_q_x_st)
        if self.pred:
            return x_tilde, z_e_x, z_q_x, z_q_x_st
        else:
            return x_tilde, z_e_x, z_q_x


class GatedActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x, y = x.chunk(2, dim=1)
        return F.tanh(x) * F.sigmoid(y)


class GatedMaskedConv2d(nn.Module):
    def __init__(self, mask_type, dim, kernel, residual=True, n_classes=10):
        super().__init__()
        assert kernel % 2 == 1, print("Kernel size must be odd")
        self.mask_type = mask_type
        self.residual = residual

        self.class_cond_embedding = nn.Embedding(
            n_classes, 2 * dim
        )

        kernel_shp = (kernel // 2 + 1, kernel)  # (ceil(n/2), n)
        padding_shp = (kernel // 2, kernel // 2)
        self.vert_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.vert_to_horiz = nn.Conv2d(2 * dim, 2 * dim, 1)

        kernel_shp = (1, kernel // 2 + 1)
        padding_shp = (0, kernel // 2)
        self.horiz_stack = nn.Conv2d(
            dim, dim * 2,
            kernel_shp, 1, padding_shp
        )

        self.horiz_resid = nn.Conv2d(dim, dim, 1)

        self.gate = GatedActivation()

    def make_causal(self):
        self.vert_stack.weight.data[:, :, -1].zero_()  # Mask final row
        self.horiz_stack.weight.data[:, :, :, -1].zero_()  # Mask final column

    def forward(self, x_v, x_h, h):
        if self.mask_type == 'A':
            self.make_causal()

        h = self.class_cond_embedding(h)
        h_vert = self.vert_stack(x_v)
        h_vert = h_vert[:, :, :x_v.size(-1), :]
        out_v = self.gate(h_vert + h[:, :, None, None])

        h_horiz = self.horiz_stack(x_h)
        h_horiz = h_horiz[:, :, :, :x_h.size(-2)]
        v2h = self.vert_to_horiz(h_vert)

        out = self.gate(v2h + h_horiz + h[:, :, None, None])
        if self.residual:
            out_h = self.horiz_resid(out) + x_h
        else:
            out_h = self.horiz_resid(out)

        return out_v, out_h


class GatedPixelCNN(nn.Module):
    def __init__(self, input_dim=256, dim=64, n_layers=15, n_classes=10):
        super().__init__()
        self.dim = dim

        # Create embedding layer to embed input
        self.embedding = nn.Embedding(input_dim, dim)

        # Building the PixelCNN layer by layer
        self.layers = nn.ModuleList()

        # Initial block with Mask-A convolution
        # Rest with Mask-B convolutions
        for i in range(n_layers):
            mask_type = 'A' if i == 0 else 'B'
            kernel = 7 if i == 0 else 3
            residual = False if i == 0 else True

            self.layers.append(
                GatedMaskedConv2d(mask_type, dim, kernel, residual, n_classes)
            )

        # Add the output layer
        self.output_conv = nn.Sequential(
            nn.Conv2d(dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, input_dim, 1)
        )

        self.apply(weights_init)

    def forward(self, x, label):
        shp = x.size() + (-1, )
        x = self.embedding(x.view(-1)).view(shp)  # (B, H, W, C)
        x = x.permute(0, 3, 1, 2)  # (B, C, W, W)

        x_v, x_h = (x, x)
        for i, layer in enumerate(self.layers):
            x_v, x_h = layer(x_v, x_h, label)

        return self.output_conv(x_h)

    def generate(self, label, shape=(8, 8), batch_size=64):
        param = next(self.parameters())
        x = torch.zeros(
            (batch_size, *shape),
            dtype=torch.int64, device=param.device
        )

        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, label)
                probs = F.softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(
                    probs.multinomial(1).squeeze().data
                )
        return x
