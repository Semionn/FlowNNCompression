from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from sklearn.datasets import make_moons
import matplotlib.pylab as plt
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F
import os
from math import log, sqrt, pi
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils
from pytorch_lightning import LightningModule, Trainer

logabs = lambda x: torch.log(torch.abs(x))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
b_size = 200


class ActNorm(LightningModule):
    def __init__(self, ndim, logdet=True):
        super().__init__()

        self.loc = nn.Parameter(torch.zeros(ndim, ))
        self.scale = nn.Parameter(torch.ones(ndim, ))

        self.register_buffer("initialized", torch.tensor(0, dtype=torch.uint8))
        self.logdet = logdet
        self.ndim = ndim

    def initialize(self, input):
        with torch.no_grad():
            flatten = input
            mean = flatten.mean(0) - 2
            std = flatten.std(0)

            # self.loc = -mean
            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input):
        b_size, input_dims = input.shape

        if self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        log_abs = logabs(self.scale)

        logdet = input_dims * torch.sum(log_abs)

        if self.logdet:
            return self.scale * (input + self.loc), logdet
        else:
            return self.scale * (input + self.loc)

    def reverse(self, output):
        return output / self.scale - self.loc


class AffineCoupling(LightningModule):
    def __init__(self, ndim, affine=True):
        super().__init__()

        self.affine = affine
        w1 = 10
        w2 = 10

        self.net = nn.Sequential(
            nn.Linear(ndim // 2, w1),
            nn.ReLU(inplace=True),
            # nn.Sigmoid(),
            nn.Linear(w1, w2),
            nn.ReLU(inplace=True),
            # nn.Sigmoid(),
            nn.Linear(w2, ndim // 2),
            nn.ReLU(inplace=True),
            # nn.Sigmoid(),
            nn.Linear(ndim // 2, 2),
            # nn.ReLU(inplace=True),
        )

        self.net[0].weight.data.normal_(0, 0.05)
        self.net[0].bias.data.zero_()
        self.net[2].weight.data.normal_(0, 0.05)
        self.net[2].bias.data.zero_()
        self.net[4].weight.data.normal_(0, 0.05)
        self.net[4].bias.data.zero_()
        self.net[6].weight.data.zero_()
        self.net[6].bias.data.zero_()
        # self.net[0].weight.data.zero_()
        # self.net[0].bias.data.zero_()

    def forward(self, input):
        in_a, in_b = input.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(in_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = torch.sigmoid(log_s + 2)
            # out_a = s * in_a + t
            out_b = (in_b + t) * s

            logdet = torch.sum(torch.sum(torch.log(s).view(input.shape[0], -1), 1))

        else:
            net_out = self.net(in_a)
            out_b = in_b + net_out
            logdet = None

        return torch.cat([in_a, out_b], 1), logdet

    def reverse(self, output):
        out_a, out_b = output.chunk(2, 1)

        if self.affine:
            log_s, t = self.net(out_a).chunk(2, 1)
            # s = torch.exp(log_s)
            s = torch.sigmoid(log_s + 2)
            # in_a = (out_a - t) / s
            in_b = out_b / s - t

        else:
            net_out = self.net(out_a)
            in_b = out_b - net_out

        return torch.cat([out_a, in_b], 1)


class InvConv1d(LightningModule):  # TODO:
    def __init__(self, in_channel):
        super().__init__()

        weight = torch.randn(in_channel, in_channel)
        q, _ = torch.qr(weight)
        weight = q.unsqueeze(2)
        self.weight = nn.Parameter(weight)

    def forward(self, input):
        _, ndim = input.shape

        out = F.conv1d(input.unsqueeze(2), self.weight).squeeze()
        logdet = (
                ndim * torch.slogdet(self.weight.squeeze().double())[1].float()
        )  # TODO: logdet doesn't work properly?

        return out, logdet

    def reverse(self, output):
        return F.conv1d(
            output.unsqueeze(2), self.weight.squeeze().inverse().unsqueeze(2)
        ).squeeze()


def gaussian_log_p(x, mean, log_sd):
    return -0.5 * log(2 * pi) - log_sd - 0.5 * (x - mean) ** 2 / torch.exp(2 * log_sd)


def gaussian_sample(eps, mean, log_sd):
    return mean + torch.exp(log_sd) * eps


class LinearPrior(LightningModule):
    def __init__(self, ndim):
        super().__init__()
        self.fc = nn.Linear(ndim, 2)
        self.fc.weight.data.zero_()
        self.fc.bias.data.zero_()

    def forward(self, input):
        return self.fc(input)


class MoonModelBLock(LightningModule):
    def __init__(
            self, ndim, split
    ):
        super().__init__()
        if ndim < 2:
            self.actnorm = ActNorm(ndim)
        else:
            self.actnorm = ActNorm(ndim)
            self.invconv = InvConv1d(ndim)
            self.coupling = AffineCoupling(ndim)
            self.actnorm2 = ActNorm(ndim)
            self.invconv2 = InvConv1d(ndim)
            self.coupling2 = AffineCoupling(ndim)
            self.actnorm3 = ActNorm(ndim)
            self.invconv3 = InvConv1d(ndim)
            self.coupling3 = AffineCoupling(ndim)
            self.actnorm4 = ActNorm(ndim)
            self.invconv4 = InvConv1d(ndim)
            self.coupling4 = AffineCoupling(ndim)
        if split:
            self.prior = LinearPrior(ndim // 2)
        else:
            self.prior = LinearPrior(ndim)
        self.ndim = ndim
        self.split = split

    def forward(self, input):
        b_size, _ = input.shape
        # out, det = self.coupling(input.float())
        # logdet = det
        if self.ndim < 2:
            out, det = self.actnorm(input)
            logdet = det
        else:
            out, det = self.actnorm(input)
            logdet = det
            out, det = self.invconv(out.float())
            logdet += det
            out, det = self.coupling(out.float())
            logdet += det
            out, det = self.actnorm2(out.float())
            logdet += det
            out, det = self.invconv(out.float())
            logdet += det
            out, det = self.coupling2(out.float())
            logdet += det
            out, det = self.actnorm3(out.float())
            logdet += det
            out, det = self.coupling3(out.float())
            logdet += det
            out, det = self.actnorm4(out.float())
            logdet += det
            out, det = self.coupling4(out.float())
            logdet += det

        # zero = torch.zeros_like(out).float()
        # mean, log_sd = self.prior(zero).chunk(2, 1)
        # log_p = gaussian_log_p(out, mean, log_sd)
        # log_p = gaussian_log_p(out, torch.zeros(1,).to(device), torch.zeros(1,).to(device))
        # log_p = log_p.view(b_size, -1).sum(1)
        # z_new = out

        if self.split:
            out, z_new = out.chunk(2, 1)
            mean, log_sd = self.prior(out).chunk(2, 1)
            # log_p = gaussian_log_p(z_new, mean, log_sd)
            log_p = gaussian_log_p(z_new, torch.zeros_like(mean), torch.zeros_like(log_sd))
            log_p = log_p.view(b_size, -1).sum(1)

        else:
            zero = torch.zeros_like(out)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            # log_p = gaussian_log_p(out, mean, log_sd)
            log_p = gaussian_log_p(out, torch.zeros_like(mean), torch.zeros_like(log_sd))
            log_p = log_p.view(b_size, -1).sum(1)
            z_new = out

        return out, logdet, log_p, z_new

    def reverse(self, output, eps=None):
        input = output

        if self.split:
            mean, log_sd = self.prior(input).chunk(2, 1)
            z = gaussian_sample(eps, mean, log_sd)
            input = torch.cat([output, z], 1)

        else:
            zero = torch.zeros_like(input)
            mean, log_sd = self.prior(zero).chunk(2, 1)
            z = gaussian_sample(eps, mean, log_sd)
            input = z

        if self.ndim < 2:
            input = self.actnorm.reverse(input)
        else:
            input = self.coupling4.reverse(input)
            input = self.actnorm4.reverse(input)
            input = self.coupling3.reverse(input)
            input = self.actnorm3.reverse(input)
            input = self.coupling2.reverse(input)
            input = self.invconv2.reverse(input)
            input = self.actnorm2.reverse(input)
            input = self.coupling.reverse(input)
            input = self.invconv.reverse(input)
            input = self.actnorm.reverse(input)
        return input


class MoonModelMLP(LightningModule):
    def __init__(
            self, ndim, X,
    ):
        super().__init__()
        self.ndim = ndim
        self.prior = LinearPrior(ndim)
        self.X = X

        temp = 0.5
        n_sample = 1000
        z_sample = torch.randn(n_sample, 4) * temp

        self.z_sample = z_sample
        w1 = 2
        w2 = 2

        self.net = nn.Sequential(
            nn.Linear(ndim // 2, w1),
            nn.ReLU(inplace=True),
            # nn.Sigmoid(),
            nn.Linear(w1, w2),
            nn.ReLU(inplace=True),
            # nn.Sigmoid(),
            nn.Linear(w2, ndim // 2),
            nn.ReLU(inplace=True),
            # nn.Sigmoid(),
            nn.Linear(ndim // 2, 2),
            # nn.ReLU(inplace=True),
        )

    def forward(self, input):
        in_a, in_b = input.float().chunk(2, 1)

        log_s, t = self.net(in_a).chunk(2, 1)
        # s = torch.exp(log_s)
        s = torch.sigmoid(log_s + 2)
        # out_a = s * in_a + t
        out_b = (in_b + t) * s

        logdet = torch.sum(torch.sum(torch.log(s).view(input.shape[0], -1), 1))
        z = torch.cat([in_a, out_b], 1)

        zero = torch.zeros_like(z)
        mean, log_sd = self.prior(zero).chunk(2, 1)
        # log_p = gaussian_log_p(z, mean, log_sd)
        # TODO: do we need not standart gauss parameters?
        log_p = gaussian_log_p(z, torch.zeros_like(mean), torch.zeros_like(log_sd))
        # log_p = log_p.view(b_size, -1).sum(1)
        log_p = log_p.sum(1)

        return z, logdet, log_p

    def reverse(self, z):
        out_a, out_b = z.chunk(2, 1)

        log_s, t = self.net(out_a).chunk(2, 1)
        # s = torch.exp(log_s)
        s = torch.sigmoid(log_s + 2)
        # in_a = (out_a - t) / s
        in_b = out_b / s - t

        return torch.cat([out_a, in_b], 1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        z, logdet, log_p = y_hat
        loss, log_p, log_det = calc_loss(log_p, logdet)
        # loss.backward()
        # print(log_p)
        # print(self.net._modules['0'].weight)
        print(self.net._modules['0'].weight.grad)
        self.log("train_log_p", log_p, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_log_det", log_det, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx == 0:
            test_data = self.reverse(self.z_sample).cpu().data
            fig, ax = plt.subplots(figsize=(6, 6))
            plt.scatter(self.X[:1000, 0], self.X[:1000, 1], alpha=0.2)
            plt.scatter(test_data[:, 0], test_data[:, 1], alpha=0.1)
            ax.set_xlim(-0.75, 0.75)
            ax.set_ylim(0.1, 0.9)
            plt.savefig(f'moons_{batch_idx}')
            plt.show()

            z_detached = z.detach().numpy()
            fig, ax = plt.subplots(figsize=(6, 6))
            plt.scatter(z_detached[:, 0], z_detached[:, 1], alpha=0.1)
            # ax.set_xlim(-0.75, 0.75)
            # ax.set_ylim(-0.75, 0.75)
            plt.show()
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        z, logdet, log_p = y_hat
        loss, log_p, log_det = calc_loss(log_p, logdet)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        z, logdet, log_p = y_hat
        loss, log_p, log_det = calc_loss(log_p, logdet)
        return loss


class MoonModelMLPNotFlow(LightningModule):
    def __init__(
            self, ndim, X,
    ):
        super().__init__()
        self.ndim = ndim
        self.prior = LinearPrior(ndim)
        self.X = X

        w1 = 2
        w2 = 2

        self.net = nn.Sequential(
            nn.Linear(ndim, w1),
            nn.ReLU(inplace=True),
            # nn.Sigmoid(),
            nn.Linear(w1, w2),
            nn.ReLU(inplace=True),
            # nn.Sigmoid(),
            nn.Linear(w2, ndim),
            # nn.ReLU(inplace=True),
        )

    def forward(self, input):
        z = self.net(input.float())

        logdet = torch.zeros_like(z)

        zero = torch.zeros_like(z)
        mean, log_sd = self.prior(zero).chunk(2, 1)
        # log_p = gaussian_log_p(z, mean, log_sd)
        # TODO: do we need not standart gauss parameters?
        log_p = gaussian_log_p(z, torch.zeros_like(mean), torch.zeros_like(log_sd))
        # log_p = log_p.view(b_size, -1).sum(1)
        log_p = log_p.sum(1)

        return z, logdet, log_p

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-1)
        return optimizer

    def training_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        z, logdet, log_p = y_hat
        loss, log_p, log_det = calc_loss(log_p, logdet)
        # loss.backward()
        # print(log_p)
        print(self.net._modules['0'].weight)
        print(self.net._modules['0'].weight.grad)
        self.log("train_log_p", log_p, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx == 0:
            z = z.detach().numpy()
            fig, ax = plt.subplots(figsize=(6, 6))
            plt.scatter(z[:, 0], z[:, 1], alpha=0.1)
            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(-0.5, 0.5)
            plt.show()
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        z, logdet, log_p = y_hat
        loss, log_p, log_det = calc_loss(log_p, logdet)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        z, logdet, log_p = y_hat
        loss, log_p, log_det = calc_loss(log_p, logdet)
        return loss



class MoonModel(LightningModule):
    def __init__(
            self, ndim, X,
    ):
        super().__init__()
        blocks_count = 1 + int(np.log2(ndim))
        self.blocks = nn.ModuleList()
        block_ndim = ndim
        for i in range(blocks_count - 1):
            self.blocks.append(MoonModelBLock(block_ndim, split=True))
            block_ndim //= 2
        self.blocks.append(MoonModelBLock(block_ndim, split=False))
        self.ndim = ndim
        self.X = X

        z_sample = []
        temp = 0.5
        n_sample = 1000
        z_new = torch.randn(n_sample, 2) * temp
        z_sample.append(z_new.to(device))
        z_new = torch.randn(n_sample, 1) * temp
        z_sample.append(z_new.to(device))
        z_new = torch.randn(n_sample, 1) * temp
        z_sample.append(z_new.to(device))

        self.z_sample = z_sample

    def forward(self, input):
        log_p_sum = 0
        logdet = 0
        out = input
        z_outs = []

        for block in self.blocks:
            out, det, log_p, z_new = block(out)
            z_outs.append(z_new)
            logdet = logdet + det

            if log_p is not None:
                log_p_sum = log_p_sum + log_p

        return log_p_sum, logdet, z_outs

    def reverse(self, z_list):
        for i, block in enumerate(self.blocks[::-1]):
            if i == 0:
                input = block.reverse(z_list[-1], z_list[-1])

            else:
                input = block.reverse(input, z_list[-(i + 1)])

        return input

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        log_p, logdet, z = y_hat
        loss, log_p, log_det = calc_loss(log_p, logdet)
        self.log("train_log_p", log_p, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("reconstr_error", torch.mean(self.reverse(z) - batch), on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_log_det", log_det, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        if batch_idx == 0:
            test_data = self.reverse(z).cpu().data
            # test_data = self.reverse(self.z_sample).cpu().data
            fig, ax = plt.subplots(figsize=(6, 6))
            plt.scatter(self.X[:1000, 0], self.X[:1000, 1], alpha=0.2)
            plt.scatter(test_data[:, 0], test_data[:, 1], alpha=0.1)
            # ax.set_xlim(-0.75, 0.75)
            # ax.set_ylim(0.1, 0.9)
            plt.savefig(f'moons_{batch_idx}')
            plt.show()

            z_detached = z[0].detach().numpy()
            fig, ax = plt.subplots(figsize=(6, 6))
            plt.scatter(z_detached[:, 0], z_detached[:, 1], alpha=0.1)
            # ax.set_xlim(-0.75, 0.75)
            # ax.set_ylim(-0.75, 0.75)
            plt.show()
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        log_p, logdet, _ = y_hat
        loss, log_p, log_det = calc_loss(log_p, logdet)
        return loss

    def test_step(self, batch, batch_idx):
        x = batch
        y_hat = self(x)
        log_p, logdet, _ = y_hat
        loss, log_p, log_det = calc_loss(log_p, logdet)
        return loss


def add_noise(X, scale=0.05):
    N = len(X)
    X[:, 0] += np.random.normal(0, scale, N)
    X[:, 1] += np.random.normal(0, scale, N)
    return X


class VectorsDataset(Dataset):
    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X[idx]


def sample_data(X, batch_size):
    dataset = VectorsDataset(X)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=1
            )
            loader = iter(loader)
            yield next(loader)


def calc_loss(log_p, logdet):
    n_pixel = 2
    loss = log_p * n_pixel + logdet
    # loss = log_p * n_pixel / torch.sum(log_p) / 1000 + logdet
    # loss = logdet
    return (
        (-loss / (log(2) * n_pixel)).mean(),
        (log_p / (log(2) * n_pixel)).mean(),
        (logdet / (log(2) * n_pixel)).mean(),
    )


def train(X, model_single, model, optimizer):
    batch = b_size
    iters = 50000
    lr = 1e-3
    dataset = iter(sample_data(X, batch))

    z_sample = []
    temp = 0.5
    n_sample = 1000
    z_new = torch.randn(n_sample, 2) * temp
    z_sample.append(z_new.to(device))
    z_new = torch.randn(n_sample, 1) * temp
    z_sample.append(z_new.to(device))
    z_new = torch.randn(n_sample, 1) * temp
    z_sample.append(z_new.to(device))

    # z_sample = torch.cat([x.float().unsqueeze(0) for x in z_sample], dim=0)

    with tqdm(range(iters)) as pbar:
        for i in pbar:
            vec = next(dataset)
            vec = vec.to(device)

            if i == 0:
                with torch.no_grad():
                    log_p, logdet, _ = model.module(
                        vec # + torch.rand_like(vec) * 0.005
                    )

                    continue

            else:
                log_p, logdet, _ = model(vec # + torch.rand_like(vec) * 0.005
                                         )

            logdet = logdet.mean()

            loss, log_p, log_det = calc_loss(log_p, logdet)
            model.zero_grad()
            loss.backward()
            # warmup_lr = args.lr * min(1, i * batch_size / (50000 * 10))
            warmup_lr = lr
            optimizer.param_groups[0]["lr"] = warmup_lr
            optimizer.step()

            pbar.set_description(
                f"Loss: {loss.item():.5f}; logP: {log_p.item():.5f}; logdet: {log_det.item():.5f}; lr: {warmup_lr:.7f}"
            )
            if i % 100 == 0:
                test_data = model_single.reverse(z_sample).cpu().data
                plt.scatter(X[:1000, 0], X[:1000, 1], alpha=0.2)
                plt.scatter(test_data[:, 0], test_data[:, 1], alpha=0.1)
                plt.savefig(f'moons_{i}')
                plt.show()
            # if i % 100 == 0:
            #     with torch.no_grad():
            #         os.makedirs('moons_sample', exist_ok=True)
            #         utils.save_image(
            #             model_single.reverse(z_sample).cpu().data,
            #             f"sample/{str(i + 1).zfill(6)}.png",
            #             normalize=True,
            #             nrow=10,
            #             range=(-0.5, 0.5),
            #         )


def main_lightning(args: Namespace):
    N = 100000
    X, y = make_moons(N)
    X = X[X[:,1] > 0.5]
    X = add_noise(X) / 2

    dimension = 4
    # Xbig = np.random.random((X.shape[0], dimension)) * 0.005
    Xbig = np.zeros((X.shape[0], dimension))
    Xbig[:, :2] = X
    Xbig[:, 2:] = X
    X = Xbig
    dataset = VectorsDataset(X)

    plt.scatter(Xbig[:, 0], Xbig[:, 1], alpha=0.05)
    plt.show()

    lit_model = MoonModel(dimension, X)

    train_loader = DataLoader(dataset, batch_size=b_size)
    trainer = Trainer()
    trainer.fit(lit_model, train_loader)
    # TODO: check if gpu exists
    # trainer = Trainer(gpus=args.gpus, strategy=args.strategy)
    # trainer.fit(lit_model, train_loader)


def main():
    N = 100000
    X, y = make_moons(N)
    X = X[X[:,1] > 0.5]
    X = add_noise(X) / 2

    dimension = 4
    Xbig = np.random.random((X.shape[0], dimension)) * 0.005
    Xbig[:, :2] = X
    X = Xbig

    z_sample = []
    temp = 0.5
    n_sample = 1000
    z_new = torch.randn(n_sample, 2) * temp
    z_sample.append(z_new.to(device))
    z_new = torch.randn(n_sample, 1) * temp
    z_sample.append(z_new.to(device))
    z_new = torch.randn(n_sample, 1) * temp
    z_sample.append(z_new.to(device))

    # plt.scatter(Xbig[:, 0], Xbig[:, 1], alpha=0.05)
    # plt.show()

    model_single = MoonModel(dimension, X, z_sample)
    model = nn.DataParallel(model_single)
    # model = model_single
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    train(X, model_single, model, optimizer)


if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument("--strategy", default="ddp", type=str, choices=["ddp", "dp", "ddp_spawn"])
    parser.add_argument("--gpus", default=1, type=int)
    args = parser.parse_args()
    main_lightning(args)
    # main()
