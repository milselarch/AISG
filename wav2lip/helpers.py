import math
import time
import random
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch import Tensor
from torch.nn import Module
from typing import Optional

bce = torch.nn.BCELoss

class WeightedLogitsBCE(Module):
    def __init__(
        self, var_scale=0.05, min_weight=1e-4,
        base_loss_ratio=0.5
    ) -> None:
        super(WeightedLogitsBCE, self).__init__()

        self.min_weight = min_weight
        self.var_scale = var_scale
        self.base_loss_ratio = base_loss_ratio

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @staticmethod
    def log_sigmoid(x):
        # gives stable log(sigmoid(x))
        return x - torch.log(1. + torch.exp(x))

    @classmethod
    def ilog_sigmoid(cls, x):
        # gives stable log(1 - sigmoid(x))
        # log(1 - sigmoid(x))  = log(sigmoid(-x))
        return cls.log_sigmoid(-x)

    @classmethod
    def calculate_loss(
        cls, preds: Tensor, target: Tensor, scaled_weight: Tensor,
        base_loss_ratio: float
    ):
        """
        bce_loss = F.binary_cross_entropy(
            input, target, scaled_weight.detach(),
            reduction='sum'
        )
        bce_loss = torch.dot(-scaled_weight, (
            target * torch.log(input) +
            (1. - target) * torch.log(1. - input)
        ))
        """
        bce_losses = -1.0 * (
            target * cls.log_sigmoid(preds) +
            (1. - target) * cls.ilog_sigmoid(preds)
        )

        base_bce_loss = torch.sum(bce_losses)
        scaled_bce_loss = torch.dot(scaled_weight, bce_losses)
        total_loss = (
            base_bce_loss * base_loss_ratio +
            scaled_bce_loss * (1. - base_loss_ratio)
        )

        return total_loss

    def forward(
        self, preds: Tensor, target: Tensor,
        weight: Optional[Tensor] = None
    ) -> Tensor:
        if weight is None:
            weight = torch.ones(preds.shape)

        try:
            assert len(weight.shape) == 1
            assert (weight >= 0).all()
        except AssertionError as e:
            print(f'BAD WEIGHT {weight}')
            raise e

        safe_weight = weight + self.min_weight
        # scale weights such that sum of weights is 1
        scaled_weight = safe_weight / sum(safe_weight)
        # normalise weights such that mean weight is 1
        norm_weight = scaled_weight * len(safe_weight)

        bce_loss = self.calculate_loss(
            preds, target, scaled_weight,
            base_loss_ratio=self.base_loss_ratio
        )

        variance = torch.var(norm_weight, unbiased=True)
        loss = bce_loss + variance * self.var_scale
        assert not torch.isnan(loss).any()
        return loss


class Binomial(object):
    def __init__(self, count, prob):
        self.count = count
        self.prob = prob

    @staticmethod
    def f(num):
        return math.factorial(num)

    @classmethod
    def comb(cls, n, k, p):
        return cls.f(n) / (cls.f(n-k) * cls.f(k))

    @classmethod
    def bin_p(cls, n, k, p):
        prob = (p ** k) * (1-p) ** (n-k)
        return cls.comb(n, k, p) * prob

    @classmethod
    def make_pdfs(cls, n, p):
        pdfs = []
        for k in range(n+1):
            pdf = cls.bin_p(n, k, p)
            pdfs.append(pdf)

        return pdfs

    @classmethod
    def make_cdfs(cls, n, p, cap=True):
        pdfs = cls.make_pdfs(n, p)
        cdf, cdfs = 0, []

        for k, pdf in enumerate(pdfs):
            cdf += pdf
            cdfs.append(cdf)

        if cap is True:
            # print('CAP,', cdfs)
            cdfs[-1] = 1

        return cdfs

    @classmethod
    def random_sample(cls, count, prob):
        cdfs = cls.make_cdfs(count, prob, cap=True)
        prob = random.random()

        for k, cdf in enumerate(cdfs):
            if prob <= cdf:
                return k

        raise ValueError(f'BAD CDF DIST {cdfs}')

    def make_random_sample(self):
        return self.random_sample(self.count, self.prob)

    def make_random_samples(self, num_samples=1000):
        samples = []
        for k in range(num_samples):
            sample = self.make_random_sample()
            samples.append(sample)

        return samples


if __name__ == '__main__':
    dist = Binomial(32, 0.5)
    pdf_list = dist.make_pdfs(32, 0.5)
    cdf_list = dist.make_cdfs(32, 0.5)
    print(cdf_list)

    rand_samples = dist.make_random_samples(100000)
    plt.hist(rand_samples, bins=list(range(33)))
    plt.show()