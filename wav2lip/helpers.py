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

class WeightedBCE(Module):
    def __init__(self, var_scale=0.05) -> None:
        super(WeightedBCE, self).__init__()
        self.var_scale = var_scale

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(
        self, input: Tensor, target: Tensor,
        weight: Optional[Tensor] = None
    ) -> Tensor:
        if weight is None:
            weight = torch.ones(input.shape)

        assert len(weight.shape) == 1
        norm_weight = weight * len(weight) / sum(weight)
        variance = torch.var(norm_weight, unbiased=True)
        bce_loss = F.binary_cross_entropy(
            input, target, weight=weight.detach(),
            reduction='mean'
        )

        loss = bce_loss + variance * self.var_scale
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