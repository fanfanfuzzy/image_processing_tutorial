#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


class Image1D(object):
    def __init__(self, N):
        self.N = N

    def generate(self, a, seed=None):
        np.random.seed(seed=seed)
        sigma = a ** 0.5
        u = [0]
        for _ in range(self.N - 1):
            u.append(u[-1] + np.random.randn() * sigma)
        return np.array(u)

    def add_noise(self, u, b, seed=None):
        np.random.seed(seed=seed)
        sigma = b ** 0.5
        v = u + np.random.randn(self.N) * sigma
        return v
