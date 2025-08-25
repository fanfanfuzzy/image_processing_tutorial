#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np


class MatrixEstimator(object):
    def __init__(self, N):
        self.N = N

    def prepare_matrix(self, lam):
        A = np.zeros((self.N, self.N))
        for i in range(self.N):
            A[i, i] = 1 + 2 * lam
            if i > 0:
                A[i, i-1] = -lam
            if i < self.N - 1:
                A[i, i+1] = -lam
        return A

    def estimate(self, v, lam):
        A = self.prepare_matrix(lam)
        u_est = np.linalg.solve(A, v)
        return u_est


class GradientEstimator(object):
    def __init__(self, N):
        self.N = N

    def estimate(self, v, lam, max_iter=1000, tol=1e-6):
        u = v.copy()
        for _ in range(max_iter):
            u_old = u.copy()
            for i in range(self.N):
                grad = u[i] - v[i]
                if i > 0:
                    grad += lam * (u[i] - u[i-1])
                if i < self.N - 1:
                    grad += lam * (u[i] - u[i+1])
                u[i] = u[i] - 0.01 * grad
            
            if np.linalg.norm(u - u_old) < tol:
                break
        return u
