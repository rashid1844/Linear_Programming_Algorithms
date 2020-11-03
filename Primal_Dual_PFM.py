import numpy as np
import math
import matplotlib.pyplot as plt
import IPython


class PrimalDualPFM:
    def __init__(self, num_of_variables=0, x_0=None):
        self.n = num_of_variables
        self.m = 0  # num of const
        self.A = np.array([])
        self.b = np.array([])
        self.c = np.array([])
        self.obj = ''
        self.x = x_0
        self.p = np.array([])
        self.s = np.array([])
        self.Q = np.array([])

    def objective(self, obj, c, Q):
        self.obj = obj
        self.c = np.array(c).reshape(self.n, 1)
        self.Q = np.array(Q)

    def dual_constraint(self, s, p):
        self.s = np.array(s)
        self.p = np.array(p)

    def constraint(self, input_list, b):
        # Ax = b
        if len(input_list) != self.n:
            raise TypeError('constraint should be of length n')
        self.A = np.append(self.A, input_list)
        self.b = np.append(self.b, b)
        self.m += 1

    def compute(self):
        Print = True
        counter = 0
        self.A = self.A.reshape(self.m, self.n)
        self.x = self.x.reshape(self.n, 1)
        self.b = self.b.reshape(self.m, 1)
        self.s = self.s.reshape(self.n, 1)
        self.p = self.p.reshape(self.m, 1)
        self.Q = self.Q.reshape(self.n, self.n)

        count = 0
        delta = 1.0
        while True:
            delta -= 0.01
            alpha = 1 - delta/math.sqrt(self.n)
            beta = 0.49
            while 0 < beta-0.01 < (beta + delta)**2/(2*(1-beta)*alpha):
                beta -= 0.01
                count += 1

            if beta >= 0.001:
                break
            if delta <= 0.001:
                raise TypeError('could not find alpha, beta, delta')

        if Print:
            print('alpha', alpha)
            print('beta', beta)
            print('delta', delta)
            print('count', count)

        # L = log2(max(input) * number of inputs
        L = math.ceil(np.log2(max(abs(self.A.min()), abs(self.A.max()), abs(self.b.min()), abs(self.b.max())) + 1)) * (self.m + self.n) #* (self.m * self.n + self.n)

        e = np.ones(self.n, dtype=float).reshape(self.n, 1)

        epsilon = 2**(-2*L)

        if 'max' in self.obj:
            self.c *= -1

        if min(self.x) <= 0 or min(self.s) <= 0:
            raise TypeError('initial point on or out of boundary')

        feasibility = True
        for i in range(self.m):
            if abs(np.matmul(self.A[i], self.x) - self.b[i]) > 0.0001:
                feasibility = False
                break

        eq2 = - np.matmul(self.Q, self.x) + np.matmul(self.A.T, self.p) + self.s

        for i in range(self.n):
            if abs(eq2[i] - self.c[i]) > 0.0001:
                feasibility = False
                break


        if Print:
            print('feasibility', feasibility)
            print('L ', L)
            print('epsilon ', epsilon)
            print('p ', self.p)
            print('s ', self.s)

        K = math.ceil(np.log(((1 + beta) * np.matmul(self.x.T, self.s)) / ((1 - beta) * epsilon)) / np.log(1 / alpha)) * 2
        mu = np.linalg.norm(self.x.T @ self.s)/self.n
        #IPython.embed()

        if Print:
            print('x_0', self.x)
            print('mu0', mu)
            print('A', self.A)
            print('c', self.c)
            print('K', K)

        while counter < K:
            if Print:
                print('duality', np.matmul(self.x.T, self.s))

            if self.x.T @ self.s < epsilon:
                return self.x, counter

            X = self.x * np.identity(self.n)
            S = self.s * np.identity(self.n)

            mu *= alpha

            xi_mu = X @ S @ e - mu * e
            S_XQ_inv = np.linalg.inv(S + X @ self.Q)
            XAT = X @ self.A.T

            print('error', np.linalg.norm(xi_mu), mu)

            if feasibility:
                d_p = np.linalg.inv(self.A @ S_XQ_inv @ XAT)
                d_p = d_p @ self.A @ S_XQ_inv @ xi_mu
                d_x = S_XQ_inv @ (XAT @ d_p - xi_mu)
                d_s = - np.linalg.inv(X) @ (xi_mu + S @ d_x)
                self.x = self.x + d_x
                self.p = self.p + d_p
                self.s = self.s + d_s

            else:
                d_p = - np.linalg.inv(self.A @ S_XQ_inv @ XAT) @ (self.A @ S_XQ_inv @ (XAT @ self.p + X @ S @ e - X @ self.c - xi_mu - X @ self.Q @ self.x) + (self.A @ self.x - self.b))
                d_x = S_XQ_inv @ (X @ self.s - X @ self.c + XAT @ self.p + XAT @ d_p - xi_mu - X @ self.Q @ self.x)
                d_s = - np.linalg.inv(X) @ (xi_mu + S @ d_x)
                self.x = self.x + d_x
                self.p = self.p + d_p
                self.s = self.s + d_s

            counter += 1
        return self.x, counter


def Problem(task='a'):

    if task == 'a':
        x_0 = np.array([1.0, 1.0, 1.0])
        p = np.array([-7.0])
        s = np.array([3, 1, 1])

    else:
        x_0 = np.array([1.0, 2.0, 2.0])
        p = np.array([-1.0])
        s = np.array([0.2, 0.2, 0.2])

    Q = np.array([[4, 0, 0], [0, 1, -1], [0, -1, 1]])
    c = np.array([-8, -6, -6])

    LP = PrimalDualPFM(num_of_variables=3, x_0=x_0)

    LP.constraint(np.array([1, 1, 1]), [3])

    LP.objective(obj='min', c=c, Q=Q)
    LP.dual_constraint(s=s, p=p)

    return LP.compute()


if __name__ == '__main__':

    print(Problem(task='b'))
