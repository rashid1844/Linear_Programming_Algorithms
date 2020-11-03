import numpy as np
import math
import matplotlib.pyplot as plt
import IPython


class Affine:
    def __init__(self, num_of_variables=0, beta=0.5, epsilon=0.001, x_0=None):
        self.n = num_of_variables
        self.m = 0  # num of const
        self.A = np.array([])
        self.b = np.array([])
        self.epsilon = epsilon
        self.c = np.array([])
        self.beta = beta
        self.obj = ''
        self.x_0 = x_0

    def objective(self, obj, c):
        self.obj = obj
        self.c = np.array(c).reshape(self.n, 1)

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
        x = self.x_0.reshape(self.n, 1)
        self.b = self.b.reshape(self.m, 1)
        #IPython.embed()

        if 'max' in self.obj:
            self.c *= -1

        if Print:
            print('x_0', x)
            print('A', self.A)
            print('b', self.b)
            print('c', self.c)

        # checks initial point to be feasible
        init = True
        for i in range(self.m):
            if abs(np.matmul(self.A[i], x) - self.b[i]) > 0.0001:
                print('row: ', i)
                init = False
                IPython.embed()
        if not init:
            raise TypeError('initial point not feasible')
        #exponentinal
        K = 2**self.n

        while counter < K:
            X = x * np.identity(self.n)
            AX2 = np.matmul(self.A, X**2)
            Lambda = np.matmul(np.linalg.inv(np.matmul(AX2, self.A.T)), np.matmul(AX2, self.c))
            u = np.matmul(X, (self.c - np.matmul(self.A.T, Lambda)))

            #IPython.embed()
            if min(u) >= 0 and u.sum() < self.epsilon:
                return x, counter

            if max(np.matmul(X, u)) < 0: #TODO check if this is correct
                print(x, u)
                return x, 'unbounded'

            y = max(u)
            y = 1  #TODO remove
            #y = 1 if y <= 0 else y
            # x = x - beta( X*u/(y*|u|)
            x = np.array(x - self.beta*(np.matmul(X, u)/(y*np.linalg.norm(u))), dtype=np.float64)
            counter += 1


def problem(n=5, ep=0.25, task='a'):

    if task == 'a':
        #gap added to first element only
        gap = 0.01
        x_0 = np.array([ep+gap])
        for i in range(2, n+1):
            x_0 = np.append(x_0, ep**i+gap)
        beta = 0.5
    elif task == 'b':
        x_0 = np.array([0.5]*n)
        beta = 0.3

    elif task == 'c':
        gap = 0.01
        x_0 = np.array([ep+gap, (1-ep**2)-gap])
        for i in range(1, n-1):
            x_0 = np.append(x_0, (ep**i)*(1-ep**2)+gap)
        beta = 0.2

    x_0 = np.append(x_0, [0]*n*2)

    LP = Affine(num_of_variables=n*3, beta=beta, epsilon=0.1, x_0=x_0)
    zero_list = np.zeros(3*n)
    zero_list[0] = 1
    zero_list[n] = -1
    # x1 - s1=ep
    LP.constraint(zero_list, [ep])
    x_0[n] = x_0[0] - ep

    zero_list = np.zeros(3 * n)
    zero_list[0] = 1
    zero_list[n+1] = 1
    # x1 +s2=1
    LP.constraint(zero_list, [1])
    x_0[n+1] = 1 - x_0[0]

    for j in range(1, n):
        zero_list = np.zeros(3 * n)
        zero_list[j-1] = ep
        zero_list[j] = -1
        zero_list[n+2*j] = 1
        # ep*xj -xj +s=0
        LP.constraint(zero_list, [0])
        x_0[n+2*j] = x_0[j] - ep * x_0[j-1]

        zero_list = np.zeros(3 * n)
        zero_list[j-1] = ep
        zero_list[j] = 1
        zero_list[n + 2 * j+1] = 1
        # ep*xj + xj +s =1
        LP.constraint(zero_list, [1])
        x_0[n + 2 * j+1] = 1 - x_0[j] - ep * x_0[j-1]

    zero_list = np.zeros(3 * n)
    zero_list[n-1] = 1
    LP.objective(obj='max', c=zero_list)
    LP.x_0 = x_0
    if min(x_0) < 0:
        raise TypeError('negative input or slack')

    return LP.compute()


if __name__ == '__main__':
    # Ax = b
    '''
    x = list(range(3, 15))
    for task in ['a', 'b', 'c']:
        y = []
        for i in x:
            sol, count = problem(n=i, ep=0.25, task=task)
            y.append(count)
            print('sol', sol[:i])
        print(y)
        print(np.polyfit(x, y, deg=5))
        plt.plot(x, y, label=str('Task '+task))
    plt.legend()
    plt.show()
    '''
    print(problem(n=14, ep=0.25, task='b'))
