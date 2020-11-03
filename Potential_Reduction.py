import numpy as np
import math
import matplotlib.pyplot as plt
import IPython


class Potential:
    def __init__(self, num_of_variables=0, beta=0.5, gama=0.4, x_0=None):
        self.n = num_of_variables
        self.m = 0  # num of const
        self.A = np.array([])
        self.b = np.array([])
        self.gama = gama
        self.c = np.array([])
        self.obj = ''
        self.x_0 = x_0
        self.y = np.array([])
        self.s = np.array([])
        self.beta = beta

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

    def dual_feasible(self):
        for i in range(self.n):
            if abs(np.matmul(self.A.T[i], self.y) + self.s[i] - self.c[i]) > 0.0001:
                return False
        return True



    def compute(self):
        Print = False
        counter = 0
        self.A = self.A.reshape(self.m, self.n)
        x = self.x_0.reshape(self.n, 1)
        self.b = self.b.reshape(self.m, 1)


        # L = log2(max(input) * number of inputs
        L = math.ceil(np.log2(max(abs(self.A.min()), abs(self.A.max()), abs(self.b.min()), abs(self.b.max())) + 1)) * (self.m + self.n) #* (self.m * self.n + self.n)

        e = np.ones(self.n, dtype=float).reshape(self.n,1)

        q = self.n + math.sqrt(self.n)

        epsilon = 2**(-2*L)

        if 'max' in self.obj:
            self.c *= -1

        if not np.array_equal(np.matmul(self.A, x), self.b):
            for i in range(self.m):
                if abs(np.matmul(self.A[i], x) - self.b[i]) > 0.0001:
                    print('row', i)
                    print('A', self.A)
                    print('x', x)
                    print('b', self.b)
                    raise TypeError('initial point not feasible')

        if min(x) <= 0:
            raise TypeError('initial point on boundary')

        self.y = np.array([-1.0] * self.m).reshape(self.m, 1)
        self.s = np.array([0.0] * self.n).reshape(self.n, 1)
        count = 0

        while not self.dual_feasible():
            #example 9 case
            if x.shape[0] == 4:
                self.y = np.array([-3.0, -1.0]).reshape(2, 1)
            else:
                self.y[-1] = -3.0
            for i in range(self.n):
                self.s[i] = self.c[i] - np.matmul(self.A.T[i].reshape(1, self.m), self.y)
            count += 1

        #IPython.embed()

        if min(self.s) <= 0:
            raise TypeError('initial dual point on boundary')

        if Print:
            print('L ', L)
            print('epsilon ', epsilon)
            print('y ', self.y)
            print('s ', self.s)
            print('count ', count)
        #IPython.embed()
        G_x_s = np.log((np.matmul(x.T, self.s)**q)/np.product(x * self.s))
        K = (G_x_s + 2 * math.sqrt(self.n) * L - self.n * math.log(self.n)) / self.gama

        if Print:
            print('x_0', x)
            print('A', self.A)
            print('c', self.c)
            print('K', K)

        while counter < K:

            if np.matmul(x.T, self.s) < epsilon:
                return x, counter

            X = x * np.identity(self.n)
            AX2 = np.matmul(self.A, X ** 2)

            c_hat = q * (self.s/(np.matmul(x.reshape(1, self.n), self.s))) - np.matmul(np.linalg.inv(X), e)

            u = np.matmul(np.matmul(X, (np.identity(self.n) - np.matmul(np.matmul(self.A.T, np.linalg.inv(np.matmul(AX2, self.A.T))), AX2))), c_hat)

            if np.linalg.norm(u) >= self.gama:
                # x = x - beta( X*u/(y*|u|)
                x = x - self.beta * (np.matmul(X, u) / (np.linalg.norm(u)))

            else:
                AX2 = np.matmul(self.A, X ** 2)
                #IPython.embed()
                self.y = self.y + np.matmul(np.matmul(np.matmul(np.linalg.inv(np.matmul(AX2, self.A.T)), self.A), X), (np.matmul(X, self.s) - np.linalg.norm(np.matmul(self.s.T, x)/q) * e))

                self.s = np.matmul(np.linalg.norm(np.matmul(x.T, self.s) / q) * np.linalg.inv(X), (u + e))

            counter += 1
        return x, counter


def problem(n=5, ep=0.25, task='a', gap=0.001):

    if task == 'a':
        #gap added to first element only
        gap = gap
        x_0 = np.array([ep+gap])
        for i in range(2, n+1):
            x_0 = np.append(x_0, ep**i+gap)
        beta = 0.2

    elif task == 'b':
        x_0 = np.array([0.5]*n)
        beta = 0.2
    elif task == 'c':
        gap = 0.01
        x_0 = np.array([ep+gap, (1-ep**2)-gap])
        for i in range(1, n-1):
            x_0 = np.append(x_0, (ep**i)*(1-ep**2)+gap)
        beta = 0.2
    x_0 = np.append(x_0, [0]*n*2)

    LP = Potential(num_of_variables=n*3, beta=beta, gama=0.5, x_0=x_0)
    zero_list = np.zeros(3*n)
    zero_list[0] = -1
    zero_list[n] = 1
    # -x1 + s1=-ep
    LP.constraint(zero_list, [-ep])
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

def Example9():
    #max x1+x2
    # s.t. x1 + x2 <= 2
    # s.t. -x1 + x2 <= 1
    x_0 = np.array([0.5, 0.5, 1.0, 1.0])
    LP = Potential(num_of_variables=4, beta=0.5, gama=0.5, x_0=x_0)

    LP.constraint(np.array([1, 1, 1, 0]), [2])
    LP.constraint(np.array([-1, 1, 0, 1]), [1])

    LP.objective(obj='max', c=np.array([1, 1, 0, 0]))

    return LP.compute()



if __name__ == '__main__':
    # Ax = b

    #print(Example9())

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
    x = list(range(3, 15))
    for gap in [0.1, 0.001, 0.00001]:
        y = []
        for i in x:
            sol, count = problem(n=i, ep=0.25, task='a', gap=gap)
            y.append(count)
            print('sol', sol[:i])
        print(y)
        print(np.polyfit(x, y, deg=5))
        plt.plot(x, y, label=str('Gap: '+ str(gap)))
    plt.legend()
    plt.show()
    '''