import numpy as np
import math
import matplotlib.pyplot as plt


class Ellipsoid:
    def __init__(self, num_of_variables=0):
        self.n = num_of_variables
        self.A = np.array([])
        self.b = np.array([])
        self.epsilon = 0
        self.c = np.array([])
        self.obj = ''
        self.num_constraint = 0

    def objective(self, obj, c):
        self.obj = obj
        self.c = np.array(c).reshape(self.n, 1)

    def constraint(self, input_list, b):
        # Ax >= b
        if len(input_list) != self.n:
            raise TypeError('constraint should be of length n')

        self.A = np.append(self.A, input_list)
        self.b = np.append(self.b, b)
        self.num_constraint += 1

    def evaluate(self, z):
        finish = True
        index = -1
        for i in range(self.num_constraint):
            if (self.A[i].reshape(self.n, 1) * z).sum() < self.b[i]:
                finish = False
                index = i
                break
        if index == -1:
            for i in range(self.num_constraint, self.A.shape[0]):
                if (self.A[i].reshape(self.n, 1) * z).sum() <= self.b[i]:
                    finish = False
                    index = i
                    break
        return finish, index


    def compute(self):
        Print = False

        U = max(max(abs(a) for a in self.A), max(abs(b) for b in self.b))
        self.A = np.reshape(self.A, (self.A.shape[0] // self.n, self.n))
        if Print:
            print('A:', self.A)
            print('U:', U)

        #V = (2*self.n)**self.n * (U * self.n)**(self.n**2)
        #v = (1.0/((self.n + 1)**U))**(self.n * (self.n + 2))

        # K = math.ceil(2*(self.n + 1) * np.log(V/v))
        # K = math.ceil(2*(self.n + 1) * np.log(2^2 * n^(n+1) * U^(2n+2)n * (n+1)^(n+2)n ))
        K = math.ceil(2*(self.n + 1) * ( self.n* np.log(2) + (self.n**2 + self.n)*np.log(self.n) + (2*self.n**2 + 2*self.n)*np.log(U) + (self.n**2 +2*self.n)*np.log(self.n + 1)))
        #K = 50000
        print('k', K)
        if Print:
            #print('V', V)
            #print('v', v)
            print('K', K)

        # I * (nU)^n
        D = (math.sqrt(self.n)*(self.n * U)**(self.n+1))**2 * np.identity(self.n)
        # z = [(nU)^n]*n
        z = np.zeros(self.n).reshape(self.n, 1)

        # epsilon = 1/((n+1)U)^(n+1)
        self.epsilon = 1/((self.n + 1) * U)**(self.n + 1)

        if Print:
            print('D:', D)
            print('z', z)
            print('Epsilon', self.epsilon)

        feasible, index = self.evaluate(z)
        counter = 0
        while not feasible and counter < K:
            counter += 1
            try:
                z = z + (1/(self.n + 1.0)) * ((np.matmul(D, self.A[index].reshape(self.n, 1))) / (math.sqrt(np.matmul(np.matmul(self.A[index].reshape(1, self.n), D), self.A[index].reshape(self.n, 1)))))
                D = (self.n**2 / (self.n**2 - 1.0)) * (D - (2.0 / (self.n + 1.0)) * (np.matmul(np.matmul(np.matmul(D, self.A[index]).reshape(self.n, 1), self.A[index].reshape(1, self.n)), D)) / (np.matmul(np.matmul(self.A[index].reshape(1, self.n), D), self.A[index].reshape(self.n, 1))))

            except:
                import IPython; IPython.embed()
            feasible, index = self.evaluate(z)
            if Print:
                print(z)

        print('_________________________')
        if counter == K:
            print('Infeasible')
            return -1

        # sliding objective
        # for max obj c^Tx, for min -c^Tx
        z_pre = np.zeros(self.n)  # used to terminate if z value doesn't change

        if 'max' in self.obj.lower():

            while counter < K:
                # cx > cx0 + sigma

                self.A = np.append(self.A, self.c.reshape(1, self.n)).reshape(self.A.shape[0] + 1, self.n)
                self.b = np.append(self.b, np.matmul(self.c.T, z)).reshape(self.b.shape[0] + 1, 1)
                #import IPython;IPython.embed()
                feasible, index = self.evaluate(z)
                z_copy = z.copy()
                #import IPython;IPython.embed()
                while not feasible and counter < K and not np.array_equal(z_pre, z):
                    z_pre = z.copy()
                    z = z + (1 / (self.n + 1.0)) * ((np.matmul(D, self.A[index].reshape(self.n, 1))) / (math.sqrt(np.matmul(np.matmul(self.A[index].reshape(1, self.n), D), self.A[index].reshape(self.n, 1)))))
                    D = (self.n ** 2 / (self.n ** 2 - 1.0)) * (D - (2.0 / (self.n + 1.0)) * (np.matmul(np.matmul(np.matmul(D, self.A[index]).reshape(self.n, 1), self.A[index].reshape(1, self.n)), D)) / (np.matmul(np.matmul(self.A[index].reshape(1, self.n), D), self.A[index].reshape(self.n, 1))))
                    counter += 1
                    feasible, index = self.evaluate(z)
                if np.array_equal(z_pre, z):
                    z_copy = z.copy()
                    break

        else:

            while counter < K:
                # -cx > -cx0 + sigma
                self.A = np.append(self.A, -1*self.c.reshape(1, self.n)).reshape(self.A.shape[0] + 1, self.n)
                self.b = np.append(self.b, -1*np.matmul(self.c.T, z)).reshape(self.b.shape[0] + 1, 1)
                feasible, index = self.evaluate(z)
                z_copy = z.copy()
                while feasible and counter < K and not np.array_equal(z_pre, z):
                    z_pre = z.copy()
                    z = z + (1 / (self.n + 1.0)) * ((np.matmul(D, -self.c.reshape(self.n, 1))) / (math.sqrt(np.matmul(np.matmul(-self.c.reshape(1, self.n), D), -self.c.reshape(self.n, 1)))))
                    D = (self.n ** 2 / (self.n ** 2 - 1.0)) * (D - (2.0 / (self.n + 1.0)) * (np.matmul(np.matmul(np.matmul(D, -self.c).reshape(self.n, 1), -self.c.reshape(1, self.n)), D)) / (np.matmul(np.matmul(-self.c.reshape(1, self.n), D), -self.c.reshape(self.n, 1))))
                    counter += 1
                    feasible, index = self.evaluate(z)
                if np.array_equal(z_pre, z):
                    z_copy = z.copy()
                    break

        # for last iteration z will violate feasibility, so we will use z_copy
        print(z_copy)
        print(counter, 'iterations')
        return counter


def problem(n=5, ep=0.25):
    LP = Ellipsoid(n)
    zero_list = np.zeros(n)
    zero_list[0] = 1
    # x1 >=ep
    LP.constraint(zero_list, [ep])
    zero_list[0] = -1
    # -x1>=-1
    LP.constraint(zero_list, [-1])

    for j in range(1, n):
        zero_list = np.zeros(n)
        zero_list[j-1] = -ep
        zero_list[j] = 1
        # -ep*xj-1 + xj >=1
        LP.constraint(zero_list, [0])
        zero_list[j-1] = -ep
        zero_list[j] = -1
        # -ep*xj-1 - xj >=-1
        LP.constraint(zero_list, [-1])

    zero_list = np.zeros(n)
    zero_list[-1] = 1
    LP.objective(obj='max', c=zero_list)
    return LP.compute()


if __name__ == '__main__':
    # Ax >= b
    '''
    LP = Ellipsoid(4)
    LP.constraint([1, -12, 1, 1], [17])
    LP.constraint([-1, -12, -1, -1], [-15])
    LP.constraint([1, -12, 100, 1], [17])
    LP.constraint([-1, -12, -1, -1], [-11])
    LP.compute()
    '''
    '''
    LP = Ellipsoid(2)
    LP.constraint([-3, -5], [-12])
    LP.constraint([-1, 0], [-3])
    LP.constraint([1, 0], [0])
    LP.constraint([0, 1], [0])
    LP.objective(obj='max', c=[1, 1])
    LP.compute()
    '''

    problem(n=11, ep=0.25)

    #ep_list = [0.05, 0.1, 0.2, 0.25, 0.3, 0.4]
    #ep_list = [0.1,0.2, 0.25, 0.3]

    x = list(range(2,11))
    #for ep in ep_list:
    y = []

    for i in range(2,11):
        y.append(problem(n=i,ep=0.25))
    plt.plot(x, y)
    print('ploy', np.polyfit(x,y,deg=4))

    #plt.legend()
    plt.show()
