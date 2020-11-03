import numpy as np
import matplotlib.pyplot as plt

class simplex:
    def __init__(self, num_of_variables=0):
        self.num_of_vars = num_of_variables
        self.A = []
        self.c = np.array([])
        self.b = np.array([])

    def constraint(self, input_list, b):
        if len(input_list) != self.num_of_vars:
            raise TypeError('constraint should be of length inputs + b')
        self.A.append(input_list)
        self.b = np.append(self.b, b)

    def objective(self, input_list):
        if input_list.shape[0] != self.num_of_vars:
            raise TypeError('objective should be of length inputs')
        self.c = input_list

    def compute(self):
        Print = False
        self.A = np.array(self.A)
        if self.A.shape[0] == 0 or self.c.shape[0] == 0 or self.b.shape[0] == 0:
            raise TypeError('one of the inputs is not defined')
        if Print:
            print('A', self.A)
            print('b', self.b)
            print('c', self.c)
        non_basic = np.array([i for i in range(self.num_of_vars - self.A.shape[0])])
        basic = np.array([i for i in range(self.num_of_vars - self.A.shape[0], self.num_of_vars, 1)])
        if Print:
            print('non_basic', non_basic)
            print('basic', basic)

        iteration = 0
        ### Loop
        while True:
            B = []
            N = []
            for c in range(self.A.shape[0]):
                temp1 = []
                temp2 = []
                for b in basic:
                    temp1.append(self.A[c][b])
                for n in non_basic:
                    temp2.append(self.A[c][n])
                B.append(temp1)
                N.append(temp2)

            B = np.array(B)
            N = np.array(N)
            if Print:
                print('B', B)
                print('N', N)
            try:
                B_inv = np.linalg.inv(B)
            except Exception:
                return None, 0
            if Print:
                print('B_inv', B_inv)

            # for x0 case
            if iteration == 0:
                x = np.zeros_like(self.c)
                for i, b in enumerate(basic):
                    x[b] = self.b[i]
            if Print:
                print('x', x)

            # calc c
            c_b = np.array([])
            c_n = np.array([])
            for i in range(self.num_of_vars):
                if i in basic:
                    c_b = np.append(c_b, self.c[i])
                else:
                    c_n = np.append(c_n, self.c[i])
            if Print:
                print('c_b', c_b)
                print('c_n', c_n)
            lumda = c_b.dot(B_inv)
            if Print:
                print('lumda', lumda)

            u_n = c_n - lumda.dot(N)
            if Print:
                print('u_n', u_n)
            u = np.zeros(self.num_of_vars)
            for i, n in enumerate(non_basic):
                u[n] = u_n[i]
            if Print:
                print('u', u)

            iteration += 1

            if iteration > 50000:
                return None, 0

            if Print:
                print('min(u)', np.amin(u))
            if np.amin(u) < 0:
                if Print:
                    print('not optimal')
                entering = np.where(u == np.amin(u))
                entering = entering[0][0]
                # picks the fist min value in case of tie
                if Print:
                    print('entering index', entering)
            else:
                if Print:
                    print('Optimal')
                    print(iteration)
                    print(x)
                break

            # to find index of entering in non_basic array
            e_l = np.zeros_like(non_basic)
            e_l[np.where(non_basic == entering)] = 1
            if Print:
                print('e_l', e_l)

            d_B = - np.matmul(B_inv, N).dot(e_l)

            if Print:
                print('d_B', d_B)

            d = np.zeros(self.num_of_vars)
            for i in range(self.num_of_vars):
                if i in basic:
                    d[i] = d_B[np.where(i == basic)]
                else:
                    d[i] = e_l[np.where(i == non_basic)]
            if Print:
                print('d', d)
            a = None
            try:
                a = min(-1*x[i]/d[i] for i in range(self.num_of_vars) if d[i] < 0)
            except Exception as e:
                print(e)
                #import IPython; IPython.embed()
                return None, 0

            if Print:
                print('a', a)
            x = x + a * d
            if Print:
                print('x', x)

            # set leaving as first basic, then check if any other
            # basic has lower x value
            leaving = basic[0]
            for b in basic:
                if x[b] < x[leaving]:
                    leaving = b
            if Print:
                print('leaving index', leaving)


            # Update basic and non basic
            basic = np.append(basic, entering)
            non_basic = np.append(non_basic, leaving)
            basic = np.delete(basic, np.where(basic == leaving))
            non_basic = np.delete(non_basic, np.where(non_basic == entering))
            basic = np.sort(basic)
            non_basic = np.sort(non_basic)
            if Print:
                print('basic', basic)
                print('non_basic', non_basic)
        return np.around(x, decimals=3), iteration



def task_1():
    output = {}
    iterations = []
    for n in range(1, 15, 1):
        LP = simplex(2*n)  # 2* for basic from constraint

        for i in range(1, n+1, 1):
            const = [2**(i-j+1) for j in range(1, i, 1)]
            const.append(1)  # for xi
            if n > i:
                for _ in range(n-i):
                    const.append(0)  # for non-basic but non participating

            if (i != 1):
                for _ in range(i-1):
                    const.append(0)  # for non participating basic
            const.append(1)  # for participating basic
            if i != n:
                for _ in range(n-i):
                   const.append(0)  # for non participating basic
            print('const', i, const)
            LP.constraint(const, [5**i])

        # -2 to convert it to maximize
        obj_arr = [-2**(n-j) for j in range(1, n+1, 1)]
        for _ in range(n):
            obj_arr.append(0)  # for basic
        print('obj', obj_arr)
        LP.objective(np.array(obj_arr))

        output[n], itr = LP.compute()
        iterations.append(itr)
    print('#################################################')
    for key in output.keys():
        print(key, output[key])
    plt.plot(range(1, len(iterations)+1), iterations)
    plt.show()



def task_2():
    np.random.seed(13)  # to make sure it works
    output = {}
    margin = 0.1
    num_of_devi = 9

    iter_devi = []  # iter_devi[devi_index][n_val]=itermation
    plot_x = []
    for devi in range(1, num_of_devi+1):
        devi = devi * margin  # deviation is from 0.5 till 5 (step: 0.5)
        iterations = []
        plot_x1 = []
        for n in range(1, 11, 1):
            LP = simplex(2*n)  # 2* for basic from constraint

            for i in range(1, n+1, 1):
                const = [2**(i-j+1)+np.random.normal(0,devi,None) for j in range(1, i, 1)]
                const.append(1+np.random.normal(0,devi,None))  # for xi
                if n > i:
                    for _ in range(n-i):
                        const.append(0)  # for non-basic but non participating

                if (i != 1):
                    for _ in range(i-1):
                        const.append(0)  # for non participating basic
                const.append(1)  # for participating basic
                if i != n:
                    for _ in range(n-i):
                       const.append(0)  # for non participating basic
                print('const', i, const)
                LP.constraint(const, [5**i])

            # -2 to convert it to maximize
            obj_arr = [(-2**(n-j))+np.random.normal(0,devi,None) for j in range(1, n+1, 1)]
            for _ in range(n):
                obj_arr.append(0)  # for basic
            print('obj', obj_arr)
            LP.objective(np.array(obj_arr))

            output[n], itr = LP.compute()
            iterations.append(itr)
            plot_x1.append(n)
        iter_devi.append(iterations)
        plot_x.append(plot_x1)
    print(iter_devi)
    #import IPython; IPython.embed()

    #print runs with not answer
    for i in range(len(iter_devi)):
        j = 0
        while j < len(iter_devi[i]):
            if iter_devi[i][j] == 0:
                print('Solution wasn\'t found at deviation:', np.round((i+1)*margin, decimals=1), 'with number of variables =', j)
                del plot_x[i][j]
                del iter_devi[i][j]
                j -= 1
            j += 1

    for i in range(num_of_devi):
        #plt.plot(range(1, len(iter_devi[i])+1), iter_devi[i], label=str(np.round((i+1)*margin, decimals=1)))
        plt.plot(plot_x[i], iter_devi[i], label=str(np.round((i+1)*margin, decimals=1)))
    plt.legend()
    plt.show()



if __name__ == '__main__':
    #LP = simplex(4)
    #LP.constraint([1, 2, 1, 0], [3])
    #LP.constraint([2, 1, 0, 1], [3])
    #LP.objective(np.array([-1, -1, 0, 0]))
    #LP.compute()
    #LP = simplex(4)
    #LP.constraint([8, 3, 1, 1], [12])
    #LP.objective(np.array([-4, -2, 5, 0]))
    #print(LP.compute())


    task_2()

