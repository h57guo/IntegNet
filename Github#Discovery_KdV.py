"""
@author: Hui Wang & Heyang Guo
"""

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.path.insert(0, '../../Utilities/')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from scipy.interpolate import griddata
from pyDOE import lhs
import cmocean
import time


np.random.seed(1234)
tf.set_random_seed(1234)

# Make a list to store the loss history
loss_PINN = []
iter_PINN = []

class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_u, u, X_f, layers, lb, ub):
        self.iter = 0
        self.lb = lb
        self.ub = ub
        self.x_u = X_u[:, 0:1]
        self.t_u = X_u[:, 1:2]
        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]
        self.u = u
        self.layers = layers

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_xxx = tf.gradients(u_xx, x)[0]
        f = u_t + u_xxx + 6 * u * u_x
        return f

    def callback(self, loss):
        self.iter = self.iter + 1
        print(self.iter, 'Loss:', loss)

        iter = self.iter
        if iter % 20 == 0:
            loss_PINN.append(loss)
            iter_PINN.append(iter)

    def train(self):
        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, X_star):
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]})
        return u_star, f_star


loss_IN = []
iter_IN = []

class IntegNet_total:
    # Initialize the class
    def __init__(self, X_u, u, X_f, X_c, X_t1, X_t2, X_t3, X_t4, X_t5, X_t0, X_tt, layers, lb, ub):
        self.iter = 0
        self.lb = lb
        self.ub = ub
        self.x_u = X_u[:, 0:1]
        self.t_u = X_u[:, 1:2]
        self.x_f = X_f[:, 0:1]
        self.t_f = X_f[:, 1:2]
        self.u = u

        # Add new positions for calculating integral of motions
        self.x_c = X_c[:, 0:1]
        self.t_c = X_c[:, 1:2]
        self.x_c_1 = X_t1[:, 0:1]
        self.t_c_1 = X_t1[:, 1:2]
        self.x_c_2 = X_t2[:, 0:1]
        self.t_c_2 = X_t2[:, 1:2]
        self.x_c_3 = X_t3[:, 0:1]
        self.t_c_3 = X_t3[:, 1:2]
        self.x_c_4 = X_t4[:, 0:1]
        self.t_c_4 = X_t4[:, 1:2]
        self.x_c_5 = X_t5[:, 0:1]
        self.t_c_5 = X_t5[:, 1:2]

        self.x_t0 = X_t0[:, 0:1]
        self.t_t0 = X_t0[:, 1:2]
        self.x_tt = X_tt[:, 0:1]
        self.t_tt = X_tt[:, 1:2]

        self.layers = layers

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_u_tf = tf.placeholder(tf.float32, shape=[None, self.x_u.shape[1]])
        self.t_u_tf = tf.placeholder(tf.float32, shape=[None, self.t_u.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.x_f_tf = tf.placeholder(tf.float32, shape=[None, self.x_f.shape[1]])
        self.t_f_tf = tf.placeholder(tf.float32, shape=[None, self.t_f.shape[1]])

        self.x_c_tf = tf.placeholder(tf.float32, shape=[None, self.x_c.shape[1]])
        self.t_c_tf = tf.placeholder(tf.float32, shape=[None, self.t_c.shape[1]])
        self.x_c_tf_1 = tf.placeholder(tf.float32, shape=[None, self.x_c_1.shape[1]])
        self.t_c_tf_1 = tf.placeholder(tf.float32, shape=[None, self.t_c_1.shape[1]])
        self.x_c_tf_2 = tf.placeholder(tf.float32, shape=[None, self.x_c_2.shape[1]])
        self.t_c_tf_2 = tf.placeholder(tf.float32, shape=[None, self.t_c_2.shape[1]])
        self.x_c_tf_3 = tf.placeholder(tf.float32, shape=[None, self.x_c_3.shape[1]])
        self.t_c_tf_3 = tf.placeholder(tf.float32, shape=[None, self.t_c_3.shape[1]])
        self.x_c_tf_4 = tf.placeholder(tf.float32, shape=[None, self.x_c_4.shape[1]])
        self.t_c_tf_4 = tf.placeholder(tf.float32, shape=[None, self.t_c_4.shape[1]])
        self.x_c_tf_5 = tf.placeholder(tf.float32, shape=[None, self.x_c_5.shape[1]])
        self.t_c_tf_5 = tf.placeholder(tf.float32, shape=[None, self.t_c_5.shape[1]])
        self.x_t0_tf = tf.placeholder(tf.float32, shape=[None, self.x_t0.shape[1]])
        self.t_t0_tf = tf.placeholder(tf.float32, shape=[None, self.t_t0.shape[1]])
        self.x_tt_tf = tf.placeholder(tf.float32, shape=[None, self.x_tt.shape[1]])
        self.t_tt_tf = tf.placeholder(tf.float32, shape=[None, self.t_tt.shape[1]])

        # Boundary conditions and PDE
        self.u_pred = self.net_u(self.x_u_tf, self.t_u_tf)
        self.f_pred = self.net_f(self.x_f_tf, self.t_f_tf)



        # Calculation: Integral of motion (1st and 2nd term)
        self.u_t1 = self.net_u(self.x_c_tf_1, self.t_c_tf_1)
        self.u_t2 = self.net_u(self.x_c_tf_2, self.t_c_tf_2)
        self.u_t3 = self.net_u(self.x_c_tf_3, self.t_c_tf_3)
        self.u_t4 = self.net_u(self.x_c_tf_4, self.t_c_tf_4)
        self.u_t5 = self.net_u(self.x_c_tf_5, self.t_c_tf_5)
        self.u_t0 = self.net_u(self.x_t0_tf, self.t_t0_tf)
        self.u_tt = self.net_u(self.x_tt_tf, self.t_tt_tf)

        # Calculation: Integral of motion (3rd term)
        self.third_pred = self.net_third(self.x_c_tf, self.t_c_tf)
        self.u_t1_third = self.net_third(self.x_c_tf_1, self.t_c_tf_1)
        self.u_t2_third = self.net_third(self.x_c_tf_2, self.t_c_tf_2)
        self.u_t3_third = self.net_third(self.x_c_tf_3, self.t_c_tf_3)
        self.u_t4_third = self.net_third(self.x_c_tf_4, self.t_c_tf_4)
        self.u_t5_third = self.net_third(self.x_c_tf_5, self.t_c_tf_5)
        self.third_t0 = self.net_third(self.x_t0_tf, self.t_t0_tf)
        self.third_tt = self.net_third(self.x_tt_tf, self.t_tt_tf)

        # Add three integral of motion terms
        # 1st: u
        # 2nd: u**2
        # 3rd: 0.5 * (u_x**2) - (u**3)
        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    0.25 * tf.reduce_mean(tf.square(self.f_pred)) + \
                    tf.reduce_mean(tf.square(tf.reduce_mean(self.u_t1) - tf.reduce_mean(self.u_t2)) + \
                                   tf.square(tf.reduce_mean(self.u_t2) - tf.reduce_mean(self.u_t3)) + \
                                   tf.square(tf.reduce_mean(self.u_t3) - tf.reduce_mean(self.u_t4 )) + \
                                   tf.square(tf.reduce_mean(self.u_t4) - tf.reduce_mean(self.u_t5))) + \
                    tf.reduce_mean(tf.square(tf.reduce_mean(self.u_t1 ** 2) - tf.reduce_mean(self.u_t2 ** 2)) + \
                                   tf.square(tf.reduce_mean(self.u_t2 ** 2) - tf.reduce_mean(self.u_t3 ** 2)) + \
                                   tf.square(tf.reduce_mean(self.u_t3 ** 2) - tf.reduce_mean(self.u_t4 ** 2)) + \
                                   tf.square(tf.reduce_mean(self.u_t4 ** 2) - tf.reduce_mean(self.u_t5 ** 2))) + \
                    tf.reduce_mean(tf.square(tf.reduce_mean(self.u_t1_third) - tf.reduce_mean(self.u_t2_third)) + \
                                   tf.square(tf.reduce_mean(self.u_t2_third) - tf.reduce_mean(self.u_t3_third)) + \
                                   tf.square(tf.reduce_mean(self.u_t3_third) - tf.reduce_mean(self.u_t4_third)) + \
                                   tf.square(tf.reduce_mean(self.u_t4_third) - tf.reduce_mean(self.u_t5_third)))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x, t):
        u = self.neural_net(tf.concat([x, t], 1), self.weights, self.biases)
        return u

    def net_f(self, x, t):
        u = self.net_u(x, t)
        u_t = tf.gradients(u, t)[0]
        u_x = tf.gradients(u, x)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_xxx = tf.gradients(u_xx, x)[0]
        f = u_t + u_xxx + 6 * u * u_x

        return f

    # Define a function to calculate 3rd term (Integral of motion)
    def net_third(self, x, t):
        u = self.net_u(x, t)
        u_x = tf.gradients(u, x)[0]
        third = ((1 / 2) * (u_x ** 2)) - (u ** 3)

        return third

    def callback(self, loss):
        self.iter = self.iter + 1
        print(self.iter, 'Loss:', loss)

        iter = self.iter
        if iter % 20 == 0:
            loss_IN.append(loss)
            iter_IN.append(iter)

    def train(self):
        tf_dict = {self.x_u_tf: self.x_u, self.t_u_tf: self.t_u, self.u_tf: self.u,
                   self.x_f_tf: self.x_f, self.t_f_tf: self.t_f,
                   self.x_c_tf: self.x_c, self.t_c_tf: self.t_c,
                   self.x_c_tf_1: self.x_c_1, self.t_c_tf_1: self.t_c_1,
                   self.x_c_tf_2: self.x_c_2, self.t_c_tf_2: self.t_c_2,
                   self.x_c_tf_3: self.x_c_3, self.t_c_tf_3: self.t_c_3,
                   self.x_c_tf_4: self.x_c_4, self.t_c_tf_4: self.t_c_4,
                   self.x_c_tf_5: self.x_c_5, self.t_c_tf_5: self.t_c_5,
                   self.x_t0_tf: self.x_t0, self.t_t0_tf: self.t_t0,
                   self.x_tt_tf: self.x_tt, self.t_tt_tf: self.t_tt}

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)


    def predict(self, X_star):
        u_star = self.sess.run(self.u_pred, {self.x_u_tf: X_star[:, 0:1], self.t_u_tf: X_star[:, 1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f_tf: X_star[:, 0:1], self.t_f_tf: X_star[:, 1:2]})

        return u_star, f_star



if __name__ == "__main__":
    N_u = 200
    N_f = 10000
    layers = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    data = scipy.io.loadmat('../Data/3KdV200x256_-10x30.mat')

    t = data['kt'].flatten()[:, None]
    x = data['kx'].flatten()[:, None]
    Exact = np.real(data['kusol'])

    X, T = np.meshgrid(x, t)

    X_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_star = Exact.flatten()[:, None]

    # Doman bounds
    lb = X_star.min(0)
    ub = X_star.max(0)

    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    uu1 = Exact[0:1, :].T
    xx2 = np.hstack((X[:, 0:1], T[:, 0:1]))
    uu2 = Exact[:, 0:1]
    xx3 = np.hstack((X[:, -1:], T[:, -1:]))
    uu3 = Exact[:, -1:]

    X_u_train = np.vstack([xx1, xx2, xx3])
    u_train = np.vstack([uu1, uu2, uu3])

    # Create blocks of data to calculate integral of motion
    # Here we take five sections as an example
    X_t0 = xx1                              # First column
    X_tt = X_star[-256:, :]                 # Last column

    kn = 256 * 40
    X_t_1 = X_star[:kn, :]
    X_t_2 = X_star[kn: 2*kn ,:]
    X_t_3 = X_star[2*kn:3*kn, :]
    X_t_4 = X_star[3*kn:4*kn, :]
    X_t_5 = X_star[4*kn:5*kn, :]


    X_f_train = lb + (ub - lb) * lhs(2, N_f)
    X_f_train = np.vstack((X_f_train, X_u_train))

    idx = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx, :]
    u_train = u_train[idx, :]

    ######################################################################
    #===================  PINN and IntegNet   =====================
    ######################################################################

    model_PINN = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub)

    model_IN = IntegNet_total(X_u_train, u_train, X_f_train, X_star, X_t_1, X_t_2, X_t_3, X_t_4, X_t_5, X_t0, X_tt, layers, lb, ub)

    start_time1 = time.time()
    model_PINN.train()
    elapsed1 = time.time() - start_time1

    start_time2 = time.time()
    model_IN.train()
    elapsed2 = time.time() - start_time2

    print('Training time of PINN: %.1f' % (elapsed1), 's = %.1f' % (elapsed1 / 60), 'min')
    print('Training time of IntegNet: %.1f' % (elapsed2), 's = %.1f' % (elapsed2 / 60), 'min')
    print('Total training time: %.1f' % (elapsed1 + elapsed2), 's = %.1f' % ((elapsed1 + elapsed2) / 60), 'min')

    # Errors of PINN model
    u_pred1, f_pred1 = model_PINN.predict(X_star)
    error_u1 = np.linalg.norm(u_star - u_pred1, 2) / np.linalg.norm(u_star, 2)

    error_u1_Linf = np.linalg.norm(u_star - u_pred1, np.inf) / np.linalg.norm(u_star, np.inf)
    error_u1_L1 = np.linalg.norm(u_star - u_pred1, 1) / np.linalg.norm(u_star, 1)

    print('Error u_PINN_Linf: %e' % (error_u1_Linf), 'L1: %e' %(error_u1_L1), 'L2: %e' %(error_u1))
    U_pred1 = griddata(X_star, u_pred1.flatten(), (X, T), method='cubic')
    Error1 = np.abs(Exact - U_pred1)

    # Errors of IntegNet model
    u_pred3, f_pred3 = model_IN.predict(X_star)
    error_u3 = np.linalg.norm(u_star - u_pred3, 2) / np.linalg.norm(u_star, 2)
    error_u3_Linf = np.linalg.norm(u_star - u_pred3, np.inf) / np.linalg.norm(u_star, np.inf)
    error_u3_L1 = np.linalg.norm(u_star - u_pred3, 1) / np.linalg.norm(u_star, 1)

    print('Error u_IntegNet_Linf: %e' % (error_u3_Linf), 'L1: %e' %(error_u3_L1), 'L2: %e' %(error_u3))
    U_pred3 = griddata(X_star, u_pred3.flatten(), (X, T), method='cubic')
    Error3 = np.abs(Exact - U_pred3)

    # Variance and Std calculation of PINN's result and IntegNet's result
    Error1 = np.nan_to_num(Error1)
    Error3 = np.nan_to_num(Error3)
    VAR1 = np.var(Error1)
    VAR1_std = np.std(Error1)
    VAR3 = np.var(Error3)
    VAR3_std = np.std(Error3)

    print('Var of PINN error', VAR1, 'Var of IntegNet error:', VAR3)
    print('Std of PINN error:', VAR1_std, 'Std of IntegNet error:', VAR3_std)


    ######################################################################
    ############################# Plotting ###############################
    ######################################################################
    plt.figure(1)
    plt.imshow(Error1, interpolation='nearest', cmap=cmocean.cm.deep,
               extent=[x.min(), x.max(), t.min(), t.max()],
               origin='lower', aspect='auto')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.colorbar()

    plt.figure(2)
    plt.imshow(Error3, interpolation='nearest', cmap=cmocean.cm.deep,
               extent=[x.min(), x.max(), t.min(), t.max()],
               origin='lower', aspect='auto')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.colorbar()
    plt.clim(Error1.min(), Error1.max())

    plt.figure(3)
    plt.imshow(Exact, interpolation='nearest', cmap=cmocean.cm.deep,
               extent=[x.min(), x.max(), t.min(), t.max()],
               origin='lower', aspect='auto')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.colorbar()

    plt.figure(4)
    plt.imshow(U_pred1, interpolation='nearest', cmap=cmocean.cm.deep,
               extent=[x.min(), x.max(), t.min(), t.max()],
               origin='lower', aspect='auto')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.colorbar()

    plt.figure(5)
    plt.imshow(U_pred3, interpolation='nearest', cmap=cmocean.cm.deep,
               extent=[x.min(), x.max(), t.min(), t.max()],
               origin='lower', aspect='auto')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.colorbar()

    plt.figure(6)
    PINN = plt.plot(iter_PINN, loss_PINN, color='black', linestyle='--', linewidth=1.5, label='PINN')
    IN = plt.plot(iter_IN, loss_IN, color='red', linewidth=1.5, label='IntegNet')
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.legend()
    plt.yscale('log')

    plt.show()

