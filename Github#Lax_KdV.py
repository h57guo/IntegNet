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
import time


np.random.seed(1234)
tf.set_random_seed(1234)

loss_history_0 = []
iter_history_0 = []
l1_history_0 = []
l2_history_0 = []
l3_history_0 = []
e1_history_0 = []
e2_history_0 = []
e3_history_0 = []


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X, u, layers, lb, ub):
        self.iter = 0
        self.lb = lb
        self.ub = ub
        self.a = 0.1
        self.x = X[:, 0:1]
        self.t = X[:, 1:2]
        self.u = u

        self.layers = layers

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        # Initialize parameters
        self.lambda_1 = tf.Variable([1.0], dtype=tf.float32)
        self.lambda_2 = tf.Variable([1.0], dtype=tf.float32)
        self.lambda_3 = tf.Variable([1.0], dtype=tf.float32)

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.u_pred = self.net_u(self.x_tf, self.t_tf)
        self.f_pred_1 = self.net_f(self.x_tf, self.t_tf)

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    tf.reduce_mean(tf.square(self.f_pred_1))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 100000,
                                                                         'maxfun': 100000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps,
                                                                         'gtol': 1.0 * np.finfo(float).eps,
                                                                        }
                                                                )


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
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        lambda_3 = self.lambda_3
        u = self.net_u(x, t)
        u_x = tf.gradients(u, x)[0]

        # Learning Lax Pairs
        self.a = 1
        U_11 = 0
        U_12 = 1
        U_21 = self.a - u
        U_22 = 0
        V_12 = lambda_1 * self.a + lambda_2 * u + lambda_3 * u_x
        V_11 = - tf.gradients(V_12, x)[0] / 2
        V_21 = tf.gradients(V_11, x)[0] + V_12 * U_21
        V_22 = -V_11
        f3 = tf.gradients(U_21, t)[0] - tf.gradients(V_21, x)[0] - (V_21 * U_11 + V_22 * U_21) + (
                U_21 * V_11 + U_22 * V_21)
        return f3

    def net_third(self, x, t):
        u = self.net_u(x, t)
        u_x = tf.gradients(u, x)[0]

        third = ((1 / 2) * (u_x ** 2)) - (u ** 3)

        return third

    def callback(self, loss, lambda_1, lambda_2, lambda_3):
        print('Loss: %e, p1: %.5f, p2: %.5f, p3: %.5f' % (loss, lambda_1, lambda_2, lambda_3))
        self.iter = self.iter + 1
        print(self.iter)

        iter = self.iter

        if iter % 20 == 0:
            err_lambda_1_0 = np.abs(lambda_1 + 4) / 4  # -4
            err_lambda_2_0 = np.abs(lambda_2 + 2) / 2  # -2
            err_lambda_3_0 = np.abs(lambda_3) / 1

            loss_history_0.append(loss)
            iter_history_0.append(iter)
            l1_history_0.append(lambda_1)
            l2_history_0.append(lambda_2)
            l3_history_0.append(lambda_3)

            e1_history_0.append(err_lambda_1_0)
            e2_history_0.append(err_lambda_2_0)
            e3_history_0.append(err_lambda_3_0)

    def train(self, nIter):
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            if it % 20 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_1_value = self.sess.run(self.lambda_1)
                lambda_2_value = self.sess.run(self.lambda_2)
                lambda_3_value = self.sess.run(self.lambda_3)
                print('It: %d, Loss: %.3e, p_1: %.5f, p_2: %.5f, p_3: %.5f, Time: %.2f' %
                      (it, loss_value, lambda_1_value, lambda_2_value, lambda_3_value, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss, self.lambda_1, self.lambda_2, self.lambda_3],
                                loss_callback=self.callback)

    def predict(self, X_star):
        tf_dict = {self.x_tf: X_star[:, 0:1], self.t_tf: X_star[:, 1:2]}
        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star_1 = self.sess.run(self.f_pred_1, tf_dict)

        return u_star, f_star_1



####################################################################
#################### IntegNet ######################################
###################################################################

loss_history = []
iter_history = []
l1_history = []
l2_history = []
l3_history = []
e1_history = []
e2_history = []
e3_history = []


class IntegNet:
    def __init__(self, X, u, X_t1, X_t2, X_t3, X_t4, X_t5, X_t0, X_tt, layers, lb, ub):
        self.iter = 0
        self.lb = lb
        self.ub = ub
        self.a = 0.1
        self.x = X[:, 0:1]
        self.t = X[:, 1:2]
        self.u = u

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
        self.weights, self.biases = self.initialize_NN(layers)  # 需要构造新函数initialize_NN

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        # Initialize parameters
        self.lambda_1 = tf.Variable([1.0], dtype=tf.float32)  # ->-4
        self.lambda_2 = tf.Variable([1.0], dtype=tf.float32)  # ->-2
        self.lambda_3 = tf.Variable([1.0], dtype=tf.float32)  # ->0

        self.x_tf = tf.placeholder(tf.float32, shape=[None, self.x.shape[1]])  # 占位，等待输入训练值x
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])  # 占位，等待输入训练值t
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])  # 占位，等待输入训练值u

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

        self.u_pred = self.net_u(self.x_tf, self.t_tf)  # 神经网络u的预测值，  # 需要构造新函数net_u
        self.f_pred_1 = self.net_f(self.x_tf, self.t_tf)

        self.u_t1 = self.net_u(self.x_c_tf_1, self.t_c_tf_1)
        self.u_t2 = self.net_u(self.x_c_tf_2, self.t_c_tf_2)
        self.u_t3 = self.net_u(self.x_c_tf_3, self.t_c_tf_3)
        self.u_t4 = self.net_u(self.x_c_tf_4, self.t_c_tf_4)
        self.u_t5 = self.net_u(self.x_c_tf_5, self.t_c_tf_5)
        self.u_t0 = self.net_u(self.x_t0_tf, self.t_t0_tf)
        self.u_tt = self.net_u(self.x_tt_tf, self.t_tt_tf)

        self.third_t1 = self.net_third(self.x_c_tf_1, self.t_c_tf_1)
        self.third_t2 = self.net_third(self.x_c_tf_2, self.t_c_tf_2)
        self.third_t3 = self.net_third(self.x_c_tf_3, self.t_c_tf_3)
        self.third_t4 = self.net_third(self.x_c_tf_4, self.t_c_tf_4)
        self.third_t5 = self.net_third(self.x_c_tf_5, self.t_c_tf_5)
        self.third_t0 = self.net_third(self.x_t0_tf, self.t_t0_tf)
        self.third_tt = self.net_third(self.x_tt_tf, self.t_tt_tf)

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    0.2 * tf.reduce_mean(tf.square(self.f_pred_1)) + \
                    tf.reduce_mean(tf.reduce_mean(tf.square(tf.reduce_mean(self.u_t0) - tf.reduce_mean(self.u_t1)) + \
                                                  tf.square(tf.reduce_mean(self.u_t1) - tf.reduce_mean(self.u_t3)) +
                                                  tf.square(tf.reduce_mean(self.u_t3) - tf.reduce_mean(self.u_t5)) + \
                                                  tf.square(tf.reduce_mean(self.u_t5) - tf.reduce_mean(self.u_tt))) + \
                                   tf.reduce_mean(tf.square(tf.reduce_mean(self.u_t0 ** 2) - tf.reduce_mean(self.u_t1 ** 2)) + \
                                                  tf.square(tf.reduce_mean(self.u_t1 ** 2) - tf.reduce_mean(self.u_t3 ** 2)) +
                                                  tf.square(tf.reduce_mean(self.u_t3 ** 2) - tf.reduce_mean(self.u_t5 ** 2)) + \
                                                  tf.square(tf.reduce_mean(self.u_t5 ** 2) - tf.reduce_mean(self.u_tt ** 2))) + \
                                   tf.reduce_mean(tf.square(tf.reduce_mean(self.third_t0) - tf.reduce_mean(self.third_t1)) + \
                                                  tf.square(tf.reduce_mean(self.third_t1) - tf.reduce_mean(self.third_t3)) + \
                                                  tf.square(tf.reduce_mean(self.third_t3) - tf.reduce_mean(self.third_t5)) + \
                                                  tf.square(tf.reduce_mean(self.third_t5) - tf.reduce_mean(self.third_tt))))


        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 100000,
                                                                         'maxfun': 100000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps,
                                                                         'gtol': 1.0 * np.finfo(float).eps,
                                                                         }
                                                                )


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
        lambda_1 = self.lambda_1
        lambda_2 = self.lambda_2
        lambda_3 = self.lambda_3
        u = self.net_u(x, t)
        u_x = tf.gradients(u, x)[0]

        self.a = 1
        U_11 = 0
        U_12 = 1
        U_21 = self.a - u
        U_22 = 0
        V_12 = lambda_1 * self.a + lambda_2 * u + lambda_3 * u_x
        V_11 = - tf.gradients(V_12, x)[0] / 2
        V_21 = tf.gradients(V_11, x)[0] + V_12 * U_21
        V_22 = -V_11
        f3 = tf.gradients(U_21, t)[0] - tf.gradients(V_21, x)[0] - (V_21 * U_11 + V_22 * U_21) + (
                U_21 * V_11 + U_22 * V_21)
        return f3

    def net_third(self, x, t):
        u = self.net_u(x, t)
        u_x = tf.gradients(u, x)[0]

        third = ((1 / 2) * (u_x ** 2)) - (u ** 3)

        return third

    def callback(self, loss, lambda_1, lambda_2, lambda_3):
        print('Loss: %e, p1: %.5f, p2: %.5f, p3: %.5f' % (loss, lambda_1, lambda_2, lambda_3))
        self.iter = self.iter + 1
        print(self.iter)

        iter = self.iter

        if iter % 20 == 0:
            err_lambda_1 = np.abs(lambda_1 + 4) / 4  # -4
            err_lambda_2 = np.abs(lambda_2 + 2) / 2  # -2
            err_lambda_3 = np.abs(lambda_3) / 1

            loss_history.append(loss)
            iter_history.append(iter)
            l1_history.append(lambda_1)
            l2_history.append(lambda_2)
            l3_history.append(lambda_3)

            e1_history.append(err_lambda_1)
            e2_history.append(err_lambda_2)
            e3_history.append(err_lambda_3)

    def train(self, nIter):
        tf_dict = {self.x_tf: self.x, self.t_tf: self.t, self.u_tf: self.u,
                   self.x_c_tf_1: self.x_c_1, self.t_c_tf_1: self.t_c_1,
                   self.x_c_tf_2: self.x_c_2, self.t_c_tf_2: self.t_c_2,
                   self.x_c_tf_3: self.x_c_3, self.t_c_tf_3: self.t_c_3,
                   self.x_c_tf_4: self.x_c_4, self.t_c_tf_4: self.t_c_4,
                   self.x_c_tf_5: self.x_c_5, self.t_c_tf_5: self.t_c_5,
                   self.x_t0_tf: self.x_t0, self.t_t0_tf: self.t_t0,
                   self.x_tt_tf: self.x_tt, self.t_tt_tf: self.t_tt}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                lambda_1_value = self.sess.run(self.lambda_1)
                lambda_2_value = self.sess.run(self.lambda_2)
                lambda_3_value = self.sess.run(self.lambda_3)
                print('It: %d, Loss: %.3e, p_1: %.5f, p_2: %.5f, p_3: %.5f, Time: %.5f' %
                      (it, loss_value, lambda_1_value, lambda_2_value, lambda_3_value, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss, self.lambda_1, self.lambda_2, self.lambda_3],
                                loss_callback=self.callback)

    def predict(self, X_star):

        tf_dict = {self.x_tf: X_star[:, 0:1], self.t_tf: X_star[:, 1:2]}

        u_star = self.sess.run(self.u_pred, tf_dict)
        f_star_1 = self.sess.run(self.f_pred_1, tf_dict)

        return u_star, f_star_1


if __name__ == "__main__":
    N_u = 5000
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

    #################################################################
    ######################## Noiseless Data #########################
    #################################################################
    #noise1 = 0.0

    idx = np.random.choice(X_star.shape[0], N_u, replace=False)
    X_u_train = X_star[idx, :]
    u_train = u_star[idx, :]

    X_t0 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    X_tt = X_star[-256:, :]

    ##################################################################
    ######################## Noise Data ##############################
    ##################################################################
    noise2 = 0.01
    u_train2 = u_train + noise2 * np.std(u_train) * np.random.randn(u_train.shape[0], u_train.shape[1])

    noise3 = 0.05
    u_train3 = u_train + noise3 * np.std(u_train) * np.random.randn(u_train.shape[0], u_train.shape[1])

    #################################################################
    ################## Five Sections ################################
    #################################################################



    kn = 256 * 40
    X_t_1 = X_star[:kn, :]
    X_t_2 = X_star[kn: 2 * kn, :]
    X_t_3 = X_star[2 * kn:3 * kn, :]
    X_t_4 = X_star[3 * kn:4 * kn, :]
    X_t_5 = X_star[4 * kn:5 * kn, :]

    model1 = PhysicsInformedNN(X_u_train, u_train, layers, lb, ub)
    #model1_2 = PhysicsInformedNN(X_u_train, u_train2, layers, lb, ub)
    #model1_3 = PhysicsInformedNN(X_u_train, u_train3, layers, lb, ub)

    model2 = IntegNet(X_u_train, u_train, X_t_1, X_t_2, X_t_3, X_t_4, X_t_5, X_t0, X_tt, layers, lb, ub)
    #model2_2 = IntegNet(X_u_train, u_train2, X_t_1, X_t_2, X_t_3, X_t_4, X_t_5, X_t0, X_tt, layers, lb, ub)
    #model2_3 = IntegNet(X_u_train, u_train3, X_t_1, X_t_2, X_t_3, X_t_4, X_t_5, X_t0, X_tt, layers, lb, ub)


    start_time1 = time.time()
    model1.train(0)
    elapsed1 = time.time() - start_time1

    start_time2 = time.time()
    model2.train(0)
    elapsed2 = time.time() - start_time2

    total_time = elapsed1 + elapsed2
    print('Training time of PINN_noiseless: %.1f' % (elapsed1), 's = %.1f' % (elapsed1 / 60), 'min')
    print('Training time of IntegNet_noiseless: %.1f' % (elapsed2), 's = %.1f' % (elapsed2 / 60), 'min')
    print('Total training time : %.1f' % (total_time), 's = %.1f' % (total_time / 60), 'min')

    ######################## PINN results ###############################################

    u_pred_P1, f_pred_P1 = model1.predict(X_star)
    error_u_P1 = np.linalg.norm(u_star - u_pred_P1, 2) / np.linalg.norm(u_star, 2)
    lambda_1_value_P1 = model1.sess.run(model1.lambda_1)  #p1 -> -4
    lambda_2_value_P1 = model1.sess.run(model1.lambda_2)  #p2 -> -2
    lambda_3_value_P1 = model1.sess.run(model1.lambda_3)  #p3 -> 0
    error_lambda_1_P1 = np.abs(lambda_1_value_P1 + 4) / 4 * 100
    error_lambda_2_P1 = np.abs(lambda_2_value_P1 + 2) / 2 * 100
    error_lambda_3_P1 = np.abs(lambda_3_value_P1) / 1 * 100
    print('Error u of PINN_noiseless: %e' % error_u_P1)
    print('Error p1 of PINN_noiseless: %.5f%%' % error_lambda_1_P1)
    print('Error p2 of PINN_noiseless: %.5f%%' % error_lambda_2_P1)
    print('Error p3 of PINN_noiseless: %.5f%%' % error_lambda_3_P1)

    u_pred_P2, f_pred_P2 = model2.predict(X_star)
    error_u_P2 = np.linalg.norm(u_star - u_pred_P2, 2) / np.linalg.norm(u_star, 2)
    lambda_1_value_P2 = model2.sess.run(model2.lambda_1)
    lambda_2_value_P2 = model2.sess.run(model2.lambda_2)
    lambda_3_value_P2 = model2.sess.run(model2.lambda_3)
    error_lambda_1_P2 = np.abs(lambda_1_value_P2 + 4) / 4 * 100
    error_lambda_2_P2 = np.abs(lambda_2_value_P2 + 2) / 2 * 100
    error_lambda_3_P2 = np.abs(lambda_3_value_P2) / 1 * 100
    print('Error u of IntegNet_noiseless: %e' % error_u_P2)
    print('Error p1 of IntegNet_noiseless: %.5f%%' % error_lambda_1_P2)
    print('Error p2 of IntegNet_noiseless: %.5f%%' % error_lambda_2_P2)
    print('Error p3 of IntegNet_noiseless: %.5f%%' % error_lambda_3_P2)

    #################################################################################
    ################################## Plot #########################################
    #################################################################################

    plt.figure(1)
    plt.xlabel('iterations')
    plt.ylabel('value')

    l1 = plt.plot(iter_history, l1_history, color='blue', linewidth=1, label='p1')
    l2 = plt.plot(iter_history, l2_history, color='orange', linewidth=1, label='p2')
    l3 = plt.plot(iter_history, l3_history, color='green', linewidth=1, label='p3')
    plt.legend()


    plt.figure(2)
    plt.xlabel('iterations')
    plt.ylabel('loss')
    plt.yscale('log')

    #loss1 = plt.plot(iter_history_0, loss_history_0, color='black', linestyle='--', linewidth=1, label='PINN')
    loss2 = plt.plot(iter_history, loss_history, color='red', linewidth=1, label='IntegNet')
    plt.legend()


    plt.figure(3)
    plt.xlabel('iterations')
    plt.ylabel('error')

    e1 = plt.plot(iter_history, e1_history, color='blue', linewidth=1, label='error of p1')
    e2 = plt.plot(iter_history, e2_history, color='orange', linewidth=1, label='error of p2')
    e3 = plt.plot(iter_history, e3_history, color='green', linewidth=1, label='error of p3')
    plt.legend()
    plt.yscale('log')


    plt.show()
